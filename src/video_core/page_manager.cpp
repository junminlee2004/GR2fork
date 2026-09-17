// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <mutex>
#include <thread>
#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/range_lock.h"
#include "common/signal_context.h"
#include "core/address_space.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

constexpr size_t PM_PAGE_SIZE = 4_KB;
constexpr size_t PM_PAGE_BITS = 12;

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers : 7;
        // At the moment only buffer cache can request read watchers.
        // And buffers cannot overlap, thus only 1 can exist per page.
        u8 num_read_watchers : 1;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <s32 delta, bool is_read>
        u8 AddDelta() {
            if constexpr (is_read) {
                if constexpr (delta == 1) {
                    return ++num_read_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                    return --num_read_watchers;
                } else {
                    return num_read_watchers;
                }
            } else {
                if constexpr (delta == 1) {
                    return ++num_write_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                    return --num_write_watchers;
                } else {
                    return num_write_watchers;
                }
            }
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / PAGES_PER_LOCK;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() = default;
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        // No-op
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    template <bool track, bool is_read>
    u32 UpdatePageWatchers(VAddr addr, u64 size) {
        RENDERER_TRACE;
        u32 calls = 0;

        size_t page = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        // The range lock below spins until it owns every lock in the range; a
        // carry held by this thread would be one of them and never yield.
        FlushCarry();

        // Acquire locks for the range of pages
        const auto lock_start = locks.begin() + (page / PAGES_PER_LOCK);
        const auto lock_end = locks.begin() + Common::DivCeil(page_end, PAGES_PER_LOCK);
        Common::RangeLockGuard lk(lock_start, lock_end);

        auto perms = cached_pages[page].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // Perform pending (un)protect action
                Protect(range_begin << PM_PAGE_BITS, range_bytes, perms);
                ++calls;
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
        }

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];

            // Apply the change to the page state
            const u8 new_count = state.AddDelta<track ? 1 : -1, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // Only start a new range if the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current range up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
        return calls;
    }

    template <bool track, bool is_read>
    u32 UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) {
        RENDERER_TRACE;
        u32 calls = 0;
        auto start_range = mask.FirstRange();
        auto end_range = mask.LastRange();

        if (start_range.second == end_range.second && carry_depth_ == 0) {
            // if all pages are contiguous, use the regular UpdatePageWatchers
            const VAddr start_addr = base_addr + (start_range.first << PM_PAGE_BITS);
            const u64 size = (start_range.second - start_range.first) << PM_PAGE_BITS;
            return UpdatePageWatchers<track, is_read>(start_addr, size);
        }

        size_t base_page = (base_addr >> PM_PAGE_BITS);
        ASSERT(base_page % PAGES_PER_LOCK == 0);
        // A carry that does not end where this region begins can never merge
        // with it, and its lock must go before a second one is requested.
        if (carry_.active && carry_.begin + carry_.bytes != base_addr) {
            FlushCarry();
        }
        std::unique_lock<LockType> lk(locks[base_page / PAGES_PER_LOCK]);
        auto perms = cached_pages[base_page + start_range.first].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // A carry from the previous region that abuts this run with the
                // same permissions becomes one call instead of two.
                if (carry_.active) {
                    if (carry_.perms == perms &&
                        carry_.begin + carry_.bytes == (range_begin << PM_PAGE_BITS)) {
                        Protect(carry_.begin, carry_.bytes + range_bytes, perms);
                        carry_merged_.fetch_add(1, std::memory_order_relaxed);
                        carry_.active = false;
                        carry_.lock.unlock();
                        ++calls;
                        range_bytes = 0;
                        potential_range_bytes = 0;
                        return;
                    }
                    FlushCarry();
                }
                // Perform pending (un)protect action
                Protect((range_begin << PM_PAGE_BITS), range_bytes, perms);
                ++calls;
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate pages
        for (size_t page = start_range.first; page < end_range.second; ++page) {
            PageState& state = cached_pages[base_page + page];
            const bool update = mask.Get(page);

            // Apply the change to the page state
            const u8 new_count =
                update ? state.AddDelta<track ? 1 : -1, is_read>() : state.AddDelta<0, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // If the page is not being updated, skip it
            if (!update) {
                continue;
            }

            // If the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current rango up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // A trailing run that reaches the region boundary may continue into the
        // next region: keep the lock and the run, and let the next call merge.
        if (carry_depth_ != 0 && !carry_.active && range_bytes > 0 &&
            (range_begin << PM_PAGE_BITS) + range_bytes == base_addr + TRACKER_HIGHER_PAGE_SIZE) {
            carry_ =
                ProtectCarry{range_begin << PM_PAGE_BITS, range_bytes, perms, std::move(lk), true};
            return calls;
        }
        // Add pending (un)protect action
        release_pending();
        // No run absorbed the carry; it must not outlive this call.
        FlushCarry();
        return calls;
    }

    std::array<PageState, NUM_ADDRESS_PAGES> cached_pages{};
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
    using LockType = Common::AdaptiveMutex;
#else
    using LockType = Common::SpinLock;
#endif
    std::array<LockType, NUM_ADDRESS_LOCKS> locks{};

    // --- protect_carry_merge ---
    // A run that ends exactly on a region boundary keeps its region lock and
    // defers its mprotect; the next region's leading run with the same perms is
    // then issued as one cross-region call. Everything here is thread_local, so
    // a thread that never opens a scope behaves exactly as before.
    //
    // LOCK ORDER. A held carry adds one new edge: page-manager lock N is held
    // while the enclosing loop takes the RegionManager lock of the next region.
    // That is acyclic only because (a) every RegionManager protects only its own
    // region (region_manager.h ArmReadWatchers/ReleaseReadWatchers/UpdateProtection
    // all pass their own cpu_addr), so an RM_X holder never asks for PM lock
    // Y != X; (b) the only holder of two RM locks at once is
    // ForEachUploadRange<is_written=true>, which cannot be concurrent with a
    // carry because carries are opened only through an explicit GpuComm-only
    // flag at the named sites -- Rasterizer::DrainPendingReadArms for every
    // ReadArmSite but Submit (vk_rasterizer.h), and, inside
    // BufferCache::FinishFaultDownload (GpuComm via SendCommand<true>), its
    // PendingUnmark and its DrainPendingReadReleases -- while every other
    // caller, the guest-thread DropPendingReadArms unmap route and the submit
    // hook included, passes false; that upload walk runs on the same GPU
    // command thread, and both read drains bail on upload_walk_depth_ != 0;
    // (c) Common::RangeLockGuard (the multi-PM-lock holder in the contiguous
    // UpdatePageWatchers) is try-lock with back-off, so it never blocks holding
    // one of the locks in its range. Opening a scope on any other thread, or
    // around a loop that calls anything but UpdatePageWatchersForRegion, breaks
    // (b) and deadlocks; BeginCarry asserts the single carrying thread. A carry is also never
    // handed from one region walk to a non-adjacent one: it is flushed before a different lock is
    // taken. Aggregate without default member initializers on purpose: carry_ below is
    // value-initialized inside this class definition, where they would not be
    // usable yet.
    struct ProtectCarry {
        VAddr begin;
        u64 bytes;
        Core::MemoryPermission perms;
        std::unique_lock<LockType> lock;
        bool active;
    };
    const bool carry_enabled_{EmulatorSettings.IsProtectCarryMerge()};
    std::atomic<u64> carry_thread_{0};
    static inline thread_local u32 carry_depth_{0};
    static inline thread_local ProtectCarry carry_{};
    std::atomic<u64> carry_scopes_{0};
    std::atomic<u64> carry_merged_{0};
    std::atomic<u64> carry_flushed_{0};

    void FlushCarry() {
        if (!carry_.active) {
            return;
        }
        Protect(carry_.begin, carry_.bytes, carry_.perms);
        carry_flushed_.fetch_add(1, std::memory_order_relaxed);
        carry_.active = false;
        carry_.lock.unlock();
    }

    void BeginCarry() {
        if (!carry_enabled_) {
            return;
        }
        if (carry_depth_++ == 0) {
            // Once per scope, never per run: catch a future caller that opens a
            // carry off the GPU command thread, which fact (b) above forbids.
            const u64 self = std::hash<std::thread::id>{}(std::this_thread::get_id()) | 1;
            u64 expected = 0;
            if (!carry_thread_.compare_exchange_strong(expected, self, std::memory_order_relaxed)) {
                ASSERT_MSG(expected == self, "Protect carry opened on a second thread");
            }
            carry_scopes_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void EndCarry() {
        if (carry_depth_ == 0) {
            return;
        }
        if (--carry_depth_ == 0) {
            FlushCarry();
        }
    }
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed");
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed");
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        // This path changes protection without going through AddressSpace, so
        // it notifies the epoch observer itself.
        Core::TrackerProtectScope tracker_scope;
        Core::NotifyProtectObserver(address, size, allow_write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? 0 : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            ASSERT_MSG(readret != -1 || errno == EAGAIN, "Unexpected result of uffd read");
            if (errno == EAGAIN) {
                continue;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            rasterizer->InvalidateMemory(addr, 1);
        }
    }
};
#endif // __linux__

struct SignalImpl : public PageManager::Impl {
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        Core::TrackerProtectScope tracker_scope;
        impl.Protect(address, size, perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        if (Common::IsWriteError(context)) {
            return rasterizer->InvalidateMemory(addr, 8);
        } else {
            return rasterizer->ReadMemory(addr, 8);
        }
        return false;
    }
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        // Write-protect faults only: the readback path needs the read faults the
        // signal implementation delivers, so this fork keeps signals.
        LOG_WARNING(Config, "userfaultfd tracking is not used: readbacks need read faults");
    }
    if (false) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

void PageManager::BeginProtectCarry() const {
    impl->BeginCarry();
}

void PageManager::EndProtectCarry() const {
    impl->EndCarry();
}

PageManager::ProtectCarryStats PageManager::DrainProtectCarryStats() const {
    return ProtectCarryStats{
        impl->carry_scopes_.exchange(0, std::memory_order_relaxed),
        impl->carry_merged_.exchange(0, std::memory_order_relaxed),
        impl->carry_flushed_.exchange(0, std::memory_order_relaxed),
    };
}

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

bool PageManager::IsWriteWatched(VAddr addr, u64 size) const {
    if (size == 0) {
        return false;
    }
    const u64 first_page = addr >> PM_PAGE_BITS;
    if (first_page >= Impl::NUM_ADDRESS_PAGES) {
        // cached_pages only covers the low 40 bits of the address space; a
        // packet pointing above it is not tracked here at all.
        return false;
    }
    const u64 end_page =
        std::min<u64>(Common::DivCeil(addr + size, PM_PAGE_SIZE), Impl::NUM_ADDRESS_PAGES);
    for (u64 page = first_page; page < end_page; ++page) {
        if (impl->cached_pages[page].num_write_watchers != 0) {
            return true;
        }
    }
    return false;
}

template <bool track>
void PageManager::UpdatePageWatchers(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<track, false>(addr, size);
}

template <bool track, bool is_read>
u32 PageManager::UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const {
    return impl->UpdatePageWatchersForRegion<track, is_read>(base_addr, mask);
}

template void PageManager::UpdatePageWatchers<true>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchers<false>(VAddr addr, u64 size) const;
template u32 PageManager::UpdatePageWatchersForRegion<true, true>(VAddr base_addr,
                                                                  RegionBits& mask) const;
template u32 PageManager::UpdatePageWatchersForRegion<true, false>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template u32 PageManager::UpdatePageWatchersForRegion<false, true>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template u32 PageManager::UpdatePageWatchersForRegion<false, false>(VAddr base_addr,
                                                                    RegionBits& mask) const;

} // namespace VideoCore
