// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>

#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"
#include "common/sync_trace.h"

namespace Libraries::Kernel {

// GR2FORK FIX: per-thread reader-recursion counts. MSVC's std::shared_timed_mutex is
// non-recursive and write-preferring: a recursive Rdlock deadlocks once a writer queues, while
// POSIX allows it and GR2 (Havok) relies on it. Recursion bumps a TLS count instead of locking.

// GR2FORK PERF: the TLS counts live in an inline slots[kInlineRwlockSlots] array, not a
// thread_local unordered_map whose per-op find costs ~0.09% (+~0.02% allocator) at 10K-100K
// ops/sec; threads rarely hold >3 read locks, and >8 distinct rwlocks falls back to a heap map.

namespace {

constexpr size_t kInlineRwlockSlots = 8;

struct TlsRwlockSlot {
    PthreadRwlock* rwlock;  // nullptr => empty
    int count;
};

struct TlsRwlockState {
    TlsRwlockSlot slots[kInlineRwlockSlots]{};
    // Overflow fallback. Allocated lazily only if a thread holds
    // >kInlineRwlockSlots distinct rwlocks at once.
    std::unordered_map<PthreadRwlock*, int>* overflow{nullptr};

    ~TlsRwlockState() { delete overflow; }
};

thread_local TlsRwlockState g_tls_rwlock_state;

// If `rw` is already held (count > 0) by this thread, increments the count
// and returns true. Otherwise returns false (caller must do real lock).
inline bool TlsReadCountTryRecursive(PthreadRwlock* rw) {
    auto& s = g_tls_rwlock_state;
    for (auto& slot : s.slots) {
        if (slot.rwlock == rw) {
            // Found in inline slots. count is guaranteed > 0 (we clear on
            // outermost release).
            slot.count++;
            return true;
        }
    }
    if (s.overflow) [[unlikely]] {
        auto it = s.overflow->find(rw);
        if (it != s.overflow->end() && it->second > 0) {
            it->second++;
            return true;
        }
    }
    return false;
}

// Initializes a fresh entry for `rw` with count = 1. Called after a real
// (first) acquire of the underlying shared_timed_mutex.
inline void TlsReadCountInitialize(PthreadRwlock* rw) {
    auto& s = g_tls_rwlock_state;
    // Try inline slots first.
    for (auto& slot : s.slots) {
        if (slot.rwlock == nullptr) {
            slot.rwlock = rw;
            slot.count = 1;
            return;
        }
    }
    // All inline slots full - overflow.
    if (!s.overflow) [[unlikely]] {
        s.overflow = new std::unordered_map<PthreadRwlock*, int>();
    }
    (*s.overflow)[rw] = 1;
}

// Decrements the count for `rw`. Returns true if this is the OUTERMOST
// release (count went to 0; caller must do real unlock_shared). Returns
// false on inner release (count still > 0; caller does nothing).
inline bool TlsReadCountRelease(PthreadRwlock* rw) {
    auto& s = g_tls_rwlock_state;
    for (auto& slot : s.slots) {
        if (slot.rwlock == rw) {
            if (--slot.count == 0) {
                slot.rwlock = nullptr;  // free the slot
                return true;
            }
            return false;
        }
    }
    if (s.overflow) [[unlikely]] {
        auto it = s.overflow->find(rw);
        if (it != s.overflow->end()) {
            if (--it->second == 0) {
                s.overflow->erase(it);
                return true;
            }
            return false;
        }
    }
    // Not tracked - defensive: assume outermost release. This matches the
    // original behavior where a missing entry would fall through to the
    // unlock_shared call.
    return true;
}

} // namespace

static std::mutex RwlockStaticLock;

#define THR_RWLOCK_INITIALIZER ((PthreadRwlock*)NULL)
#define THR_RWLOCK_DESTROYED ((PthreadRwlock*)1)

#define CHECK_AND_INIT_RWLOCK                                                                      \
    if (prwlock = (*rwlock); prwlock <= THR_RWLOCK_DESTROYED) [[unlikely]] {                       \
        if (prwlock == THR_RWLOCK_INITIALIZER) {                                                   \
            int ret;                                                                               \
            ret = InitStatic(g_curthread, rwlock);                                                 \
            if (ret)                                                                               \
                return (ret);                                                                      \
        } else if (prwlock == THR_RWLOCK_DESTROYED) {                                              \
            return POSIX_EINVAL;                                                                   \
        }                                                                                          \
        prwlock = *rwlock;                                                                         \
    }

static int RwlockInit(PthreadRwlockT* rwlock, const PthreadRwlockAttrT* attr) {
    auto* prwlock = new (std::nothrow) PthreadRwlock{};
    if (prwlock == nullptr) {
        return POSIX_ENOMEM;
    }
    *rwlock = prwlock;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlock_destroy(PthreadRwlockT* rwlock) {
    PthreadRwlockT prwlock = *rwlock;
    if (prwlock == THR_RWLOCK_INITIALIZER) {
        return 0;
    }
    if (prwlock == THR_RWLOCK_DESTROYED) {
        return POSIX_EINVAL;
    }
    *rwlock = THR_RWLOCK_DESTROYED;
    delete prwlock;
    return 0;
}

static int InitStatic(Pthread* thread, PthreadRwlockT* rwlock) {
    std::scoped_lock lk{RwlockStaticLock};
    if (*rwlock == THR_RWLOCK_INITIALIZER) {
        return RwlockInit(rwlock, nullptr);
    }
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlock_init(PthreadRwlockT* rwlock, const PthreadRwlockAttrT* attr) {
    *rwlock = nullptr;
    return RwlockInit(rwlock, attr);
}

int PthreadRwlock::Rdlock(const OrbisKernelTimespec* abstime) {
    Pthread* curthread = g_curthread;

    // GR2FORK FIX: recursive read-lock shortcut. If this thread already
    // holds a read lock on `this`, just bump the TLS count - don't go through
    // std::shared_timed_mutex (which would deadlock when a writer is queued).
    if (TlsReadCountTryRecursive(this)) {
        curthread->rdlock_count++;
        return 0;
    }

    /*
     * POSIX said the validity of the abstimeout parameter need
     * not be checked if the lock can be immediately acquired.
     */
    if (lock.try_lock_shared()) {
        curthread->rdlock_count++;
        TlsReadCountInitialize(this);
        Common::SyncTrace::RwRdlockAcquired(this);
        return 0;
    }
    if (abstime && (abstime->tv_nsec >= 1000000000 || abstime->tv_nsec < 0)) [[unlikely]] {
        return POSIX_EINVAL;
    }

    Common::SyncTrace::RwWaitBegin(this, /*want_write=*/false);

    // Note: On interruption an attempt to relock the mutex is made.
    if (abstime != nullptr) {
        if (!lock.try_lock_shared_until(abstime->TimePoint())) {
            Common::SyncTrace::RwWaitEnd(this);
            return POSIX_ETIMEDOUT;
        }
    } else {
        lock.lock_shared();
    }

    Common::SyncTrace::RwWaitEnd(this);
    Common::SyncTrace::RwRdlockAcquired(this);
    TlsReadCountInitialize(this);
    curthread->rdlock_count++;
    return 0;
}

int PthreadRwlock::Wrlock(const OrbisKernelTimespec* abstime) {
    Pthread* curthread = g_curthread;

    /*
     * POSIX said the validity of the abstimeout parameter need
     * not be checked if the lock can be immediately acquired.
     */
    if (lock.try_lock()) {
        owner = curthread;
        // GR2FORK FIX: track write-lock acquisition.
        Common::SyncTrace::RwWrlockAcquired(this);
        return 0;
    }

    if (abstime && (abstime->tv_nsec >= 1000000000 || abstime->tv_nsec < 0)) {
        return POSIX_EINVAL;
    }

    // GR2FORK FIX: track that we're blocking on the writer slot.
    Common::SyncTrace::RwWaitBegin(this, /*want_write=*/true);

    // Note: On interruption an attempt to relock the mutex is made.
    if (abstime != nullptr) {
        if (!lock.try_lock_until(abstime->TimePoint())) {
            Common::SyncTrace::RwWaitEnd(this);
            return POSIX_ETIMEDOUT;
        }
    } else {
        lock.lock();
    }

    Common::SyncTrace::RwWaitEnd(this);
    Common::SyncTrace::RwWrlockAcquired(this);
    owner = curthread;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlock_rdlock(PthreadRwlockT* rwlock) {
    PthreadRwlockT prwlock{};
    CHECK_AND_INIT_RWLOCK
    return prwlock->Rdlock(nullptr);
}

int PS4_SYSV_ABI posix_pthread_rwlock_timedrdlock(PthreadRwlockT* rwlock,
                                                  const OrbisKernelTimespec* abstime) {
    PthreadRwlockT prwlock{};
    CHECK_AND_INIT_RWLOCK
    return prwlock->Rdlock(abstime);
}

int PS4_SYSV_ABI posix_pthread_rwlock_tryrdlock(PthreadRwlockT* rwlock) {
    Pthread* curthread = g_curthread;
    PthreadRwlockT prwlock{};
    CHECK_AND_INIT_RWLOCK

    // GR2FORK FIX: recursive read-lock shortcut (same reason as Rdlock).
    if (TlsReadCountTryRecursive(prwlock)) {
        curthread->rdlock_count++;
        return 0;
    }

    if (!prwlock->lock.try_lock_shared()) {
        return POSIX_EBUSY;
    }

    Common::SyncTrace::RwRdlockAcquired(prwlock);
    TlsReadCountInitialize(prwlock);
    curthread->rdlock_count++;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlock_trywrlock(PthreadRwlockT* rwlock) {
    Pthread* curthread = g_curthread;
    PthreadRwlockT prwlock{};
    CHECK_AND_INIT_RWLOCK

    if (!prwlock->lock.try_lock()) {
        return POSIX_EBUSY;
    }
    prwlock->owner = curthread;
    // GR2FORK FIX: track successful trywrlock acquisition.
    Common::SyncTrace::RwWrlockAcquired(prwlock);
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlock_wrlock(PthreadRwlockT* rwlock) {
    PthreadRwlockT prwlock{};
    CHECK_AND_INIT_RWLOCK
    return prwlock->Wrlock(nullptr);
}

int PS4_SYSV_ABI posix_pthread_rwlock_timedwrlock(PthreadRwlockT* rwlock,
                                                  const OrbisKernelTimespec* abstime) {
    PthreadRwlockT prwlock{};
    CHECK_AND_INIT_RWLOCK
    return prwlock->Wrlock(abstime);
}

int PS4_SYSV_ABI posix_pthread_rwlock_unlock(PthreadRwlockT* rwlock) {
    Pthread* curthread = g_curthread;
    PthreadRwlockT prwlock = *rwlock;
    if (prwlock <= THR_RWLOCK_DESTROYED) [[unlikely]] {
        return POSIX_EINVAL;
    }

    if (prwlock->owner == curthread) {
        prwlock->owner = nullptr;
        Common::SyncTrace::RwWrlockReleased(prwlock);
        prwlock->lock.unlock();
    } else {
        if (prwlock->owner == nullptr) {
            curthread->rdlock_count--;
        }
        // GR2FORK FIX: recursive read-lock counting - unlock_shared only on the outermost
        // release; inner releases decrement the TLS counter (inline pointer-array scan, no
        // hashmap, no allocation).
        if (!TlsReadCountRelease(prwlock)) {
            return 0; // inner release; don't touch the underlying mutex
        }
        Common::SyncTrace::RwRdlockReleased(prwlock);
        prwlock->lock.unlock_shared();
    }

    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlockattr_destroy(PthreadRwlockAttrT* rwlockattr) {
    if (rwlockattr == nullptr) {
        return POSIX_EINVAL;
    }
    PthreadRwlockAttrT prwlockattr = *rwlockattr;
    if (prwlockattr == nullptr) {
        return POSIX_EINVAL;
    }

    delete prwlockattr;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlockattr_getpshared(const PthreadRwlockAttrT* rwlockattr,
                                                     int* pshared) {
    *pshared = (*rwlockattr)->pshared;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlockattr_init(PthreadRwlockAttrT* rwlockattr) {
    if (rwlockattr == nullptr) {
        return POSIX_EINVAL;
    }

    auto prwlockattr = new (std::nothrow) PthreadRwlockAttr{};
    if (prwlockattr == nullptr) {
        return POSIX_ENOMEM;
    }

    prwlockattr->pshared = 0;
    *rwlockattr = prwlockattr;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_rwlockattr_setpshared(PthreadRwlockAttrT* rwlockattr, int pshared) {
    /* Only PTHREAD_PROCESS_PRIVATE is supported. */
    if (pshared != 0) {
        return POSIX_EINVAL;
    }

    (*rwlockattr)->pshared = pshared;
    return 0;
}

void RegisterRwlock(Core::Loader::SymbolsResolver* sym) {
    // Posix-Kernel
    LIB_FUNCTION("1471ajPzxh0", "libkernel", 1, "libkernel", posix_pthread_rwlock_destroy);
    LIB_FUNCTION("ytQULN-nhL4", "libkernel", 1, "libkernel", posix_pthread_rwlock_init);
    LIB_FUNCTION("iGjsr1WAtI0", "libkernel", 1, "libkernel", posix_pthread_rwlock_rdlock);
    LIB_FUNCTION("lb8lnYo-o7k", "libkernel", 1, "libkernel", posix_pthread_rwlock_timedrdlock);
    LIB_FUNCTION("9zklzAl9CGM", "libkernel", 1, "libkernel", posix_pthread_rwlock_timedwrlock);
    LIB_FUNCTION("SFxTMOfuCkE", "libkernel", 1, "libkernel", posix_pthread_rwlock_tryrdlock);
    LIB_FUNCTION("XhWHn6P5R7U", "libkernel", 1, "libkernel", posix_pthread_rwlock_trywrlock);
    LIB_FUNCTION("EgmLo6EWgso", "libkernel", 1, "libkernel", posix_pthread_rwlock_unlock);
    LIB_FUNCTION("sIlRvQqsN2Y", "libkernel", 1, "libkernel", posix_pthread_rwlock_wrlock);
    LIB_FUNCTION("qsdmgXjqSgk", "libkernel", 1, "libkernel", posix_pthread_rwlockattr_destroy);
    LIB_FUNCTION("VqEMuCv-qHY", "libkernel", 1, "libkernel", posix_pthread_rwlockattr_getpshared);
    LIB_FUNCTION("xFebsA4YsFI", "libkernel", 1, "libkernel", posix_pthread_rwlockattr_init);
    LIB_FUNCTION("OuKg+kRDD7U", "libkernel", 1, "libkernel", posix_pthread_rwlockattr_setpshared);

    // Posix
    LIB_FUNCTION("1471ajPzxh0", "libScePosix", 1, "libkernel", posix_pthread_rwlock_destroy);
    LIB_FUNCTION("ytQULN-nhL4", "libScePosix", 1, "libkernel", posix_pthread_rwlock_init);
    LIB_FUNCTION("iGjsr1WAtI0", "libScePosix", 1, "libkernel", posix_pthread_rwlock_rdlock);
    LIB_FUNCTION("lb8lnYo-o7k", "libScePosix", 1, "libkernel", posix_pthread_rwlock_timedrdlock);
    LIB_FUNCTION("9zklzAl9CGM", "libScePosix", 1, "libkernel", posix_pthread_rwlock_timedwrlock);
    LIB_FUNCTION("SFxTMOfuCkE", "libScePosix", 1, "libkernel", posix_pthread_rwlock_tryrdlock);
    LIB_FUNCTION("XhWHn6P5R7U", "libScePosix", 1, "libkernel", posix_pthread_rwlock_trywrlock);
    LIB_FUNCTION("EgmLo6EWgso", "libScePosix", 1, "libkernel", posix_pthread_rwlock_unlock);
    LIB_FUNCTION("sIlRvQqsN2Y", "libScePosix", 1, "libkernel", posix_pthread_rwlock_wrlock);
    LIB_FUNCTION("qsdmgXjqSgk", "libScePosix", 1, "libkernel", posix_pthread_rwlockattr_destroy);
    LIB_FUNCTION("VqEMuCv-qHY", "libScePosix", 1, "libkernel", posix_pthread_rwlockattr_getpshared);
    LIB_FUNCTION("xFebsA4YsFI", "libScePosix", 1, "libkernel", posix_pthread_rwlockattr_init);
    LIB_FUNCTION("OuKg+kRDD7U", "libScePosix", 1, "libkernel", posix_pthread_rwlockattr_setpshared);

    // Orbis
    LIB_FUNCTION("i2ifZ3fS2fo", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_rwlockattr_destroy));
    LIB_FUNCTION("LcOZBHGqbFk", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_rwlockattr_getpshared));
    LIB_FUNCTION("yOfGg-I1ZII", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlockattr_init));
    LIB_FUNCTION("-ZvQH18j10c", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_rwlockattr_setpshared));
    LIB_FUNCTION("BB+kb08Tl9A", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlock_destroy));
    LIB_FUNCTION("6ULAa0fq4jA", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlock_init));
    LIB_FUNCTION("Ox9i0c7L5w0", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlock_rdlock));
    LIB_FUNCTION("iPtZRWICjrM", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_rwlock_timedrdlock));
    LIB_FUNCTION("adh--6nIqTk", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_rwlock_timedwrlock));
    LIB_FUNCTION("XD3mDeybCnk", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlock_tryrdlock));
    LIB_FUNCTION("bIHoZCTomsI", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlock_trywrlock));
    LIB_FUNCTION("+L98PIbGttk", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlock_unlock));
    LIB_FUNCTION("mqdNorrB+gI", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rwlock_wrlock));
}

} // namespace Libraries::Kernel
