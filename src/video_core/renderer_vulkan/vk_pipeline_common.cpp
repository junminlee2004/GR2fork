// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <boost/container/static_vector.hpp>
#include "common/assert.h"
#include "common/logging/log.h"

#include "core/emulator_settings.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/skipcache/skipcache.h"

namespace Vulkan {

namespace Skipcache = VideoCore::Skipcache;

namespace {

// The four key header fields are contiguous, and so are the payload fields of
// every info struct, so each run copies as raw bytes. An image info is 24
// deterministic bytes: the rasterizer's AppendImageInfo zeroes the four tail
// padding bytes after construction, so copies and compares take the struct whole.
// Offsets are checked on the C structs because MSVC rejects offsetof on the
// vulkan.hpp wrappers; the wrappers are what gets indexed, so their strides are
// checked against the C ones.
static_assert(offsetof(VkWriteDescriptorSet, dstArrayElement) ==
                  offsetof(VkWriteDescriptorSet, dstBinding) + 4 &&
              offsetof(VkWriteDescriptorSet, descriptorCount) ==
                  offsetof(VkWriteDescriptorSet, dstBinding) + 8 &&
              offsetof(VkWriteDescriptorSet, descriptorType) ==
                  offsetof(VkWriteDescriptorSet, dstBinding) + 12);
static_assert(sizeof(vk::DescriptorBufferInfo) == sizeof(VkDescriptorBufferInfo) &&
              sizeof(VkDescriptorBufferInfo) == 24 &&
              offsetof(VkDescriptorBufferInfo, buffer) == 0 &&
              offsetof(VkDescriptorBufferInfo, offset) == 8 &&
              offsetof(VkDescriptorBufferInfo, range) == 16);
static_assert(sizeof(vk::DescriptorImageInfo) == sizeof(VkDescriptorImageInfo) &&
              sizeof(VkDescriptorImageInfo) == 24 &&
              offsetof(VkDescriptorImageInfo, sampler) == 0 &&
              offsetof(VkDescriptorImageInfo, imageView) == 8 &&
              offsetof(VkDescriptorImageInfo, imageLayout) == 16);
static_assert(sizeof(vk::BufferView) == 8);
static_assert(sizeof(Skipcache::Framework::DescDeltaSlot::write_changed) >=
              Pipeline::NUM_DESCRIPTOR_WRITES);

// Serialize a descriptor write list into a deterministic byte stream, payload
// contents included. Returns 0 on any unknown descriptor type or overflow:
// unknowns fail toward the slow path.
size_t SerializeDescriptorWrites(std::span<const vk::WriteDescriptorSet> writes,
                                 std::array<u8, 16384>& out) {
    u8* cursor = out.data();
    const u8* const limit = out.data() + out.size();
    const auto put = [&](const void* p, size_t n) {
        std::memcpy(cursor, p, n);
        cursor += n;
    };
    for (const auto& w : writes) {
        const u32 count = w.descriptorCount;
        // Bail where the overflow would happen rather than on a whole-list
        // worst case: 24 bounds every payload, so this can only overestimate.
        if (static_cast<size_t>(limit - cursor) < 16 + size_t{count} * 24) {
            return 0;
        }
        const VkWriteDescriptorSet& raw = w;
        put(&raw.dstBinding, 16);
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer: {
            const auto* const infos = w.pBufferInfo;
            for (u32 i = 0; i < count; ++i) {
                put(&infos[i], 24);
            }
            break;
        }
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampler: {
            const auto* const infos = w.pImageInfo;
            for (u32 i = 0; i < count; ++i) {
                put(&infos[i], 24);
            }
            break;
        }
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer: {
            const auto* const views = w.pTexelBufferView;
            for (u32 i = 0; i < count; ++i) {
                put(&views[i], 8);
            }
            break;
        }
        default:
            return 0;
        }
    }
    return static_cast<size_t>(cursor - out.data());
}

// XOR-OR of two byte ranges, 8 bytes at a time; bytes is a multiple of 8.
// Accumulating instead of exiting early keeps this a plain vector reduction
// rather than a library compare.
u64 DiffWords(const void* a, const void* b, size_t bytes) noexcept {
    const u8* pa = static_cast<const u8*>(a);
    const u8* pb = static_cast<const u8*>(b);
    u64 acc = 0;
    for (size_t i = 0; i < bytes; i += 8) {
        u64 x, y;
        std::memcpy(&x, pa + i, 8);
        std::memcpy(&y, pb + i, 8);
        acc |= x ^ y;
    }
    return acc;
}

template <size_t N>
u64 DiffFixed(const void* a, const void* b) noexcept {
    static_assert(N % 8 == 0);
    const u8* pa = static_cast<const u8*>(a);
    const u8* pb = static_cast<const u8*>(b);
    u64 acc = 0;
    for (size_t i = 0; i < N; i += 8) {
        u64 x, y;
        std::memcpy(&x, pa + i, 8);
        std::memcpy(&y, pb + i, 8);
        acc |= x ^ y;
    }
    return acc;
}

// One descriptor of a compile-time stride, copied without ever forming the
// runtime-length memcpy the walk must not call on its hit path.
template <size_t N>
void CopyFixed(void* dst, const void* src) noexcept {
#if defined(__clang__)
    __builtin_memcpy_inline(dst, src, N);
#else
    std::memcpy(dst, src, N);
#endif
}

// Compares n 24-byte descriptors at src with dst, leaves dst equal to src and
// returns a bit per descriptor that differed (bit i = descriptor i). n < 64.
// Outlined: a leaf with two call sites inside BindResources' frame would
// otherwise be inlined into a function with no registers to spare.
SHAD_NO_INLINE u64 SyncDescriptors24(const u8* src, u8* dst, u32 n) noexcept {
    u64 mask = 0;
    u32 i = 0;
#if defined(__AVX2__)
    for (; i + 4 <= n; i += 4, src += 96, dst += 96) {
        const __m256i s0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        const __m256i s1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
        const __m256i s2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 64));
        const __m256i d0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst));
        const __m256i d1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + 32));
        const __m256i d2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + 64));
        const u32 eq =
            static_cast<u32>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(s0, d0)))) |
            (static_cast<u32>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(s1, d1))))
             << 4) |
            (static_cast<u32>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(s2, d2))))
             << 8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), s0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 32), s1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 64), s2);
        // Qword k belongs to descriptor k / 3: fold each triple onto its
        // first bit, then gather bits 0, 3, 6, 9.
        u32 ne = ~eq & 0xfffu;
        ne |= (ne >> 1) | (ne >> 2);
#if defined(__BMI2__)
        const u32 bits = _pext_u32(ne, 0x249u);
#else
        const u32 bits = (ne & 1u) | ((ne >> 2) & 2u) | ((ne >> 4) & 4u) | ((ne >> 6) & 8u);
#endif
        mask |= u64{bits} << i;
    }
#endif
    for (; i < n; ++i, src += 24, dst += 24) {
        u64 x0;
        u64 x1;
        u64 x2;
        u64 y0;
        u64 y1;
        u64 y2;
        std::memcpy(&x0, src, 8);
        std::memcpy(&x1, src + 8, 8);
        std::memcpy(&x2, src + 16, 8);
        std::memcpy(&y0, dst, 8);
        std::memcpy(&y1, dst + 8, 8);
        std::memcpy(&y2, dst + 16, 8);
        const u64 d = (x0 ^ y0) | (x1 ^ y1) | (x2 ^ y2);
        CopyFixed<24>(dst, src);
        mask |= u64{d != 0} << i;
    }
    return mask;
}

// Walks the write list once, leaving blob equal to the serialized form and
// reporting whether any byte differed. Each write is compared whole first;
// only a changed write pays the per-descriptor verdicts and the copy, so an
// unchanged set stores nothing. Same chunking and bail rules as
// SerializeDescriptorWrites.
// The mapped form also records a per-write verdict and, for changed writes,
// a per-descriptor verdict in the slot, which a partial push consumes; the
// unmapped form is the plain walk.
// Inlined at every call site: with a second caller (the heap shadow census)
// the walk was outlined and its loop lost its registers.
template <bool kMap>
SHAD_FORCE_INLINE size_t MatchDescriptorWrites(std::span<const vk::WriteDescriptorSet> writes,
                                               Skipcache::Framework::DescDeltaSlot& slot,
                                               bool& changed) {
    auto& blob = slot.blob;
    u8* cursor = blob.data();
    const u8* const limit = blob.data() + blob.size();
    u64 diff = 0;
    // The verdict counters live in locals: they share the slot with the blob
    // the cursor stores into, so a slot-resident counter is reloaded after
    // every store. Every exit writes them back, so a bail leaves the slot as
    // before.
    [[maybe_unused]] u8* const verdicts = kMap ? slot.changed.data() : nullptr;
    [[maybe_unused]] u8* const write_verdicts = kMap ? slot.write_changed.data() : nullptr;
    [[maybe_unused]] u32 n = 0;
    [[maybe_unused]] u32 n_changed = 0;
    [[maybe_unused]] u32 w_idx = 0;
    [[maybe_unused]] u64 hdr_diff = 0;
    const auto write_back = [&] {
        if constexpr (kMap) {
            slot.desc_count = n;
            slot.desc_changed = n_changed;
            slot.header_changed = hdr_diff != 0;
            slot.write_count = w_idx;
        }
    };
    const auto sync_payload = [&]<size_t Stride>(const void* payload, u32 count) {
        const size_t bytes = size_t{count} * Stride;
        const u64 d =
            count == 1 ? DiffFixed<Stride>(payload, cursor) : DiffWords(payload, cursor, bytes);
        diff |= d;
        if constexpr (kMap) {
            write_verdicts[w_idx++] = d != 0;
        }
        if (d == 0) {
            if constexpr (kMap) {
                n += count;
            }
            cursor += bytes;
            return;
        }
        const u8* src = static_cast<const u8*>(payload);
        for (u32 i = 0; i < count; ++i) {
            if constexpr (kMap) {
                const u32 c = DiffFixed<Stride>(src, cursor) != 0;
                verdicts[n++] = static_cast<u8>(c);
                n_changed += c;
            }
            CopyFixed<Stride>(cursor, src);
            src += Stride;
            cursor += Stride;
        }
    };
    for (const auto& w : writes) {
        const u32 count = w.descriptorCount;
        if (static_cast<size_t>(limit - cursor) < 16 + size_t{count} * 24) {
            write_back();
            return 0;
        }
        if constexpr (kMap) {
            // Reached only past the size check above, so the u32 sum cannot wrap.
            if (n + count > slot.changed.size()) {
                write_back();
                return 0;
            }
        }
        const VkWriteDescriptorSet& raw = w;
        const u64 header = DiffFixed<16>(&raw.dstBinding, cursor);
        diff |= header;
        if constexpr (kMap) {
            hdr_diff |= header;
        }
        if (header != 0) {
            CopyFixed<16>(cursor, &raw.dstBinding);
        }
        cursor += 16;
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
            sync_payload.template operator()<24>(w.pBufferInfo, count);
            break;
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampler:
            sync_payload.template operator()<24>(w.pImageInfo, count);
            break;
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer:
            sync_payload.template operator()<8>(w.pTexelBufferView, count);
            break;
        default:
            write_back();
            return 0;
        }
    }
    write_back();
    changed = diff != 0;
    return static_cast<size_t>(cursor - blob.data());
}

// The count of a compaction that would not fit the write list: larger than any
// bound the caller compares against, so the set is pushed whole. Unreachable on
// radv, whose 32 push descriptors bound the descriptor count, but a real hole
// elsewhere, and the write list itself only fails closed by dropping runs.
constexpr size_t kRunsOverflow = std::numeric_limits<size_t>::max();

// One write of a compaction: the identity fields and exactly one payload
// pointer, the others null. A field copy of the source write would hand the
// driver whatever the other pointer slots last held.
vk::WriteDescriptorSet MakeRun(const vk::WriteDescriptorSet& w, u32 first, u32 count) {
    vk::WriteDescriptorSet run{};
    run.dstSet = w.dstSet;
    run.dstBinding = w.dstBinding + first;
    run.dstArrayElement = 0;
    run.descriptorCount = count;
    run.descriptorType = w.descriptorType;
    switch (w.descriptorType) {
    case vk::DescriptorType::eUniformBuffer:
    case vk::DescriptorType::eStorageBuffer:
        run.pBufferInfo = w.pBufferInfo + first;
        break;
    case vk::DescriptorType::eUniformTexelBuffer:
    case vk::DescriptorType::eStorageTexelBuffer:
        run.pTexelBufferView = w.pTexelBufferView + first;
        break;
    default:
        run.pImageInfo = w.pImageInfo + first;
        break;
    }
    return run;
}

// Compacts the write list to the descriptors the mapped walk saw change.
// Buffer and sampler writes span consecutive count-1 bindings, so each
// maximal run of changed descriptors becomes one write on its own binding;
// image writes are one binding with an array, so they go whole or not at all.
// The layouts come from the graphics and compute pipeline builders. Returns
// kRunsOverflow when the runs would not fit the write list.
size_t CompactDescriptorWrites(std::span<const vk::WriteDescriptorSet> in,
                               const Skipcache::Framework::DescDeltaSlot& map,
                               Pipeline::DescriptorWrites& out) {
    out.clear();
    u32 k = 0;
    u32 w_idx = 0;
    for (const auto& w : in) {
        const u32 count = w.descriptorCount;
        ASSERT(k + count <= map.desc_count);
        // A write the mapped walk saw unchanged carries no per-descriptor verdicts.
        if (!map.write_changed[w_idx++]) {
            k += count;
            continue;
        }
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eSampler: {
            u32 i = 0;
            while (i < count) {
                if (!map.changed[k + i]) {
                    ++i;
                    continue;
                }
                u32 j = i + 1;
                while (j < count && map.changed[k + j]) {
                    ++j;
                }
                if (out.size() >= Pipeline::NUM_DESCRIPTOR_WRITES) {
                    return kRunsOverflow;
                }
                out.push_back(MakeRun(w, i, j - i));
                i = j;
            }
            break;
        }
        default: {
            bool any = false;
            for (u32 i = 0; i < count; ++i) {
                any |= map.changed[k + i] != 0;
            }
            if (any) {
                if (out.size() >= Pipeline::NUM_DESCRIPTOR_WRITES) {
                    return kRunsOverflow;
                }
                out.push_back(MakeRun(w, 0, count));
            }
            break;
        }
        }
        k += count;
    }
    return out.size();
}

// The flat form of CompactDescriptorWrites: the change bits come from the
// plan's tiling instead of the mapped walk. A flat plan holds at most 63
// descriptors, so the runs always fit the write list.
static_assert(64 <= Pipeline::NUM_DESCRIPTOR_WRITES);
size_t CompactDescriptorWritesFlat(std::span<const vk::WriteDescriptorSet> in, const u8* first_desc,
                                   u64 mask, Pipeline::DescriptorWrites& out) {
    out.clear();
    u32 w_idx = 0;
    for (const auto& w : in) {
        const u32 count = w.descriptorCount;
        u64 bits = (mask >> first_desc[w_idx++]) & ((u64{1} << count) - 1);
        if (bits == 0) {
            continue;
        }
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eSampler: {
            while (bits != 0) {
                const u32 first = static_cast<u32>(std::countr_zero(bits));
                const u32 len = static_cast<u32>(std::countr_zero(~(bits >> first)));
                out.push_back(MakeRun(w, first, len));
                bits &= ~(((u64{1} << len) - 1) << first);
            }
            break;
        }
        default:
            out.push_back(MakeRun(w, 0, count));
            break;
        }
    }
    return out.size();
}

Pipeline::DescriptorWrites partial_scratch;

// The maintenance6 entry points take the same arguments in a struct and skip
// the runtime's forwarding wrapper (a struct build plus a tail call per push).
SHAD_FORCE_INLINE void PushSet(vk::CommandBuffer cmdbuf, bool direct,
                               vk::ShaderStageFlags stage_flags, vk::PipelineLayout layout,
                               vk::PipelineBindPoint bind_point, u32 count,
                               const vk::WriteDescriptorSet* writes) {
    if (direct) {
        const vk::PushDescriptorSetInfo info{
            .pNext = nullptr,
            .stageFlags = stage_flags,
            .layout = layout,
            .set = 0u,
            .descriptorWriteCount = count,
            .pDescriptorWrites = writes,
        };
        cmdbuf.pushDescriptorSet2(info);
    } else {
        cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0u,
                                    vk::ArrayProxy<const vk::WriteDescriptorSet>(count, writes));
    }
}

} // namespace

// Out of line and externally linked: the header's inline Next() calls it from
// every translation unit that fills a descriptor write list.
void DescWriteOverflow() {
    UNREACHABLE_MSG("Descriptor write list is full");
}

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   PipelineLayoutCache* layouts_, bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      layouts{layouts_}, is_compute{is_compute_}, direct_push{instance_.IsMaintenance6Supported()} {
}

void Pipeline::AssignLayouts(std::span<const vk::DescriptorSetLayoutBinding> bindings,
                             vk::DescriptorSetLayoutCreateFlags flags,
                             const vk::PushConstantRange& push_constants,
                             std::string_view debug_name) {
    if (layouts) {
        const auto shared = layouts->Acquire(bindings, flags, push_constants, debug_name);
        desc_layout = shared.set;
        pipeline_layout = shared.pipeline;
        return;
    }
    const auto device = instance.GetDevice();
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto [set_result, set] = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(set_result == vk::Result::eSuccess, "Failed to create descriptor set layout: {}",
               vk::to_string(set_result));
    owned_desc_layout = std::move(set);
    desc_layout = *owned_desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &desc_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    owned_pipeline_layout = std::move(layout);
    pipeline_layout = *owned_pipeline_layout;
    SetObjectName(device, pipeline_layout, "{} PipelineLayout {}",
                  is_compute ? "Compute" : "Graphics", debug_name);
}

bool Pipeline::FitsPushDescriptors(u32 count) const {
    // The limit is inclusive; the older test kept a pipeline whose total
    // equals it on the descriptor heap.
    static const bool full = EmulatorSettings.IsPushDescFullLimit();
    return full ? count <= instance.MaxPushDescriptors() : count < instance.MaxPushDescriptors();
}

Pipeline::~Pipeline() {
    // An owned layout dies with the pipeline and the driver may hand its
    // handle value to a layout of another shape: the heap forgets the sets
    // it cached under it, in the recycling ring and in the batch map alike.
    if (owned_desc_layout) {
        desc_heap.Forget(*owned_desc_layout);
    }
}

void Pipeline::BindResources(std::span<vk::WriteDescriptorSet> set_writes,
                             const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data, u32 buffer_info_n,
                             u32 image_info_n) const {
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    if (!buffer_barriers.empty()) {
        const auto dependencies = vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = u32(buffer_barriers.size()),
            .pBufferMemoryBarriers = buffer_barriers.data(),
        };
        scheduler.EndRendering();
        cmdbuf.pipelineBarrier2(dependencies);
    }

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    static const bool push_const_dedup = EmulatorSettings.IsPushConstDedup();
    static const bool heap_shadow = EmulatorSettings.IsDescHeapShadowCensus();
    bool push_same = false;
    if (push_const_dedup) {
        // A push identical to the last one on this command buffer, through
        // the same layout, leaves the same bytes in place. The slot is written
        // either way, so a taken skip stores what the storage already holds.
        using namespace VideoCore::Skipcache;
        auto& sc = Framework::Instance();
        auto& slot = sc.PushConstState();
        if (sc.Active()) [[likely]] {
            static_assert(sizeof(push_data) % sizeof(u64) == 0 &&
                          sizeof(push_data) <= sizeof(slot.words));
            constexpr size_t kWords = sizeof(push_data) / sizeof(u64);
            std::array<u64, kWords> words;
            std::memcpy(words.data(), &push_data, sizeof(push_data));
            const u64 layout_bits =
                std::bit_cast<u64>(static_cast<VkPipelineLayout>(pipeline_layout));
            const u32 flag_bits = static_cast<u32>(static_cast<vk::ShaderStageFlags>(stage_flags));
            u64 diff = 0;
            for (size_t i = 0; i < kWords; ++i) {
                diff |= words[i] ^ slot.words[i];
                slot.words[i] = words[i];
            }
            diff |= slot.layout ^ layout_bits;
            diff |= slot.stage_flags ^ flag_bits;
            const bool same = slot.valid && diff == 0;
            slot.layout = layout_bits;
            slot.stage_flags = flag_bits;
            slot.valid = true;
            sc.CountPushConst(same);
            push_same = same && sc.ActiveMode() != Mode::ValidateOnly;
        } else {
            slot.valid = false;
        }
    }
    if (!push_same) {
        if (direct_push) {
            const vk::PushConstantsInfo info{
                .pNext = nullptr,
                .layout = pipeline_layout,
                .stageFlags = stage_flags,
                .offset = 0u,
                .size = sizeof(push_data),
                .pValues = &push_data,
            };
            cmdbuf.pushConstants2(info);
        } else {
            cmdbuf.pushConstants(pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);
        }
    }

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        if (heap_shadow) {
            // A push pipeline leaves set 0 holding its layout, which the next
            // heap pipeline's shadow probe must see as a key miss.
            Skipcache::Framework::Instance().HeapShadow(IsCompute() ? 1 : 0).valid = false;
        }
        // Descriptor delta cache: push descriptors are command buffer state,
        // so re-pushing a byte-identical write set onto the same command
        // buffer and layout is a spec-level no-op; skipping it saves the
        // driver-side descriptor update. Byte equality of the serialized
        // form proves redundancy, so the shadow verdict is structural.
        auto& sc = Skipcache::Framework::Instance();
        constexpr auto kCache = Skipcache::CacheId::DescDelta;
        if (sc.Active() && sc.ShouldProbe(kCache)) {
            auto& ctr = sc.Counters(kCache);
            ++ctr.eligible;
            const bool timed = sc.SampleTimer(kCache);
            const u64 t0 = timed ? sc.Now() : 0;
            const size_t idx = IsCompute() ? 1 : 0;
            auto& slot = sc.DescDeltaState(idx);
            auto& scratch = sc.DescDeltaScratch();
            static const bool inplace = EmulatorSettings.IsDescDeltaInplace();
            static const bool partial_enabled = [] {
                const bool partial = EmulatorSettings.IsDescDeltaPartial();
                if (partial && !EmulatorSettings.IsDescDeltaInplace()) {
                    LOG_WARNING(Render_Vulkan, "partial descriptor pushes need desc_delta_inplace; "
                                               "every content miss pushes the whole set");
                }
                return partial;
            }();
            static const bool flat_enabled = [] {
                const bool f = EmulatorSettings.IsDescDeltaFlat();
                if (f && (!EmulatorSettings.IsDescDeltaInplace() ||
                          !EmulatorSettings.IsDescLayoutShare())) {
                    LOG_WARNING(Render_Vulkan, "flat descriptor deltas need desc_delta_inplace "
                                               "and desc_layout_share; the write walk runs");
                    return false;
                }
                return f;
            }();
            const u64 tick = scheduler.CurrentTick();
            const u64 layout = std::bit_cast<u64>(static_cast<VkPipelineLayout>(pipeline_layout));
            const u64 foreign = sc.ForeignPushGen(idx);
            bool changed = false;
            // Flat delta: under a plan hit the write headers are a function of
            // the layout, so only the two info arrays the plan tiles are
            // compared, as 24-byte descriptors against the blob's flat form.
            // The extents fail closed to the walk if an emission ever drifts.
            const bool plan_hit = set_writes.data() == bind_plan.writes.get();
            const bool flat = flat_enabled && bind_plan.flat && plan_hit &&
                              buffer_info_n == bind_plan.buffer_descs &&
                              image_info_n == bind_plan.image_descs;
            u64 shape = 0;
            size_t size;
            if (flat) {
                const u32 nb = bind_plan.buffer_descs;
                const u32 ni = bind_plan.image_descs;
                u8* blob = slot.blob.data();
                const u64 mask =
                    SyncDescriptors24(bind_plan.buffer_base, blob, nb) |
                    (SyncDescriptors24(bind_plan.image_base, blob + size_t{nb} * 24, ni) << nb);
                slot.mask = mask;
                slot.desc_count = nb + ni;
                slot.desc_changed = static_cast<u32>(std::popcount(mask));
                slot.header_changed = false;
                changed = mask != 0;
                size = size_t{nb + ni} * 24;
                shape = (u64{nb} << 32 | ni) | (u64{1} << 63);
                ++slot.flat;
            } else {
                size = inplace ? (partial_enabled
                                      ? MatchDescriptorWrites<true>(set_writes, slot, changed)
                                      : MatchDescriptorWrites<false>(set_writes, slot, changed))
                               : SerializeDescriptorWrites(set_writes, scratch);
                ++slot.unflat;
                if (flat_enabled && bind_plan.flat && plan_hit) {
                    ++slot.extmiss;
                }
            }
            bool would_hit = false;
            bool partial_ok = false;
            if (size == 0) {
                ++ctr.veto[0]; // unknown type or overflow: fail closed
            } else if (!slot.valid) {
                ++ctr.miss_cold;
            } else if (slot.tick != tick || slot.foreign_gen != foreign) {
                ++ctr.miss_gen[Skipcache::LaneTick];
            } else if (slot.layout != layout || slot.flat_shape != shape) {
                // The shape discriminates the flat blob form from the
                // serialized one: a flat probe against a serialized slot and
                // the reverse must key-miss. The header certificate rests on
                // layout handles never being reused within a tick, which
                // holds for shared layouts (the cache never evicts).
                ++ctr.miss_key;
            } else if (slot.size != size ||
                       (inplace ? changed
                                : std::memcmp(slot.blob.data(), scratch.data(), size) != 0)) {
                ++ctr.veto[1]; // content changed
                // Same command buffer, foreign gen and layout: the driver's
                // set still holds the last push, so only the changed
                // descriptors need to go. Near-total changes push whole.
                partial_ok = inplace && partial_enabled && size == slot.size &&
                             !slot.header_changed && slot.desc_changed * 4 <= slot.desc_count * 3;
                if (partial_ok) {
                    ++slot.partial;
                    slot.descs += slot.desc_count;
                    slot.pushed += slot.desc_changed;
                }
            } else {
                would_hit = true;
                ++ctr.hits;
            }
            ++slot.probes;
            slot.hits += would_hit;
            if (timed) {
                ctr.guard_ns += sc.CorrectSample(sc.Now() - t0);
                ++ctr.guard_samples;
            }
            if (would_hit && sc.MayConsume(kCache) && !sc.ShouldVerify(kCache)) {
                return; // identical set already pushed on this command buffer
            }
            const bool timed_miss = timed && !would_hit;
            const u64 m0 = timed_miss ? sc.Now() : 0;
            if (partial_ok && sc.MayConsume(kCache) && !sc.ShouldVerify(kCache)) {
                // Each extra write costs the driver's per-write prologue, so a
                // compaction that saves fewer descriptors than it adds writes
                // is sent whole.
                const size_t runs =
                    flat ? CompactDescriptorWritesFlat(set_writes, bind_plan.first_desc.data(),
                                                       slot.mask, partial_scratch)
                         : CompactDescriptorWrites(set_writes, slot, partial_scratch);
                if (runs <= set_writes.size() + (slot.desc_count - slot.desc_changed) / 2) {
                    slot.runs += runs;
                    PushSet(cmdbuf, direct_push, stage_flags, pipeline_layout, bind_point,
                            static_cast<u32>(partial_scratch.size()), partial_scratch.data());
                } else {
                    ++slot.split;
                    PushSet(cmdbuf, direct_push, stage_flags, pipeline_layout, bind_point,
                            static_cast<u32>(set_writes.size()), set_writes.data());
                }
            } else {
                PushSet(cmdbuf, direct_push, stage_flags, pipeline_layout, bind_point,
                        static_cast<u32>(set_writes.size()), set_writes.data());
            }
            if (timed_miss) {
                ctr.miss_ns += sc.CorrectSample(sc.Now() - m0);
                ++ctr.miss_samples;
            }
            if (would_hit) {
                if (sc.GetState(kCache) != Skipcache::State::Learning) {
                    sc.RecordVerifyClean(kCache);
                }
                return;
            }
            if (size != 0) {
                slot.valid = true;
                slot.tick = tick;
                slot.layout = layout;
                slot.flat_shape = shape;
                slot.foreign_gen = foreign;
                slot.size = static_cast<u32>(size);
                // The in-place walk already left the blob in its serialized form.
                if (!inplace) {
                    std::memcpy(slot.blob.data(), scratch.data(), size);
                }
                sc.NotifyPopulated(kCache);
            } else {
                // A bailed in-place walk leaves the blob partially rewritten.
                slot.valid = false;
            }
            return;
        }
        PushSet(cmdbuf, direct_push, stage_flags, pipeline_layout, bind_point,
                static_cast<u32>(set_writes.size()), set_writes.data());
        return;
    }

    const auto desc_set = desc_heap.Commit(desc_layout);
    {
        // The heap leg is the delta-free stream; its size prices any future
        // second push set. DescDeltaState is ungated.
        auto& hs = VideoCore::Skipcache::Framework::Instance().DescDeltaState(
            bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0);
        ++hs.heap;
        u32 descs = 0;
        u32 images = 0;
        u32 samplers = 0;
        for (const auto& set_write : set_writes) {
            descs += set_write.descriptorCount;
            if (set_write.descriptorType == vk::DescriptorType::eSampledImage ||
                set_write.descriptorType == vk::DescriptorType::eStorageImage) {
                images += set_write.descriptorCount;
            } else if (set_write.descriptorType == vk::DescriptorType::eSampler) {
                samplers += set_write.descriptorCount;
            }
        }
        hs.heap_descs += descs;
        hs.heap_images += images;
        hs.heap_samplers += samplers;
        const u32 limit = instance.MaxPushDescriptors();
        ++hs.heap_hist[descs <= limit ? 0
                       : descs <= 40  ? 1
                       : descs <= 48  ? 2
                       : descs <= 64  ? 3
                                      : 4];
    }
    if (heap_shadow) {
        // Set 0 = the first split descriptors pushed through the delta cache,
        // the rest a heap tail: the walk runs against a shadow slot before
        // dstSet is stamped (the serialized header excludes it); the bind
        // below is unchanged and no verdict is consumed.
        auto& sc = VideoCore::Skipcache::Framework::Instance();
        const size_t idx = bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0;
        auto& sh = sc.HeapShadow(idx);
        auto& hc = sc.HeapShadowCount();
        const u64 shadow_tick = scheduler.CurrentTick();
        const u64 foreign = sc.ForeignPushGen(idx);
        const u64 layout = std::bit_cast<u64>(static_cast<VkPipelineLayout>(pipeline_layout));
        const u32 split = instance.MaxPushDescriptors();
        u32 total = 0;
        for (const auto& set_write : set_writes) {
            total += set_write.descriptorCount;
        }
        const u32 prefix = std::min<u32>(total, split);
        const u32 tail = total - prefix;
        bool changed = false;
        const size_t size = MatchDescriptorWrites<true>(set_writes, sh, changed);
        ++hc.probes;
        hc.compute += idx;
        hc.prefix += prefix;
        hc.tail_descs += tail;
        if (size == 0) {
            ++hc.veto;
        } else if (!sh.valid) {
            ++hc.key;
        } else if (sh.tick != shadow_tick || sh.foreign_gen != foreign) {
            ++hc.gen;
        } else if (sh.layout != layout) {
            ++hc.key;
        } else if (!changed) {
            ++hc.hits;
            hc.tail_probed += tail;
            hc.tail_same += tail;
        } else {
            u32 prefix_changed = 0;
            u32 tail_changed = 0;
            u32 k = 0;
            for (size_t w = 0; w < set_writes.size() && w < sh.write_changed.size(); ++w) {
                const u32 n = set_writes[w].descriptorCount;
                if (k + n > sh.changed.size()) {
                    break;
                }
                if (sh.write_changed[w]) {
                    for (u32 i = 0; i < n; ++i) {
                        if (!sh.changed[k + i]) {
                            continue;
                        }
                        if (k + i < split) {
                            ++prefix_changed;
                        } else {
                            ++tail_changed;
                        }
                    }
                }
                k += n;
            }
            hc.tail_probed += tail;
            hc.tail_same += tail - tail_changed;
            if (prefix_changed == 0) {
                ++hc.hits;
            } else if (!sh.header_changed && prefix_changed * 4 <= prefix * 3) {
                ++hc.partial;
                hc.pushed += prefix_changed;
            } else {
                ++hc.whole;
                hc.pushed += prefix;
            }
        }
        sh.valid = size != 0;
        sh.tick = shadow_tick;
        sh.layout = layout;
        sh.size = static_cast<u32>(size);
        // The bump this bind makes below is the one the two-set world would
        // not make; the shadow expects the generation after it.
        sh.foreign_gen = foreign + 1;
    }
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(
        vk::ArrayProxy<const vk::WriteDescriptorSet>(static_cast<uint32_t>(set_writes.size()),
                                                     set_writes.data()),
        {});
    cmdbuf.bindDescriptorSets(bind_point, pipeline_layout, 0, desc_set, {});
    // The heap set replaces this bind point's set 0 behind the delta cache;
    // the bump makes its next probe miss.
    VideoCore::Skipcache::Framework::Instance().BumpForeignPushGen(
        bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0);
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
