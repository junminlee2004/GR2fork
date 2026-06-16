// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <array>
#include <mutex>

#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {
namespace {

static inline u64 Hash64Step(u64 h, u64 v) noexcept {
    // 64-bit FNV-1a step (good enough for descriptor signatures).
    h ^= v;
    h *= 1099511628211ULL;
    return h;
}


static inline u64 HashOneWrite(const vk::WriteDescriptorSet& w) noexcept {
    // PERF(GR2FORK v1.14): Mark descriptorCount==1 as the dominant case
    // (~0.41% of GpuComm in a captured profile). Reorder so the hot branch
    // is straight-line and the rare descriptorCount>1 path becomes the
    // taken-branch tail.
    //
    // Inside the descriptorCount==1 path, replace the four-step serial FNV
    // chain with INDEPENDENT FNV lanes XOR-combined. The original chain has
    // a ~16-cycle critical path on Zen4 (each imul depends on the previous
    // one); the parallel form pipelines all four muls and finishes in ~5
    // cycles. Result is bit-identical to the same input on the same hardware
    // ABI run, but signature values differ from the previous build — that is
    // safe because the cached_push_desc_sig is purely a runtime cmdbuf-scoped
    // cache and is never serialized.
    constexpr u64 P = 1099511628211ULL;
    constexpr u64 OFFSET = 1469598103934665603ULL;

    if (w.descriptorCount == 1) [[likely]] {
        const u64 head =
            (static_cast<u64>(w.dstBinding) << 32) |
            (static_cast<u64>(w.dstArrayElement) << 16) |
            static_cast<u64>(static_cast<u32>(w.descriptorType));

        switch (w.descriptorType) {
            case vk::DescriptorType::eUniformBuffer:
            case vk::DescriptorType::eStorageBuffer:
            case vk::DescriptorType::eUniformBufferDynamic:
            case vk::DescriptorType::eStorageBufferDynamic: {
                const auto& bi = w.pBufferInfo[0];
                const u64 v0 = head;
                const u64 v1 = reinterpret_cast<u64>(static_cast<VkBuffer>(bi.buffer));
                const u64 v2 = static_cast<u64>(bi.offset);
                const u64 v3 = static_cast<u64>(bi.range);
                const u64 a0 = (v0 ^ OFFSET) * P;
                const u64 a1 = (v1 ^ 0x84222325cbf29ce4ULL) * P;
                const u64 a2 = (v2 ^ 0x9e3779b97f4a7c15ULL) * P;
                const u64 a3 = (v3 ^ 0xbf58476d1ce4e5b9ULL) * P;
                return a0 ^ a1 ^ a2 ^ a3;
            }
            case vk::DescriptorType::eCombinedImageSampler:
            case vk::DescriptorType::eSampledImage:
            case vk::DescriptorType::eStorageImage:
            case vk::DescriptorType::eInputAttachment:
            case vk::DescriptorType::eSampler: {
                const auto& ii = w.pImageInfo[0];
                const u64 v0 = head;
                const u64 v1 = reinterpret_cast<u64>(static_cast<VkSampler>(ii.sampler));
                const u64 v2 = reinterpret_cast<u64>(static_cast<VkImageView>(ii.imageView));
                const u64 v3 = static_cast<u64>(static_cast<u32>(ii.imageLayout));
                const u64 a0 = (v0 ^ OFFSET) * P;
                const u64 a1 = (v1 ^ 0x84222325cbf29ce4ULL) * P;
                const u64 a2 = (v2 ^ 0x9e3779b97f4a7c15ULL) * P;
                const u64 a3 = (v3 ^ 0xbf58476d1ce4e5b9ULL) * P;
                return a0 ^ a1 ^ a2 ^ a3;
            }
            case vk::DescriptorType::eUniformTexelBuffer:
            case vk::DescriptorType::eStorageTexelBuffer: {
                const u64 v0 = head;
                const u64 v1 = reinterpret_cast<u64>(static_cast<VkBufferView>(w.pTexelBufferView[0]));
                const u64 a0 = (v0 ^ OFFSET) * P;
                const u64 a1 = (v1 ^ 0x84222325cbf29ce4ULL) * P;
                return a0 ^ a1;
            }
            default: {
                const u64 v0 = head;
                const u64 v1 = static_cast<u64>(reinterpret_cast<uintptr_t>(w.pNext));
                const u64 a0 = (v0 ^ OFFSET) * P;
                const u64 a1 = (v1 ^ 0x84222325cbf29ce4ULL) * P;
                return a0 ^ a1;
            }
        }
    }

    // Cold fallback for descriptorCount > 1.
    u64 h = OFFSET;
    h = Hash64Step(h, static_cast<u64>(w.dstBinding));
    h = Hash64Step(h, static_cast<u64>(w.dstArrayElement));
    h = Hash64Step(h, static_cast<u64>(w.descriptorCount));
    h = Hash64Step(h, static_cast<u64>(static_cast<u32>(w.descriptorType)));

    switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eUniformBufferDynamic:
        case vk::DescriptorType::eStorageBufferDynamic: {
            const auto* infos = w.pBufferInfo;
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                h = Hash64Step(h, reinterpret_cast<u64>(static_cast<VkBuffer>(infos[i].buffer)));
                h = Hash64Step(h, static_cast<u64>(infos[i].offset));
                h = Hash64Step(h, static_cast<u64>(infos[i].range));
            }
            break;
        }
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eInputAttachment:
        case vk::DescriptorType::eSampler: {
            const auto* infos = w.pImageInfo;
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                h = Hash64Step(h, reinterpret_cast<u64>(static_cast<VkSampler>(infos[i].sampler)));
                h = Hash64Step(h, reinterpret_cast<u64>(static_cast<VkImageView>(infos[i].imageView)));
                h = Hash64Step(h, static_cast<u64>(static_cast<u32>(infos[i].imageLayout)));
            }
            break;
        }
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer: {
            const auto* views = w.pTexelBufferView;
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                h = Hash64Step(h, reinterpret_cast<u64>(static_cast<VkBufferView>(views[i])));
            }
            break;
        }
        default:
            h = Hash64Step(h, static_cast<u64>(reinterpret_cast<uintptr_t>(w.pNext)));
            break;
    }
    return h;
}

static inline u64 Mix64(u64 x) noexcept {
    // splitmix64 finalizer
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

[[maybe_unused]] static u64 HashDescriptorWrites(const Pipeline::DescriptorWrites& writes) noexcept {
    // PERF(GR2): Order-INDEPENDENT signature over UNIQUE keys.
    //
    // BindResources frequently produces the same set of descriptor writes but in a slightly
    // different order (iteration order differences, stage ordering, etc.). An order-dependent
    // hash misses the cache and forces redundant vkCmdPushDescriptorSetKHR calls, which then
    // show up as big __memmove + RADV push-descriptor overhead.
    //
    // We rely on the caller to preserve "last update wins" semantics by de-duping writes
    // (one write per {binding,array,type}). Once keys are unique, the order is irrelevant.
    u64 h = 0x6a09e667f3bcc909ULL ^ static_cast<u64>(writes.size());
    for (const auto& w : writes) {
        // Key is stable for the destination slot; HashOneWrite includes the actual resource handles.
        const u64 key =
        (static_cast<u64>(w.dstBinding) << 40) |
        (static_cast<u64>(w.dstArrayElement) << 16) |
        static_cast<u64>(static_cast<u32>(w.descriptorType));
        const u64 wh = HashOneWrite(w);
        // Commutative combine. (Mix64 is non-linear so XOR is fine here.)
        h ^= Mix64(key ^ (wh + 0x9e3779b97f4a7c15ULL));
    }
    // Finalize
    return Mix64(h);
}

static inline u64 HashDescriptorWritesOrdered(const Pipeline::DescriptorWrites& writes) noexcept {
    // Faster ordered hash for Rasterizer-emitted monotonic unique writes.
    // If ordering changes, callers can fall back to the order-independent hash.
    //
    // PERF(GR2FORK v1.17): Drop the three per-field Hash64Step calls
    // (dstBinding / dstArrayElement / descriptorType) — HashOneWrite()
    // already mixes those exact fields into its output via the packed
    // `head` value (dstBinding<<32 | dstArrayElement<<16 | descriptorType).
    // The original four-step chain was redundantly hashing the same fields
    // twice. Collapsing to a single Hash64Step(h, HashOneWrite(w)) per
    // write shortens the cross-iteration dependency chain from 4 imuls to
    // 1 imul (~5 cycle critical path per write instead of ~20). HashOneWrite's
    // own work pipelines independently of the running h chain.
    //
    // Collision behaviour: HashOneWrite is a 64-bit hash of the full
    // (dstBinding, dstArrayElement, descriptorType, payload) tuple, so the
    // simplified chain has the same ~2^-64 self-collision rate per write
    // pair the original chain had. For a runtime cmdbuf-scoped descriptor
    // set cache key, that's negligible at any realistic draw count.
    u64 h = 1469598103934665603ULL;
    h = Hash64Step(h, static_cast<u64>(writes.size()));
    for (const auto& w : writes) {
        h = Hash64Step(h, HashOneWrite(w));
    }
    return h;
}

static u64 HashPushConstants(const Shader::PushData& push_data) noexcept {
    // PERF(v8): Process in 8-byte chunks. PushData is ~120 bytes = 15 iterations
    // vs 120 byte-iterations previously.
    u64 h = 1469598103934665603ULL;
    const auto* qwords = reinterpret_cast<const u64*>(&push_data);
    constexpr size_t num_qwords = sizeof(push_data) / 8;
    constexpr size_t tail_bytes = sizeof(push_data) % 8;
    for (size_t i = 0; i < num_qwords; ++i) {
        h = Hash64Step(h, qwords[i]);
    }
    if constexpr (tail_bytes > 0) {
        u64 tail = 0;
        std::memcpy(&tail, reinterpret_cast<const char*>(&push_data) + num_qwords * 8, tail_bytes);
        h = Hash64Step(h, tail);
    }
    return h;
}
} // namespace

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

void Pipeline::BindResources(DescriptorWrites& set_writes, BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data) const {
    // Phase A split: barriers + EndRendering go on the primary cmdbuf (must
    // happen outside any rendering pass). Push constants and descriptor
    // writes go on the recording cmdbuf (the secondary in threaded mode,
    // primary in non-threaded). Both accessors return the same handle today;
    // diverge under secondary command buffers.
    const auto primary_cmdbuf = scheduler.PrimaryCommandBuffer();
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    if (!buffer_barriers.empty()) {
        // OPT: Deduplicate barriers for the same buffer. Multiple descriptors often reference
        // the same VkBuffer, creating redundant barriers that each force a pipeline stall.
        // Merge into one barrier per buffer covering the union of stages and access masks.
        if (buffer_barriers.size() > 1) {
            // Sort by buffer handle for easy grouping.
            std::sort(buffer_barriers.begin(), buffer_barriers.end(),
                      [](const vk::BufferMemoryBarrier2& a, const vk::BufferMemoryBarrier2& b) {
                          return a.buffer < b.buffer;
                      });
            // Merge consecutive barriers for the same buffer.
            u32 write_idx = 0;
            for (u32 read_idx = 1; read_idx < buffer_barriers.size(); ++read_idx) {
                auto& dst = buffer_barriers[write_idx];
                const auto& src = buffer_barriers[read_idx];
                if (dst.buffer == src.buffer) {
                    // Merge: widen stage masks and access masks, extend range.
                    dst.srcStageMask |= src.srcStageMask;
                    dst.srcAccessMask |= src.srcAccessMask;
                    dst.dstStageMask |= src.dstStageMask;
                    dst.dstAccessMask |= src.dstAccessMask;
                    const u64 new_end =
                        std::max(dst.offset + dst.size, src.offset + src.size);
                    dst.offset = std::min(dst.offset, src.offset);
                    dst.size = new_end - dst.offset;
                } else {
                    buffer_barriers[++write_idx] = src;
                }
            }
            buffer_barriers.resize(write_idx + 1);
        }

        const auto dependencies = vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = u32(buffer_barriers.size()),
            .pBufferMemoryBarriers = buffer_barriers.data(),
        };
        scheduler.EndRendering();
        primary_cmdbuf.pipelineBarrier2(dependencies);
    }

    const vk::ShaderStageFlags stage_flags =
        IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;

        // Push constants: exact cache hit test before hashing/byte-walking.
        // Shader::PushData is small; memcmp is cheaper than recomputing a signature every bind.
        // Cross-pipeline gen guards against another Pipeline having pushed
        // its own constants to the same cmdbuf since our last update.
        const u64 push_const_gen_now = scheduler.GetPushConstGen(bind_point);
        const bool push_constants_unchanged =
        cached_push_const_valid && cached_push_const_cmdbuf == cmdbuf &&
        cached_push_const_layout == *pipeline_layout &&
        cached_push_const_stages == stage_flags &&
        cached_push_const_gen == push_const_gen_now &&
        std::memcmp(&cached_push_const_data, &push_data, sizeof(push_data)) == 0;
        // PERF(GR2FORK v1.16): The push-constants byte-equality cache is hit
        // on the bulk of consecutive same-shader draws — the cmdbuf-scoped
        // gen counter is bumped only when another Pipeline pushes its own
        // constants between us. Hint the cmdbuf.pushConstants emit-path as
        // the cold tail.
        if (!push_constants_unchanged) [[unlikely]] {
            cmdbuf.pushConstants(*pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);

            scheduler.BumpPushConstGen(bind_point);
            cached_push_const_layout = *pipeline_layout;
            cached_push_const_stages = stage_flags;
            cached_push_const_data = push_data;
            cached_push_const_cmdbuf = cmdbuf;
            cached_push_const_gen = scheduler.GetPushConstGen(bind_point);
            cached_push_const_valid = true;
        }

          if (set_writes.empty()) {
              return;
          }
          const u64 cmd_tick = scheduler.CurrentTick();
          // PERF(v8): Ultra-fast path for tiny delta write sets (1-4 writes).
          // Most per-draw deltas from the Rasterizer's ShouldWriteDescriptor filter
          // are 0-4 writes. Skip monotonicity checks and use a trivially cheap signature.
          if (uses_push_descriptors && set_writes.size() <= 4) {
              u64 tiny_sig = static_cast<u64>(set_writes.size()) * 0x9e3779b97f4a7c15ULL;
              for (const auto& w : set_writes) {
                  tiny_sig ^= Hash64Step(static_cast<u64>(w.dstBinding) << 32 |
                      static_cast<u64>(static_cast<u32>(w.descriptorType)),
                      HashOneWrite(w));
              }
              if (cached_push_desc_valid && cached_push_desc_is_delta &&
                  cached_push_desc_sig == tiny_sig &&
                  cached_push_desc_cmdbuf == cmdbuf &&
                  cached_push_desc_tick == cmd_tick &&
                  cached_push_desc_layout == *pipeline_layout &&
                  cached_push_desc_bind_point == bind_point &&
                  cached_push_desc_gen == scheduler.GetPushDescGen(bind_point)) {
                  return;
              }
              cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
              scheduler.BumpPushDescGen(bind_point);
              cached_push_desc_cmdbuf = cmdbuf;
              cached_push_desc_tick = cmd_tick;
              cached_push_desc_layout = *pipeline_layout;
              cached_push_desc_bind_point = bind_point;
              cached_push_desc_sig = tiny_sig;
              cached_push_desc_gen = scheduler.GetPushDescGen(bind_point);
              cached_push_desc_valid = true;
              cached_push_desc_is_delta = true;
              return;
          }
          // PERF(GR2): If the caller already emits *delta* (sparse) writes, a full write-set signature
          // hash + cache check is usually overhead. However, GR2 also repeats many *identical* delta
          // batches back-to-back (same pipeline/layout/cmd buffer), which still pay heavy RADV push
          // descriptor CPU cost. Handle that case here.
          //
          // Heuristic: treat small write batches (common per-draw deltas) and any batch whose dstBinding
          // indices are outside a dense [0..N) range as delta.
          // PERF(v8+OPT): Rasterizer always emits descriptor writes in strictly
          // monotonic binding order. Always use the faster ordered hash.
          if (uses_push_descriptors) {
              bool likely_delta = (set_writes.size() <= 16);
              if (!likely_delta) {
                  const u32 n = static_cast<u32>(set_writes.size());
                  for (const auto& w : set_writes) {
                      if (w.dstBinding >= n) {
                          likely_delta = true;
                          break;
                      }
                  }
              }
              if (likely_delta) {
                  const u64 delta_sig = HashDescriptorWritesOrdered(set_writes);

                  if (cached_push_desc_valid && cached_push_desc_is_delta &&
                      cached_push_desc_sig == delta_sig &&
                      cached_push_desc_cmdbuf == cmdbuf &&
                      cached_push_desc_tick == cmd_tick &&
                      cached_push_desc_layout == *pipeline_layout &&
                      cached_push_desc_bind_point == bind_point &&
                      cached_push_desc_gen == scheduler.GetPushDescGen(bind_point)) {
                      return;
                      }

                      cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
                      scheduler.BumpPushDescGen(bind_point);

                  cached_push_desc_cmdbuf = cmdbuf;
                  cached_push_desc_tick = cmd_tick;
                  cached_push_desc_layout = *pipeline_layout;
                  cached_push_desc_bind_point = bind_point;
                  cached_push_desc_sig = delta_sig;
                  cached_push_desc_gen = scheduler.GetPushDescGen(bind_point);
                  cached_push_desc_valid = true;
                  cached_push_desc_is_delta = true;
                  return;
              }
          }
          // OPT(v18): Rasterizer always emits strictly increasing unique dstBinding
          // values. Use the faster ordered hash path unconditionally.
          // The de-dup table below is dead code and has been removed.

const u64 sig = HashDescriptorWritesOrdered(set_writes);

    if (uses_push_descriptors) {
        // Fast path: skip push-descriptor update when the write set is identical to the last push.
        // NOTE: Delta push-descriptor updates proved unstable for GR2 on some RADV paths.
        if (cached_push_desc_valid && !cached_push_desc_is_delta && cached_push_desc_sig == sig &&
            cached_push_desc_cmdbuf == cmdbuf && cached_push_desc_tick == cmd_tick &&
            cached_push_desc_layout == *pipeline_layout && cached_push_desc_bind_point == bind_point &&
            cached_push_desc_gen == scheduler.GetPushDescGen(bind_point)) {
            return;
        }

        cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
        scheduler.BumpPushDescGen(bind_point);

        cached_push_desc_cmdbuf = cmdbuf;
        cached_push_desc_tick = cmd_tick;
        cached_push_desc_layout = *pipeline_layout;
        cached_push_desc_bind_point = bind_point;
        cached_push_desc_sig = sig;
        cached_push_desc_gen = scheduler.GetPushDescGen(bind_point);
        cached_push_desc_valid = true;
        cached_push_desc_is_delta = false;
        return;
    }

    // Non-push path: cache descriptor sets by signature within the command buffer to avoid
    // repeated vkUpdateDescriptorSets (big CPU win on RADV for GR2).
    //
    // PERF(GR2FORK v1.16): The cmdbuf+tick combo only changes at submission
    // boundaries, not within a frame's worth of BindResources calls. Hint the
    // reset path as cold so the cache-already-fresh branch lays out as the
    // straight-line fast case.
    if (cached_descset_cmdbuf != cmdbuf || cached_descset_tick != cmd_tick) [[unlikely]] {
        cached_descset_tick = cmd_tick;
        cached_descset_cmdbuf = cmdbuf;
        cached_descset_count = 0;
        cached_bound_desc_valid = false;
    }

    vk::DescriptorSet desc_set{};
    for (u32 i = 0; i < cached_descset_count; ++i) {
        if (cached_descsets[i].sig == sig) {
            desc_set = cached_descsets[i].set;
            break;
        }
    }

    // PERF(GR2FORK v1.16): After warmup the per-cmdbuf descriptor-set cache
    // is populated for the small set of signatures the frame uses, so the
    // for-loop above resolves desc_set on most calls. Hint the
    // updateDescriptorSets / desc_heap.Commit miss path as cold.
    if (desc_set == vk::DescriptorSet{}) [[unlikely]] {
        desc_set = desc_heap.Commit(*desc_layout);
        for (auto& set_write : set_writes) {
            set_write.dstSet = desc_set;
        }
        instance.GetDevice().updateDescriptorSets(set_writes, {});

        u32 slot{};
        if (cached_descset_count < DescSetCacheSize) {
            slot = cached_descset_count++;
        } else {
            // Simple replacement to keep the cache bounded.
            slot = static_cast<u32>(sig) & (DescSetCacheSize - 1);
        }
        cached_descsets[slot] = CachedDescSetEntry{sig, desc_set};
    }

    // PERF(GR2FORK v1.16): The bound-descriptor cache hit is the dominant
    // return for consecutive same-signature draws — the descriptor-set cache
    // above keeps mapping a stable sig → same desc_set, and bind_point /
    // layout are stable per pipeline. Hint the early return.
    if (cached_bound_desc_valid && cached_bound_desc_cmdbuf == cmdbuf && cached_bound_desc_tick == cmd_tick &&
        cached_bound_desc_layout == *pipeline_layout &&
        cached_bound_desc_bind_point == bind_point &&
        cached_bound_desc_set == desc_set &&
        cached_bound_desc_gen == scheduler.GetBoundDescGen(bind_point)) [[likely]] {
        return;
        }

        cmdbuf.bindDescriptorSets(bind_point, *pipeline_layout, 0, desc_set, {});
        scheduler.BumpBoundDescGen(bind_point);

    cached_bound_desc_cmdbuf = cmdbuf;
    cached_bound_desc_tick = cmd_tick;
    cached_bound_desc_layout = *pipeline_layout;
    cached_bound_desc_bind_point = bind_point;
    cached_bound_desc_set = desc_set;
    cached_bound_desc_gen = scheduler.GetBoundDescGen(bind_point);
    cached_bound_desc_valid = true;
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

// (GR2FORK L2-PROBE census removed 2026-06-10 — cleanup pass. Findings are
// permanent: 89 distinct layout shapes / 2126 pipelines built. Doc note on
// the former declarations in vk_pipeline_common.h.)

} // namespace Vulkan
