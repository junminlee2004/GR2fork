// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Based on imgui_impl_vulkan.cpp from Dear ImGui repository

#include <cstdio>
#include <mutex>

#include <imgui.h>

#include "imgui_impl_vulkan.h"

#ifndef IM_MAX
#define IM_MAX(A, B) (((A) >= (B)) ? (A) : (B))
#endif

#define IDX_SIZE sizeof(ImDrawIdx)

namespace ImGui::Vulkan {

struct RenderBuffer {
    vk::DeviceMemory buffer_memory{};
    vk::DeviceSize buffer_size{};
    vk::Buffer buffer{};
};

// Reusable buffers used for rendering 1 current in-flight frame, for RenderDrawData()
struct FrameRenderBuffers {
    RenderBuffer vertex;
    RenderBuffer index;
};

// Each viewport will hold 1 WindowRenderBuffers
struct WindowRenderBuffers {
    uint32_t index{};
    uint32_t count{};
    std::vector<FrameRenderBuffers> frame_render_buffers{};
};

// Vulkan data
struct VkData {
    const InitInfo init_info;
    vk::DeviceSize buffer_memory_alignment = 256;
    vk::PipelineCreateFlags pipeline_create_flags{};
    vk::DescriptorPool descriptor_pool{};
    vk::DescriptorSetLayout descriptor_set_layout{};
    vk::PipelineLayout pipeline_layout{};
    vk::Pipeline pipeline{};
    vk::ShaderModule shader_module_vert{};
    vk::ShaderModule shader_module_frag{};

    std::mutex command_pool_mutex;
    vk::CommandPool command_pool{};
    vk::Sampler simple_sampler{};

    // Font data
    vk::DeviceMemory font_memory{};
    vk::Image font_image{};
    vk::ImageView font_view{};
    ImTextureID font_texture{};
    vk::CommandBuffer font_command_buffer{};

    // Render buffers
    WindowRenderBuffers render_buffers{};
    bool enabled_blending{true};

    VkData(const InitInfo init_info) : init_info(init_info) {
        render_buffers.count = init_info.image_count;
        render_buffers.frame_render_buffers.resize(render_buffers.count);
    }
};

//-----------------------------------------------------------------------------
// SHADERS
//-----------------------------------------------------------------------------

// backends/vulkan/glsl_shader.vert, compiled with:
// # glslangValidator -V -x -o glsl_shader.vert.u32 glsl_shader.vert
/*
#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;

out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;

void main()
{
    Out.Color = aColor;
    Out.UV = aUV;
    gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
}
*/
static uint32_t glsl_shader_vert_spv[] = {
    0x07230203, 0x00010000, 0x00080001, 0x0000002e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000a000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000000f, 0x00000015,
    0x0000001b, 0x0000001c, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00030005, 0x00000009, 0x00000000, 0x00050006, 0x00000009, 0x00000000, 0x6f6c6f43,
    0x00000072, 0x00040006, 0x00000009, 0x00000001, 0x00005655, 0x00030005, 0x0000000b, 0x0074754f,
    0x00040005, 0x0000000f, 0x6c6f4361, 0x0000726f, 0x00030005, 0x00000015, 0x00565561, 0x00060005,
    0x00000019, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x00000019, 0x00000000,
    0x505f6c67, 0x7469736f, 0x006e6f69, 0x00030005, 0x0000001b, 0x00000000, 0x00040005, 0x0000001c,
    0x736f5061, 0x00000000, 0x00060005, 0x0000001e, 0x73755075, 0x6e6f4368, 0x6e617473, 0x00000074,
    0x00050006, 0x0000001e, 0x00000000, 0x61635375, 0x0000656c, 0x00060006, 0x0000001e, 0x00000001,
    0x61725475, 0x616c736e, 0x00006574, 0x00030005, 0x00000020, 0x00006370, 0x00040047, 0x0000000b,
    0x0000001e, 0x00000000, 0x00040047, 0x0000000f, 0x0000001e, 0x00000002, 0x00040047, 0x00000015,
    0x0000001e, 0x00000001, 0x00050048, 0x00000019, 0x00000000, 0x0000000b, 0x00000000, 0x00030047,
    0x00000019, 0x00000002, 0x00040047, 0x0000001c, 0x0000001e, 0x00000000, 0x00050048, 0x0000001e,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000001e, 0x00000001, 0x00000023, 0x00000008,
    0x00030047, 0x0000001e, 0x00000002, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040017,
    0x00000008, 0x00000006, 0x00000002, 0x0004001e, 0x00000009, 0x00000007, 0x00000008, 0x00040020,
    0x0000000a, 0x00000003, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000003, 0x00040015,
    0x0000000c, 0x00000020, 0x00000001, 0x0004002b, 0x0000000c, 0x0000000d, 0x00000000, 0x00040020,
    0x0000000e, 0x00000001, 0x00000007, 0x0004003b, 0x0000000e, 0x0000000f, 0x00000001, 0x00040020,
    0x00000011, 0x00000003, 0x00000007, 0x0004002b, 0x0000000c, 0x00000013, 0x00000001, 0x00040020,
    0x00000014, 0x00000001, 0x00000008, 0x0004003b, 0x00000014, 0x00000015, 0x00000001, 0x00040020,
    0x00000017, 0x00000003, 0x00000008, 0x0003001e, 0x00000019, 0x00000007, 0x00040020, 0x0000001a,
    0x00000003, 0x00000019, 0x0004003b, 0x0000001a, 0x0000001b, 0x00000003, 0x0004003b, 0x00000014,
    0x0000001c, 0x00000001, 0x0004001e, 0x0000001e, 0x00000008, 0x00000008, 0x00040020, 0x0000001f,
    0x00000009, 0x0000001e, 0x0004003b, 0x0000001f, 0x00000020, 0x00000009, 0x00040020, 0x00000021,
    0x00000009, 0x00000008, 0x0004002b, 0x00000006, 0x00000028, 0x00000000, 0x0004002b, 0x00000006,
    0x00000029, 0x3f800000, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003d, 0x00000007, 0x00000010, 0x0000000f, 0x00050041, 0x00000011, 0x00000012,
    0x0000000b, 0x0000000d, 0x0003003e, 0x00000012, 0x00000010, 0x0004003d, 0x00000008, 0x00000016,
    0x00000015, 0x00050041, 0x00000017, 0x00000018, 0x0000000b, 0x00000013, 0x0003003e, 0x00000018,
    0x00000016, 0x0004003d, 0x00000008, 0x0000001d, 0x0000001c, 0x00050041, 0x00000021, 0x00000022,
    0x00000020, 0x0000000d, 0x0004003d, 0x00000008, 0x00000023, 0x00000022, 0x00050085, 0x00000008,
    0x00000024, 0x0000001d, 0x00000023, 0x00050041, 0x00000021, 0x00000025, 0x00000020, 0x00000013,
    0x0004003d, 0x00000008, 0x00000026, 0x00000025, 0x00050081, 0x00000008, 0x00000027, 0x00000024,
    0x00000026, 0x00050051, 0x00000006, 0x0000002a, 0x00000027, 0x00000000, 0x00050051, 0x00000006,
    0x0000002b, 0x00000027, 0x00000001, 0x00070050, 0x00000007, 0x0000002c, 0x0000002a, 0x0000002b,
    0x00000028, 0x00000029, 0x00050041, 0x00000011, 0x0000002d, 0x0000001b, 0x0000000d, 0x0003003e,
    0x0000002d, 0x0000002c, 0x000100fd, 0x00010038};

// backends/vulkan/glsl_shader.frag, compiled with:
// # glslangValidator -V -x -o glsl_shader.frag.u32 glsl_shader.frag
/*
#version 450 core
layout(location = 0) out vec4 fColor;
layout(set=0, binding=0) uniform sampler2D sTexture;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;
void main()
{
    fColor = In.Color * texture(sTexture, In.UV.st);
}
*/
static uint32_t glsl_shader_frag_spv[] = {
    0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000d, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6c6f4366, 0x0000726f, 0x00030005, 0x0000000b, 0x00000000,
    0x00050006, 0x0000000b, 0x00000000, 0x6f6c6f43, 0x00000072, 0x00040006, 0x0000000b, 0x00000001,
    0x00005655, 0x00030005, 0x0000000d, 0x00006e49, 0x00050005, 0x00000016, 0x78655473, 0x65727574,
    0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000d, 0x0000001e,
    0x00000000, 0x00040047, 0x00000016, 0x00000022, 0x00000000, 0x00040047, 0x00000016, 0x00000021,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003,
    0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040017, 0x0000000a, 0x00000006,
    0x00000002, 0x0004001e, 0x0000000b, 0x00000007, 0x0000000a, 0x00040020, 0x0000000c, 0x00000001,
    0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000001, 0x00040015, 0x0000000e, 0x00000020,
    0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040020, 0x00000010, 0x00000001,
    0x00000007, 0x00090019, 0x00000013, 0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00000000, 0x0003001b, 0x00000014, 0x00000013, 0x00040020, 0x00000015, 0x00000000,
    0x00000014, 0x0004003b, 0x00000015, 0x00000016, 0x00000000, 0x0004002b, 0x0000000e, 0x00000018,
    0x00000001, 0x00040020, 0x00000019, 0x00000001, 0x0000000a, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x00000010, 0x00000011, 0x0000000d,
    0x0000000f, 0x0004003d, 0x00000007, 0x00000012, 0x00000011, 0x0004003d, 0x00000014, 0x00000017,
    0x00000016, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x00000018, 0x0004003d, 0x0000000a,
    0x0000001b, 0x0000001a, 0x00050057, 0x00000007, 0x0000001c, 0x00000017, 0x0000001b, 0x00050085,
    0x00000007, 0x0000001d, 0x00000012, 0x0000001c, 0x0003003e, 0x00000009, 0x0000001d, 0x000100fd,
    0x00010038};

// Backend data stored in io.BackendRendererUserData to allow support for multiple Dear ImGui
// contexts It is STRONGLY preferred that you use docking branch with multi-viewports (== single
// Dear ImGui context + multiple windows) instead of multiple Dear ImGui contexts.
static VkData* GetBackendData() {
    return ImGui::GetCurrentContext() ? (VkData*)ImGui::GetIO().BackendRendererUserData : nullptr;
}

static uint32_t FindMemoryType(vk::MemoryPropertyFlags properties, uint32_t type_bits) {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;
    const auto prop = v.physical_device.getMemoryProperties();
    for (uint32_t i = 0; i < prop.memoryTypeCount; i++)
        if ((prop.memoryTypes[i].propertyFlags & properties) == properties && type_bits & (1 << i))
            return i;
    return 0xFFFFFFFF; // Unable to find memoryType
}

template <typename T>
static T CheckVkResult(T res) {
    return res;
}

template <typename T>
static T CheckVkResult(vk::ResultValue<T> res) {
    if (res.result != vk::Result::eSuccess) {
        const VkData* bd = GetBackendData();
        if (bd && bd->init_info.check_vk_result_fn) {
            bd->init_info.check_vk_result_fn(res.result);
        }
    }
    return std::move(res.value);
}

static void CheckVkErr(vk::Result res) {
    if (res == vk::Result::eSuccess) {
        return;
    }
    const VkData* bd = GetBackendData();
    if (!bd) {
        return;
    }
    const InitInfo& v = bd->init_info;
    if (v.check_vk_result_fn) {
        v.check_vk_result_fn(res);
    }
}

// Same as IM_MEMALIGN(). 'alignment' must be a power of two.
static inline vk::DeviceSize AlignBufferSize(vk::DeviceSize size, vk::DeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void UploadTextureData::Upload() {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;

    vk::SubmitInfo submit_info{};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    CheckVkErr(v.queue.submit({submit_info}));
    CheckVkErr(v.queue.waitIdle());

    v.device.destroyBuffer(upload_buffer, v.allocator);
    v.device.freeMemory(upload_buffer_memory, v.allocator);
    {
        std::unique_lock lk(bd->command_pool_mutex);
        v.device.freeCommandBuffers(bd->command_pool, {command_buffer});
    }
    upload_buffer = VK_NULL_HANDLE;
    upload_buffer_memory = VK_NULL_HANDLE;
}

void UploadTextureData::Destroy() {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;

    CheckVkErr(v.device.waitIdle());
    RemoveTexture(im_texture);
    im_texture = nullptr;

    v.device.destroyImageView(image_view, v.allocator);
    image_view = VK_NULL_HANDLE;
    v.device.destroyImage(image, v.allocator);
    image = VK_NULL_HANDLE;
    v.device.freeMemory(image_memory, v.allocator);
    image_memory = VK_NULL_HANDLE;
}

// Register a texture
ImTextureID AddTexture(vk::ImageView image_view, vk::ImageLayout image_layout,
                       vk::Sampler sampler) {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;

    if (sampler == VK_NULL_HANDLE) {
        sampler = bd->simple_sampler;
    }

    // Create Descriptor Set:
    vk::DescriptorSet descriptor_set;
    {
        vk::DescriptorSetAllocateInfo alloc_info{};
        alloc_info.descriptorPool = bd->descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &bd->descriptor_set_layout;
        descriptor_set = CheckVkResult(v.device.allocateDescriptorSets(alloc_info)).front();
    }

    // Update the Descriptor Set:
    {
        vk::DescriptorImageInfo desc_image[1]{};
        desc_image[0].sampler = sampler;
        desc_image[0].imageView = image_view;
        desc_image[0].imageLayout = image_layout;
        vk::WriteDescriptorSet write_desc[1]{};
        write_desc[0].dstSet = descriptor_set;
        write_desc[0].descriptorCount = 1;
        write_desc[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        write_desc[0].pImageInfo = desc_image;
        v.device.updateDescriptorSets({write_desc}, {});
    }
    return new Texture{
        .descriptor_set = descriptor_set,
    };
}
UploadTextureData UploadTexture(const void* data, vk::Format format, u32 width, u32 height,
                                size_t size) {
    ImGuiIO& io = GetIO();
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;

    UploadTextureData info{};
    {
        std::unique_lock lk(bd->command_pool_mutex);
        vk::CommandBufferAllocateInfo _di_tmp1{};
        _di_tmp1.commandPool = bd->command_pool;
        _di_tmp1.commandBufferCount = 1;
        info.command_buffer =
            CheckVkResult(v.device.allocateCommandBuffers(_di_tmp1))
                .front();
        vk::CommandBufferBeginInfo _di_tmp2{};
        _di_tmp2.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        CheckVkErr(info.command_buffer.begin(_di_tmp2));
    }

    // Create Image
    {
        vk::ImageCreateInfo image_info{};
        image_info.imageType = vk::ImageType::e2D;
        image_info.format = format;
        image_info.extent.width = width;
        image_info.extent.height = height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = vk::SampleCountFlagBits::e1;
        image_info.tiling = vk::ImageTiling::eOptimal;
        image_info.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
        image_info.sharingMode = vk::SharingMode::eExclusive;
        image_info.initialLayout = vk::ImageLayout::eUndefined;
        info.image = CheckVkResult(v.device.createImage(image_info, v.allocator));
        auto req = v.device.getImageMemoryRequirements(info.image);
        vk::MemoryAllocateInfo alloc_info{};
        alloc_info.allocationSize = IM_MAX(v.min_allocation_size, req.size);
        alloc_info.memoryTypeIndex = FindMemoryType(vk::MemoryPropertyFlagBits::eDeviceLocal, req.memoryTypeBits);
        info.image_memory = CheckVkResult(v.device.allocateMemory(alloc_info, v.allocator));
        CheckVkErr(v.device.bindImageMemory(info.image, info.image_memory, 0));
    }

    // Create Image View
    {
        vk::ImageViewCreateInfo view_info{};
        view_info.image = info.image;
        view_info.viewType = vk::ImageViewType::e2D;
        view_info.format = format;
        view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.layerCount = 1;
        info.image_view = CheckVkResult(v.device.createImageView(view_info, v.allocator));
    }

    // Create descriptor set (ImTextureID)
    info.im_texture = AddTexture(info.image_view, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create Upload Buffer
    {
        vk::BufferCreateInfo buffer_info{};
        buffer_info.size = size;
        buffer_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
        buffer_info.sharingMode = vk::SharingMode::eExclusive;
        info.upload_buffer = CheckVkResult(v.device.createBuffer(buffer_info, v.allocator));
        auto req = v.device.getBufferMemoryRequirements(info.upload_buffer);
        auto alignemtn = IM_MAX(bd->buffer_memory_alignment, req.alignment);
        vk::MemoryAllocateInfo alloc_info{};
        alloc_info.allocationSize = IM_MAX(v.min_allocation_size, req.size);
        alloc_info.memoryTypeIndex = FindMemoryType(vk::MemoryPropertyFlagBits::eHostVisible, req.memoryTypeBits);
        info.upload_buffer_memory = CheckVkResult(v.device.allocateMemory(alloc_info, v.allocator));
        CheckVkErr(v.device.bindBufferMemory(info.upload_buffer, info.upload_buffer_memory, 0));
    }

    // Upload to Buffer
    {
        char* map = (char*)CheckVkResult(v.device.mapMemory(info.upload_buffer_memory, 0, size));
        memcpy(map, data, size);
        vk::MappedMemoryRange range[1]{};
        range[0].memory = info.upload_buffer_memory;
        range[0].size = size;
        CheckVkErr(v.device.flushMappedMemoryRanges(range));
        v.device.unmapMemory(info.upload_buffer_memory);
    }

    // Copy to Image
    {
        vk::ImageMemoryBarrier copy_barrier[1]{};
        copy_barrier[0].dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        copy_barrier[0].oldLayout = vk::ImageLayout::eUndefined;
        copy_barrier[0].newLayout = vk::ImageLayout::eTransferDstOptimal;
        copy_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier[0].image = info.image;
        copy_barrier[0].subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        copy_barrier[0].subresourceRange.levelCount = 1;
        copy_barrier[0].subresourceRange.layerCount = 1;
        info.command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                                            {copy_barrier});

        vk::BufferImageCopy region{};
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = width;
        region.imageExtent.height = height;
        region.imageExtent.depth = 1;
        info.command_buffer.copyBufferToImage(info.upload_buffer, info.image,
                                              vk::ImageLayout::eTransferDstOptimal, {region});

        vk::ImageMemoryBarrier use_barrier[1]{};
        use_barrier[0].srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        use_barrier[0].dstAccessMask = vk::AccessFlagBits::eShaderRead;
        use_barrier[0].oldLayout = vk::ImageLayout::eTransferDstOptimal;
        use_barrier[0].newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier[0].image = info.image;
        use_barrier[0].subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        use_barrier[0].subresourceRange.levelCount = 1;
        use_barrier[0].subresourceRange.layerCount = 1;
        info.command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                            vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
                                            {use_barrier});
    }

    CheckVkErr(info.command_buffer.end());

    return info;
}

void RemoveTexture(ImTextureID texture) {
    IM_ASSERT(texture != nullptr);
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;
    CheckVkErr(v.device.freeDescriptorSets(bd->descriptor_pool, {texture->descriptor_set}));
    delete texture;
}

static void CreateOrResizeBuffer(RenderBuffer& rb, size_t new_size, vk::BufferUsageFlagBits usage) {
    VkData* bd = GetBackendData();
    IM_ASSERT(bd != nullptr);
    const InitInfo& v = bd->init_info;
    if (rb.buffer != VK_NULL_HANDLE) {
        v.device.destroyBuffer(rb.buffer, v.allocator);
    }
    if (rb.buffer_memory != VK_NULL_HANDLE) {
        v.device.freeMemory(rb.buffer_memory, v.allocator);
    }

    const vk::DeviceSize buffer_size_aligned =
        AlignBufferSize(IM_MAX(v.min_allocation_size, new_size), bd->buffer_memory_alignment);
    vk::BufferCreateInfo buffer_info{};
    buffer_info.size = buffer_size_aligned;
    buffer_info.usage = usage;
    buffer_info.sharingMode = vk::SharingMode::eExclusive;
    rb.buffer = CheckVkResult(v.device.createBuffer(buffer_info, v.allocator));

    const vk::MemoryRequirements req = v.device.getBufferMemoryRequirements(rb.buffer);
    bd->buffer_memory_alignment = IM_MAX(bd->buffer_memory_alignment, req.alignment);
    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = FindMemoryType(vk::MemoryPropertyFlagBits::eHostVisible, req.memoryTypeBits);
    rb.buffer_memory = CheckVkResult(v.device.allocateMemory(alloc_info, v.allocator));

    CheckVkErr(v.device.bindBufferMemory(rb.buffer, rb.buffer_memory, 0));
    rb.buffer_size = buffer_size_aligned;
}

static void SetupRenderState(ImDrawData& draw_data, vk::Pipeline pipeline, vk::CommandBuffer cmdbuf,
                             FrameRenderBuffers& frb, int fb_width, int fb_height) {
    VkData* bd = GetBackendData();

    // Bind pipeline:
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    // Bind Vertex And Index Buffer:
    if (draw_data.TotalVtxCount > 0) {
        vk::Buffer vertex_buffers[1] = {frb.vertex.buffer};
        vk::DeviceSize vertex_offset[1] = {0};
        cmdbuf.bindVertexBuffers(0, {vertex_buffers}, vertex_offset);
        cmdbuf.bindIndexBuffer(frb.index.buffer, 0,
                               IDX_SIZE == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
    }

    // Setup viewport:
    {
        vk::Viewport viewport{};
        viewport.x = 0;
        viewport.y = 0;
        viewport.width = (float)fb_width;
        viewport.height = (float)fb_height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        cmdbuf.setViewport(0, {viewport});
    }

    // Setup scale and translation:
    // Our visible imgui space lies from draw_data->DisplayPps (top left) to
    // draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single
    // viewport apps.
    {
        float scale[2];
        scale[0] = 2.0f / draw_data.DisplaySize.x;
        scale[1] = 2.0f / draw_data.DisplaySize.y;
        float translate[2];
        translate[0] = -1.0f - draw_data.DisplayPos.x * scale[0];
        translate[1] = -1.0f - draw_data.DisplayPos.y * scale[1];
        cmdbuf.pushConstants(bd->pipeline_layout, vk::ShaderStageFlagBits::eVertex,
                             sizeof(float) * 0, sizeof(float) * 2, scale);
        cmdbuf.pushConstants(bd->pipeline_layout, vk::ShaderStageFlagBits::eVertex,
                             sizeof(float) * 2, sizeof(float) * 2, translate);
    }
}

// Render function
void RenderDrawData(ImDrawData& draw_data, vk::CommandBuffer command_buffer,
                    vk::Pipeline pipeline) {
    // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates !=
    // framebuffer coordinates)
    int fb_width = (int)(draw_data.DisplaySize.x * draw_data.FramebufferScale.x);
    int fb_height = (int)(draw_data.DisplaySize.y * draw_data.FramebufferScale.y);
    if (fb_width <= 0 || fb_height <= 0) {
        return;
    }

    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;
    if (pipeline == VK_NULL_HANDLE) {
        pipeline = bd->pipeline;
    }

    // Allocate array to store enough vertex/index buffers
    WindowRenderBuffers& wrb = bd->render_buffers;
    wrb.index = (wrb.index + 1) % wrb.count;
    FrameRenderBuffers& frb = wrb.frame_render_buffers[wrb.index];

    if (draw_data.TotalVtxCount > 0) {
        // Create or resize the vertex/index buffers
        size_t vertex_size = AlignBufferSize(draw_data.TotalVtxCount * sizeof(ImDrawVert),
                                             bd->buffer_memory_alignment);
        size_t index_size =
            AlignBufferSize(draw_data.TotalIdxCount * IDX_SIZE, bd->buffer_memory_alignment);
        if (frb.vertex.buffer == VK_NULL_HANDLE || frb.vertex.buffer_size < vertex_size) {
            CreateOrResizeBuffer(frb.vertex, vertex_size, vk::BufferUsageFlagBits::eVertexBuffer);
        }
        if (frb.index.buffer == VK_NULL_HANDLE || frb.index.buffer_size < index_size) {
            CreateOrResizeBuffer(frb.index, index_size, vk::BufferUsageFlagBits::eIndexBuffer);
        }

        // Upload vertex/index data into a single contiguous GPU buffer
        ImDrawVert* vtx_dst = nullptr;
        ImDrawIdx* idx_dst = nullptr;
        vtx_dst = (ImDrawVert*)CheckVkResult(
            v.device.mapMemory(frb.vertex.buffer_memory, 0, vertex_size, vk::MemoryMapFlags{}));
        idx_dst = (ImDrawIdx*)CheckVkResult(
            v.device.mapMemory(frb.index.buffer_memory, 0, index_size, vk::MemoryMapFlags{}));
        for (int n = 0; n < draw_data.CmdListsCount; n++) {
            const ImDrawList* cmd_list = draw_data.CmdLists[n];
            memcpy(vtx_dst, cmd_list->VtxBuffer.Data,
                   cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
            memcpy(idx_dst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * IDX_SIZE);
            vtx_dst += cmd_list->VtxBuffer.Size;
            idx_dst += cmd_list->IdxBuffer.Size;
        }
        vk::MappedMemoryRange range[2]{};
        range[0].memory = frb.vertex.buffer_memory;
        range[0].size = VK_WHOLE_SIZE;
        range[1].memory = frb.index.buffer_memory;
        range[1].size = VK_WHOLE_SIZE;
        CheckVkErr(v.device.flushMappedMemoryRanges({range}));
        v.device.unmapMemory(frb.vertex.buffer_memory);
        v.device.unmapMemory(frb.index.buffer_memory);
    }

    // Setup desired Vulkan state
    SetupRenderState(draw_data, pipeline, command_buffer, frb, fb_width, fb_height);

    // Will project scissor/clipping rectangles into framebuffer space

    // (0,0) unless using multi-viewports
    ImVec2 clip_off = draw_data.DisplayPos;
    // (1,1) unless using retina display which are often (2,2)
    ImVec2 clip_scale = draw_data.FramebufferScale;

    // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own offset into them)
    int global_vtx_offset = 0;
    int global_idx_offset = 0;
    for (int n = 0; n < draw_data.CmdListsCount; n++) {
        const ImDrawList* cmd_list = draw_data.CmdLists[n];
        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback != nullptr) {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to
                // request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState) {
                    SetupRenderState(draw_data, pipeline, command_buffer, frb, fb_width, fb_height);
                } else {
                    pcmd->UserCallback(cmd_list, pcmd);
                }
            } else {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x,
                                (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
                ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x,
                                (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);

                // Clamp to viewport as vk::CmdSetScissor() won't accept values that are off bounds
                if (clip_min.x < 0.0f) {
                    clip_min.x = 0.0f;
                }
                if (clip_min.y < 0.0f) {
                    clip_min.y = 0.0f;
                }
                if (clip_max.x > fb_width) {
                    clip_max.x = (float)fb_width;
                }
                if (clip_max.y > fb_height) {
                    clip_max.y = (float)fb_height;
                }
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                    continue;

                // Apply scissor/clipping rectangle
                vk::Rect2D scissor{};
                scissor.offset.x = (int32_t)(clip_min.x);
                scissor.offset.y = (int32_t)(clip_min.y);
                scissor.extent.width = (uint32_t)(clip_max.x - clip_min.x);
                scissor.extent.height = (uint32_t)(clip_max.y - clip_min.y);
                command_buffer.setScissor(0, 1, &scissor);

                // Bind DescriptorSet with font or user texture
                vk::DescriptorSet desc_set[1]{pcmd->TextureId->descriptor_set};
                command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                                  bd->pipeline_layout, 0, {desc_set}, {});

                // Draw
                command_buffer.drawIndexed(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset,
                                           pcmd->VtxOffset + global_vtx_offset, 0);
            }
        }
        global_idx_offset += cmd_list->IdxBuffer.Size;
        global_vtx_offset += cmd_list->VtxBuffer.Size;
    }
    //    vk::Rect2D scissor = {{0, 0}, {(uint32_t)fb_width, (uint32_t)fb_height}};
    //    command_buffer.setScissor(0, 1, &scissor);
}

static void DestroyFontsTexture();

static bool CreateFontsTexture() {
    ImGuiIO& io = ImGui::GetIO();
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;

    // Destroy existing texture (if any)
    if (bd->font_view || bd->font_image || bd->font_memory || bd->font_texture) {
        CheckVkErr(v.queue.waitIdle());
        DestroyFontsTexture();
    }

    // Create command buffer
    if (bd->font_command_buffer == VK_NULL_HANDLE) {
        vk::CommandBufferAllocateInfo info{};
        info.commandPool = bd->command_pool;
        info.commandBufferCount = 1;
        std::unique_lock lk(bd->command_pool_mutex);
        bd->font_command_buffer = CheckVkResult(v.device.allocateCommandBuffers(info)).front();
    }

    // Start command buffer
    {
        CheckVkErr(bd->font_command_buffer.reset());
        vk::CommandBufferBeginInfo begin_info{};
        begin_info.flags |= vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        CheckVkErr(bd->font_command_buffer.begin(begin_info));
    }

    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    size_t upload_size = width * height * 4 * sizeof(char);

    // Create the Image:
    {
        vk::ImageCreateInfo info{};
        info.imageType = vk::ImageType::e2D;
        info.format = vk::Format::eR8G8B8A8Unorm;
        info.extent.width = static_cast<uint32_t>(width);
        info.extent.height = static_cast<uint32_t>(height);
        info.extent.depth = 1;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = vk::SampleCountFlagBits::e1;
        info.tiling = vk::ImageTiling::eOptimal;
        info.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        info.sharingMode = vk::SharingMode::eExclusive;
        info.initialLayout = vk::ImageLayout::eUndefined;
        bd->font_image = CheckVkResult(v.device.createImage(info, v.allocator));
        vk::MemoryRequirements req = v.device.getImageMemoryRequirements(bd->font_image);
        vk::MemoryAllocateInfo alloc_info{};
        alloc_info.allocationSize = IM_MAX(v.min_allocation_size, req.size);
        alloc_info.memoryTypeIndex = FindMemoryType(vk::MemoryPropertyFlagBits::eDeviceLocal, req.memoryTypeBits);
        bd->font_memory = CheckVkResult(v.device.allocateMemory(alloc_info, v.allocator));
        CheckVkErr(v.device.bindImageMemory(bd->font_image, bd->font_memory, 0));
    }

    // Create the Image View:
    {
        vk::ImageViewCreateInfo info{};
        info.image = bd->font_image;
        info.viewType = vk::ImageViewType::e2D;
        info.format = vk::Format::eR8G8B8A8Unorm;
        info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        bd->font_view = CheckVkResult(v.device.createImageView(info, v.allocator));
    }

    // Create the Descriptor Set:
    bd->font_texture = AddTexture(bd->font_view, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create the Upload Buffer:
    vk::DeviceMemory upload_buffer_memory{};
    vk::Buffer upload_buffer{};
    {
        vk::BufferCreateInfo buffer_info{};
        buffer_info.size = upload_size;
        buffer_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
        buffer_info.sharingMode = vk::SharingMode::eExclusive;
        upload_buffer = CheckVkResult(v.device.createBuffer(buffer_info, v.allocator));
        vk::MemoryRequirements req = v.device.getBufferMemoryRequirements(upload_buffer);
        bd->buffer_memory_alignment = IM_MAX(bd->buffer_memory_alignment, req.alignment);
        vk::MemoryAllocateInfo alloc_info{};
        alloc_info.allocationSize = IM_MAX(v.min_allocation_size, req.size);
        alloc_info.memoryTypeIndex = FindMemoryType(vk::MemoryPropertyFlagBits::eHostVisible, req.memoryTypeBits);
        upload_buffer_memory = CheckVkResult(v.device.allocateMemory(alloc_info, v.allocator));
        CheckVkErr(v.device.bindBufferMemory(upload_buffer, upload_buffer_memory, 0));
    }

    // Upload to Buffer:
    {
        char* map = (char*)CheckVkResult(v.device.mapMemory(upload_buffer_memory, 0, upload_size));
        memcpy(map, pixels, upload_size);
        vk::MappedMemoryRange range[1]{};
        range[0].memory = upload_buffer_memory;
        range[0].size = upload_size;
        CheckVkErr(v.device.flushMappedMemoryRanges({range}));
        v.device.unmapMemory(upload_buffer_memory);
    }

    // Copy to Image:
    {
        vk::ImageMemoryBarrier copy_barrier[1]{};
        copy_barrier[0].dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        copy_barrier[0].oldLayout = vk::ImageLayout::eUndefined;
        copy_barrier[0].newLayout = vk::ImageLayout::eTransferDstOptimal;
        copy_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier[0].image = bd->font_image;
        copy_barrier[0].subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        copy_barrier[0].subresourceRange.levelCount = 1;
        copy_barrier[0].subresourceRange.layerCount = 1;
        bd->font_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                                vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                                                {copy_barrier});

        vk::BufferImageCopy region{};
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = static_cast<uint32_t>(width);
        region.imageExtent.height = static_cast<uint32_t>(height);
        region.imageExtent.depth = 1;
        bd->font_command_buffer.copyBufferToImage(upload_buffer, bd->font_image,
                                                  vk::ImageLayout::eTransferDstOptimal, {region});

        vk::ImageMemoryBarrier use_barrier[1]{};
        use_barrier[0].srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        use_barrier[0].dstAccessMask = vk::AccessFlagBits::eShaderRead;
        use_barrier[0].oldLayout = vk::ImageLayout::eTransferDstOptimal;
        use_barrier[0].newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier[0].image = bd->font_image;
        use_barrier[0].subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        use_barrier[0].subresourceRange.levelCount = 1;
        use_barrier[0].subresourceRange.layerCount = 1;
        bd->font_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                vk::PipelineStageFlagBits::eFragmentShader, {}, {},
                                                {}, {use_barrier});
    }

    // Store our identifier
    io.Fonts->SetTexID(bd->font_texture);

    // End command buffer
    vk::SubmitInfo end_info = {};
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &bd->font_command_buffer;
    CheckVkErr(bd->font_command_buffer.end());
    CheckVkErr(v.queue.submit({end_info}));

    CheckVkErr(v.queue.waitIdle());

    v.device.destroyBuffer(upload_buffer, v.allocator);
    v.device.freeMemory(upload_buffer_memory, v.allocator);

    return true;
}

// You probably never need to call this, as it is called by CreateFontsTexture()
// and Shutdown().
static void DestroyFontsTexture() {
    ImGuiIO& io = ImGui::GetIO();
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;

    if (bd->font_texture) {
        RemoveTexture(bd->font_texture);
        bd->font_texture = nullptr;
        io.Fonts->SetTexID(nullptr);
    }

    if (bd->font_view) {
        v.device.destroyImageView(bd->font_view, v.allocator);
        bd->font_view = VK_NULL_HANDLE;
    }
    if (bd->font_image) {
        v.device.destroyImage(bd->font_image, v.allocator);
        bd->font_image = VK_NULL_HANDLE;
    }
    if (bd->font_memory) {
        v.device.freeMemory(bd->font_memory, v.allocator);
        bd->font_memory = VK_NULL_HANDLE;
    }
}

static void DestroyFrameRenderBuffers(vk::Device device, RenderBuffer& rb,
                                      const vk::AllocationCallbacks* allocator) {
    if (rb.buffer) {
        device.destroyBuffer(rb.buffer, allocator);
        rb.buffer = VK_NULL_HANDLE;
    }
    if (rb.buffer_memory) {
        device.freeMemory(rb.buffer_memory, allocator);
        rb.buffer_memory = VK_NULL_HANDLE;
    }
    rb.buffer_size = 0;
}

static void DestroyWindowRenderBuffers(vk::Device device, WindowRenderBuffers& buffers,
                                       const vk::AllocationCallbacks* allocator) {
    for (uint32_t n = 0; n < buffers.count; n++) {
        auto& frb = buffers.frame_render_buffers[n];
        DestroyFrameRenderBuffers(device, frb.index, allocator);
        DestroyFrameRenderBuffers(device, frb.vertex, allocator);
    }
    buffers = {};
}

static void CreateShaderModules(vk::Device device, const vk::AllocationCallbacks* allocator) {
    // Create the shader modules
    VkData* bd = GetBackendData();
    if (bd->shader_module_vert == VK_NULL_HANDLE) {
        vk::ShaderModuleCreateInfo vert_info{};
        vert_info.codeSize = sizeof(glsl_shader_vert_spv);
        vert_info.pCode = (uint32_t*)glsl_shader_vert_spv;
        bd->shader_module_vert = CheckVkResult(device.createShaderModule(vert_info, allocator));
    }
    if (bd->shader_module_frag == VK_NULL_HANDLE) {
        vk::ShaderModuleCreateInfo frag_info{};
        frag_info.codeSize = sizeof(glsl_shader_frag_spv);
        frag_info.pCode = (uint32_t*)glsl_shader_frag_spv;
        bd->shader_module_frag = CheckVkResult(device.createShaderModule(frag_info, allocator));
    }
}

static void CreatePipeline(vk::Device device, const vk::AllocationCallbacks* allocator,
                           vk::PipelineCache pipeline_cache, vk::RenderPass render_pass,
                           vk::Pipeline* pipeline, uint32_t subpass) {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;

    CreateShaderModules(device, allocator);

    vk::PipelineShaderStageCreateInfo stage[2]{};
    stage[0].stage = vk::ShaderStageFlagBits::eVertex;
    stage[0].module = bd->shader_module_vert;
    stage[0].pName = "main";
    stage[1].stage = vk::ShaderStageFlagBits::eFragment;
    stage[1].module = bd->shader_module_frag;
    stage[1].pName = "main";

    vk::VertexInputBindingDescription binding_desc[1]{};
    binding_desc[0].stride = sizeof(ImDrawVert);
    binding_desc[0].inputRate = vk::VertexInputRate::eVertex;

    vk::VertexInputAttributeDescription attribute_desc[3]{};
    attribute_desc[0].location = 0;
    attribute_desc[0].binding = binding_desc[0].binding;
    attribute_desc[0].format = vk::Format::eR32G32Sfloat;
    attribute_desc[0].offset = offsetof(ImDrawVert, pos);
    attribute_desc[1].location = 1;
    attribute_desc[1].binding = binding_desc[0].binding;
    attribute_desc[1].format = vk::Format::eR32G32Sfloat;
    attribute_desc[1].offset = offsetof(ImDrawVert, uv);
    attribute_desc[2].location = 2;
    attribute_desc[2].binding = binding_desc[0].binding;
    attribute_desc[2].format = vk::Format::eR8G8B8A8Unorm;
    attribute_desc[2].offset = offsetof(ImDrawVert, col);

    vk::PipelineVertexInputStateCreateInfo vertex_info{};
    vertex_info.vertexBindingDescriptionCount = 1;
    vertex_info.pVertexBindingDescriptions = binding_desc;
    vertex_info.vertexAttributeDescriptionCount = 3;
    vertex_info.pVertexAttributeDescriptions = attribute_desc;

    vk::PipelineInputAssemblyStateCreateInfo ia_info{};
    ia_info.topology = vk::PrimitiveTopology::eTriangleList;

    vk::PipelineViewportStateCreateInfo viewport_info{};
    viewport_info.viewportCount = 1;
    viewport_info.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo raster_info{};
    raster_info.polygonMode = vk::PolygonMode::eFill;
    raster_info.cullMode = vk::CullModeFlagBits::eNone;
    raster_info.frontFace = vk::FrontFace::eCounterClockwise;
    raster_info.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo ms_info{};
    ms_info.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState color_attachment[1]{};
    color_attachment[0].blendEnable = VK_TRUE;
    color_attachment[0].srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    color_attachment[0].dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    color_attachment[0].colorBlendOp = vk::BlendOp::eAdd;
    color_attachment[0].srcAlphaBlendFactor = vk::BlendFactor::eOne;
    color_attachment[0].dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    color_attachment[0].alphaBlendOp = vk::BlendOp::eAdd;
    color_attachment[0].colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    vk::PipelineDepthStencilStateCreateInfo depth_info{};

    vk::PipelineColorBlendStateCreateInfo blend_info{};
    blend_info.attachmentCount = 1;
    blend_info.pAttachments = color_attachment;

    vk::DynamicState dynamic_states[2]{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    vk::PipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.dynamicStateCount = (uint32_t)IM_ARRAYSIZE(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

    vk::GraphicsPipelineCreateInfo info{};
    info.pNext = &v.pipeline_rendering_create_info;
    info.flags = bd->pipeline_create_flags;
    info.stageCount = 2;
    info.pStages = stage;
    info.pVertexInputState = &vertex_info;
    info.pInputAssemblyState = &ia_info;
    info.pViewportState = &viewport_info;
    info.pRasterizationState = &raster_info;
    info.pMultisampleState = &ms_info;
    info.pDepthStencilState = &depth_info;
    info.pColorBlendState = &blend_info;
    info.pDynamicState = &dynamic_state;
    info.layout = bd->pipeline_layout;
    info.renderPass = render_pass;
    info.subpass = subpass;

    *pipeline =
        CheckVkResult(device.createGraphicsPipelines(pipeline_cache, {info}, allocator)).front();
}

bool CreateDeviceObjects() {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;
    vk::Result err;

    if (!bd->descriptor_pool) {
        // large enough descriptor pool
        vk::DescriptorPoolSize pool_sizes[]{
            {vk::DescriptorType::eSampler, 1000},
            {vk::DescriptorType::eCombinedImageSampler, 1000},
            {vk::DescriptorType::eSampledImage, 1000},
            {vk::DescriptorType::eStorageImage, 1000},
            {vk::DescriptorType::eUniformTexelBuffer, 1000},
            {vk::DescriptorType::eStorageTexelBuffer, 1000},
            {vk::DescriptorType::eUniformBuffer, 1000},
            {vk::DescriptorType::eStorageBuffer, 1000},
            {vk::DescriptorType::eUniformBufferDynamic, 1000},
            {vk::DescriptorType::eStorageBufferDynamic, 1000},
            {vk::DescriptorType::eInputAttachment, 1000},
        };

        vk::DescriptorPoolCreateInfo pool_info{};
        pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        pool_info.maxSets = 1000;
        pool_info.poolSizeCount = std::size(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;

        bd->descriptor_pool = CheckVkResult(v.device.createDescriptorPool(pool_info));
    }

    if (!bd->descriptor_set_layout) {
        vk::DescriptorSetLayoutBinding binding[1]{};
        binding[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        binding[0].descriptorCount = 1;
        binding[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
        vk::DescriptorSetLayoutCreateInfo info{};
        info.bindingCount = 1;
        info.pBindings = binding;
        bd->descriptor_set_layout =
            CheckVkResult(v.device.createDescriptorSetLayout(info, v.allocator));
    }

    if (!bd->pipeline_layout) {
        // Constants: we are using 'vec2 offset' and 'vec2 scale' instead of a full 3d projection
        // matrix
        vk::PushConstantRange push_constants[1]{};
        push_constants[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
        push_constants[0].offset = sizeof(float) * 0;
        push_constants[0].size = sizeof(float) * 4;
        vk::DescriptorSetLayout set_layout[1] = {bd->descriptor_set_layout};
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = push_constants;
        bd->pipeline_layout =
            CheckVkResult(v.device.createPipelineLayout(layout_info, v.allocator));
    }

    CreatePipeline(v.device, v.allocator, v.pipeline_cache, nullptr, &bd->pipeline, v.subpass);

    if (bd->command_pool == VK_NULL_HANDLE) {
        vk::CommandPoolCreateInfo info{};
        info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        info.queueFamilyIndex = v.queue_family;
        std::unique_lock lk(bd->command_pool_mutex);
        bd->command_pool = CheckVkResult(v.device.createCommandPool(info, v.allocator));
    }

    if (!bd->simple_sampler) {
        // Bilinear sampling is required by default. Set 'io.Fonts->Flags |=
        // ImFontAtlasFlags_NoBakedLines' or 'style.AntiAliasedLinesUseTex = false' to allow
        // point/nearest sampling.
        vk::SamplerCreateInfo info{};
        info.magFilter = vk::Filter::eLinear;
        info.minFilter = vk::Filter::eLinear;
        info.mipmapMode = vk::SamplerMipmapMode::eLinear;
        info.addressModeU = vk::SamplerAddressMode::eRepeat;
        info.addressModeV = vk::SamplerAddressMode::eRepeat;
        info.addressModeW = vk::SamplerAddressMode::eRepeat;
        info.maxAnisotropy = 1.0f;
        info.minLod = -1000;
        info.maxLod = 1000;
        bd->simple_sampler = CheckVkResult(v.device.createSampler(info, v.allocator));
    }

    return true;
}

void ImGuiImplVulkanDestroyDeviceObjects() {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;
    DestroyWindowRenderBuffers(v.device, bd->render_buffers, v.allocator);
    DestroyFontsTexture();

    if (bd->font_command_buffer) {
        std::unique_lock lk(bd->command_pool_mutex);
        v.device.freeCommandBuffers(bd->command_pool, {bd->font_command_buffer});
        bd->font_command_buffer = VK_NULL_HANDLE;
    }
    if (bd->command_pool) {
        std::unique_lock lk(bd->command_pool_mutex);
        v.device.destroyCommandPool(bd->command_pool, v.allocator);
        bd->command_pool = VK_NULL_HANDLE;
    }
    if (bd->shader_module_vert) {
        v.device.destroyShaderModule(bd->shader_module_vert, v.allocator);
        bd->shader_module_vert = VK_NULL_HANDLE;
    }
    if (bd->shader_module_frag) {
        v.device.destroyShaderModule(bd->shader_module_frag, v.allocator);
        bd->shader_module_frag = VK_NULL_HANDLE;
    }
    if (bd->simple_sampler) {
        v.device.destroySampler(bd->simple_sampler, v.allocator);
        bd->simple_sampler = VK_NULL_HANDLE;
    }
    if (bd->descriptor_set_layout) {
        v.device.destroyDescriptorSetLayout(bd->descriptor_set_layout, v.allocator);
        bd->descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (bd->pipeline_layout) {
        v.device.destroyPipelineLayout(bd->pipeline_layout, v.allocator);
        bd->pipeline_layout = VK_NULL_HANDLE;
    }
    if (bd->pipeline) {
        v.device.destroyPipeline(bd->pipeline, v.allocator);
        bd->pipeline = VK_NULL_HANDLE;
    }
}

bool Init(InitInfo info) {

    IM_ASSERT(info.instance != VK_NULL_HANDLE);
    IM_ASSERT(info.physical_device != VK_NULL_HANDLE);
    IM_ASSERT(info.device != VK_NULL_HANDLE);
    IM_ASSERT(info.image_count >= 2);

    ImGuiIO& io = ImGui::GetIO();
    IMGUI_CHECKVERSION();
    IM_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized a renderer backend!");

    // Setup backend capabilities flags
    auto* bd = IM_NEW(VkData)(info);
    io.BackendRendererUserData = (void*)bd;
    io.BackendRendererName = "imgui_impl_vulkan_shadps4";
    // We can honor the ImDrawCmd::VtxOffset field, allowing for large meshes.
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

    CreateDeviceObjects();
    CreateFontsTexture();

    return true;
}

void Shutdown() {
    VkData* bd = GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();

    ImGuiImplVulkanDestroyDeviceObjects();
    io.BackendRendererName = nullptr;
    io.BackendRendererUserData = nullptr;
    io.BackendFlags &= ~ImGuiBackendFlags_RendererHasVtxOffset;
    IM_DELETE(bd);
}

void OnSurfaceFormatChange(vk::Format surface_format) {
    VkData* bd = GetBackendData();
    const InitInfo& v = bd->init_info;
    auto& pl_format = const_cast<vk::Format&>(
        bd->init_info.pipeline_rendering_create_info.pColorAttachmentFormats[0]);
    if (pl_format != surface_format) {
        pl_format = surface_format;
        if (bd->pipeline) {
            v.device.destroyPipeline(bd->pipeline, v.allocator);
            bd->pipeline = VK_NULL_HANDLE;
            CreatePipeline(v.device, v.allocator, v.pipeline_cache, nullptr, &bd->pipeline,
                           v.subpass);
        }
    }
}

} // namespace ImGui::Vulkan
