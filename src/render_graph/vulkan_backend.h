#pragma once

#include "backend.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>

namespace render_graph
{
    // Physical Resource Table for Vulkan (SoA)
    // This is NOT visible to the generic Render Graph, only to the Vulkan Backend.
    struct vulkan_image_table
    {
        std::vector<VkImage>        images;
        std::vector<VkImageView>    image_views;
        std::vector<VkDeviceMemory> memories;
        std::vector<VkFormat>       vk_formats; // Converted from generic Format

        void resize(size_t size)
        {
            images.resize(size, VK_NULL_HANDLE);
            image_views.resize(size, VK_NULL_HANDLE);
            memories.resize(size, VK_NULL_HANDLE);
            vk_formats.resize(size, VK_FORMAT_UNDEFINED);
        }

        void clear()
        {
            images.clear();
            image_views.clear();
            memories.clear();
            vk_formats.clear();
        }
    };

    class vulkan_backend : public backend
    {
    public:
        vulkan_image_table physical_images;
        // VulkanBufferTable physical_buffers;

        VkDevice device; // Assumed to be initialized

        // Helper to convert generic format to Vulkan format
        static VkFormat to_vk_format(format format)
        {
            switch (format)
            {
            case format::R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
            case format::R8G8B8A8_SRGB:  return VK_FORMAT_R8G8B8A8_SRGB;
            case format::D32_SFLOAT:     return VK_FORMAT_D32_SFLOAT;
            // ...
            default: return VK_FORMAT_UNDEFINED;
            }
        }

        static VkImageUsageFlags to_vk_usage(image_usage usage)
        {
            VkImageUsageFlags flags = 0;
            if ((uint32_t)usage & (uint32_t)image_usage::TRANSFER_SRC) flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if ((uint32_t)usage & (uint32_t)image_usage::SAMPLED)      flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
            // ...
            return flags;
        }

        void create_resources(const resource_meta_table& registry) override
        {
            // 1. Resize physical tables to match meta tables
            size_t image_count = registry.image_metas.names.size();
            physical_images.resize(image_count);

            // 2. Iterate and create resources
            for (size_t i = 0; i < image_count; ++i)
            {
                // Skip if imported or transient (handled differently)
                if (registry.image_metas.is_imported[i]) continue;

                // Convert Meta to Vulkan Create Info
                VkImageCreateInfo create_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
                create_info.imageType = VK_IMAGE_TYPE_2D;
                create_info.format = to_vk_format(registry.image_metas.formats[i]);
                create_info.extent = { .width=registry.image_metas.extents[i].width, .height=registry.image_metas.extents[i].height, .depth=registry.image_metas.extents[i].depth };
                create_info.mipLevels = registry.image_metas.mip_levels[i];
                create_info.arrayLayers = registry.image_metas.array_layers[i];
                create_info.samples = static_cast<VkSampleCountFlagBits>(registry.image_metas.sample_counts[i]);
                create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
                create_info.usage = to_vk_usage(registry.image_metas.usages[i]);
                
                // Store the converted format for later use
                physical_images.vk_formats[i] = create_info.format;

                // Actual Vulkan creation (simplified)
                // vkCreateImage(device, &create_info, nullptr, &physical_images.images[i]);
                // Allocate memory...
                // Bind memory...
                // Create ImageView...
                
                std::cout << "Created Vulkan Image: " << registry.image_metas.names[i] << '\n';
            }
        }

        void destroy_resources() override
        {
            // Loop and destroy
            physical_images.clear();
        }
    };
}
