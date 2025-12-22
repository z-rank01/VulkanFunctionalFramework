#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "resource_types.h"

namespace render_graph
{
    using resource_handle = uint32_t;
    using pass_handle = uint32_t;

    // Helper struct for user convenience
    struct image_info
    {
        std::string name;
        format fmt             = format::UNDEFINED;
        extent_3d extent       = {.width = 1, .height = 1, .depth = 1};
        image_usage usage      = image_usage::NONE;
        image_type type        = image_type::TYPE_2D;
        image_flags flags      = image_flags::NONE;
        uint32_t mip_levels    = 1;
        uint32_t array_layers  = 1;
        uint32_t sample_counts = 1;
        bool imported;
    };

    struct buffer_info
    {
        std::string name;
        uint64_t size      = 0;
        buffer_usage usage = buffer_usage::NONE;
        bool imported;
    };

    // Meta Table for Images (SoA)
    // Stores all creation information required to create the physical resource later.
    struct image_meta
    {
        // Generic properties (Cross-API)
        std::vector<std::string> names;
        std::vector<format> formats;
        std::vector<extent_3d> extents;
        std::vector<image_usage> usages;
        std::vector<image_type> types;
        std::vector<image_flags> flags;
        std::vector<uint32_t> mip_levels;
        std::vector<uint32_t> array_layers;
        std::vector<uint32_t> sample_counts;

        // Lifecycle / Graph properties
        std::vector<bool> is_imported;  // If true, handle is provided externally (backbuffer, etc.)
        std::vector<bool> is_transient; // If true, memory can be aliased/lazy allocated

        // Helper to add a new image meta and return its handle (index)
        resource_handle add(const image_info& info)
        {
            auto handle = static_cast<resource_handle>(names.size());
            names.push_back(info.name);
            formats.push_back(info.fmt);
            extents.push_back(info.extent);
            usages.push_back(info.usage);
            types.push_back(info.type);
            flags.push_back(info.flags);
            mip_levels.push_back(info.mip_levels);
            array_layers.push_back(info.array_layers);
            sample_counts.push_back(info.sample_counts);

            // Defaults
            is_imported.push_back(info.imported);
            is_transient.push_back(!info.imported);

            return handle;
        }

        void clear()
        {
            names.clear();
            formats.clear();
            extents.clear();
            usages.clear();
            types.clear();
            flags.clear();
            mip_levels.clear();
            array_layers.clear();
            sample_counts.clear();
            is_imported.clear();
            is_transient.clear();
        }
    };

    // Meta Table for Buffers (SoA)
    struct buffer_meta
    {
        std::vector<std::string> names;
        std::vector<uint64_t> sizes;
        std::vector<buffer_usage> usages;

        resource_handle add(const buffer_info& info)
        {
            auto handle = static_cast<resource_handle>(names.size());
            names.push_back(info.name);
            sizes.push_back(info.size);
            usages.push_back(info.usage);
            return handle;
        }

        void clear()
        {
            names.clear();
            sizes.clear();
            usages.clear();
        }
    };

    // The main registry that holds all resource descriptions
    struct resource_meta_table
    {
        image_meta image_metas;
        buffer_meta buffer_metas;

        void clear()
        {
            image_metas.clear();
            buffer_metas.clear();
        }
    };

    struct resource_producer_lookup_table
    {
        std::vector<pass_handle> img_proc_map; // Indexed by image handle
        std::vector<pass_handle> buf_proc_map; // Indexed by buffer handle
    };

} // namespace render_graph
