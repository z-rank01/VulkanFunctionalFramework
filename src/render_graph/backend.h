#pragma once

#include "resource.h"

namespace render_graph
{
    // Abstract interface for the render backend (Vulkan, DX12, Metal)
    // The backend is responsible for allocating physical resources based on the Meta Registry.
    class backend
    {
    public:
        virtual ~backend() = default;

        // Called after the Render Graph is compiled and culled.
        // The backend should iterate over the registry and create resources that are not culled.
        virtual void create_resources(const resource_meta_table& registry) = 0;

        // Called to destroy resources
        virtual void destroy_resources() = 0;

        // ... execute_pass, etc.
    };
}
