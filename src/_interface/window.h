#pragma once

#include <vulkan/vulkan.h>

#include <functional>
#include <string>
#include <vector>

#include "input.h"


namespace interface
{

    struct WindowConfig
    {
        std::string title;
        int width;
        int height;
        bool resizable = true;
    };

    class Window
    {
    public:
        virtual ~Window() = default;

        virtual bool open(const WindowConfig& config) = 0;

        virtual void close() = 0;

        virtual bool tick(InputEvent& e) = 0;

        virtual bool should_close() const = 0;

        // Vulkan integration

        virtual std::vector<const char*> get_required_instance_extensions() const = 0;

        virtual bool create_vulkan_surface(VkInstance instance, VkSurfaceKHR* surface) const = 0;

        // Window properties

        virtual void get_extent(int& width, int& height) const = 0;

        virtual float get_aspect_ratio() const = 0;
    };

} // namespace interface