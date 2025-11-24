#pragma once

#include <string>
#include <vector>
#include <functional>
#include <vulkan/vulkan.h>
#include "input.h"

namespace platform {

struct WindowConfig {
    std::string title;
    int width;
    int height;
    bool resizable = true;
};

class IWindow {
public:
    virtual ~IWindow() = default;

    virtual bool Initialize(const WindowConfig& config) = 0;
    virtual void Shutdown() = 0;
    virtual void PollEvents() = 0;
    virtual bool ShouldClose() const = 0;

    // Vulkan integration
    virtual std::vector<const char*> GetRequiredInstanceExtensions() const = 0;
    virtual bool CreateSurface(VkInstance instance, VkSurfaceKHR* surface) const = 0;

    // Window properties
    virtual void GetExtent(int& width, int& height) const = 0;
    virtual float GetAspectRatio() const = 0;

    // Input callbacks
    using EventCallback = std::function<void(const InputEvent&)>;
    virtual void SetEventCallback(EventCallback callback) = 0;
};

} // namespace platform
