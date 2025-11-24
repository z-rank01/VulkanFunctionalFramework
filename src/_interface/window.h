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

    virtual bool Initialize(const WindowConfig& config) = 0;
    virtual void Shutdown()                             = 0;
    virtual void PollEvents()                           = 0;
    virtual bool ShouldClose() const                    = 0;

    // Vulkan integration
    virtual std::vector<const char*> GetRequiredInstanceExtensions() const       = 0;
    virtual bool CreateSurface(VkInstance instance, VkSurfaceKHR* surface) const = 0;

    // Window properties
    virtual void GetExtent(int& width, int& height) const = 0;
    virtual float GetAspectRatio() const                  = 0;

    // Input callbacks
    using EventCallback                                   = std::function<void(const InputEvent&)>;
    virtual void SetEventCallback(EventCallback callback) = 0;
};

} // namespace interface
