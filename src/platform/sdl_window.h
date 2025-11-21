#pragma once

#include "window.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace platform {

class SDLWindow : public IWindow {
public:
    SDLWindow();
    ~SDLWindow() override;

    bool Initialize(const WindowConfig& config) override;
    void Shutdown() override;
    void PollEvents() override;
    bool ShouldClose() const override;

    // Vulkan integration implementation
    std::vector<const char*> GetRequiredInstanceExtensions() const override;
    bool CreateSurface(VkInstance instance, VkSurfaceKHR* surface) const override;

    // Window properties implementation
    void GetExtent(int& width, int& height) const override;
    float GetAspectRatio() const override;

    // Input callbacks implementation
    void SetEventCallback(EventCallback callback) override;

private:
    SDL_Window* window_ = nullptr;
    bool should_close_ = false;
    EventCallback event_callback_;

    KeyCode translate_key_code(SDL_Keycode key);
    MouseButton translate_mouse_button(uint8_t button);
};

} // namespace platform
