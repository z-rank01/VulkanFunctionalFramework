#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "window.h"

namespace interface
{

    class SDLWindow : public Window
    {
    public:
        SDLWindow();

        ~SDLWindow() override;

        bool open(const WindowConfig& config) override;

        void close() override;

        bool SDLWindow::tick(InputEvent& e) override;

        bool should_close() const override;

        // Vulkan integration implementation

        std::vector<const char*> get_required_instance_extensions() const override;

        bool create_vulkan_surface(VkInstance instance, VkSurfaceKHR* surface) const override;

        // Window properties implementation

        void get_extent(int& width, int& height) const override;

        float get_aspect_ratio() const override;

    private:
        SDL_Window* window_ = nullptr;
        bool should_close_  = false;

        KeyCode translate_key_code(SDL_Keycode key);

        MouseButton translate_mouse_button(uint8_t button);
    };

} // namespace interface