#include "sdl_window.h"

#include <iostream>


namespace interface
{

    SDLWindow::SDLWindow() = default;

    SDLWindow::~SDLWindow()
    {
        SDLWindow::close();
    }

    bool SDLWindow::open(const WindowConfig& config)
    {
        if (static_cast<int>(SDL_Init(SDL_INIT_VIDEO)) < 0)
        {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << '\n';
            return false;
        }

        auto window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        window = SDL_CreateWindow(config.title.c_str(), config.width, config.height, window_flags);

        if (window == nullptr)
        {
            std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << '\n';
            return false;
        }

        return true;
    }

    void SDLWindow::close()
    {
        if (window != nullptr)
        {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        SDL_Quit();
    }

    void SDLWindow::tick(input_event& e)
    {
        SDL_Event event;
        SDL_PollEvent(&event);
        switch (event.type)
        {
        case SDL_EVENT_WINDOW_RESIZED:
            e.type = event_type::Resize;
            e.resize.width  = event.window.data1;
            e.resize.height = event.window.data2;
            break;
        case SDL_EVENT_KEY_DOWN:
            e.type = event_type::KeyDown;
            e.key.key = translate_key_code(event.key.key);
            break;
        case SDL_EVENT_KEY_UP:
            e.type = event_type::KeyUp;
            e.key.key = translate_key_code(event.key.key);
            break;
        case SDL_EVENT_MOUSE_MOTION:
            e.type = event_type::MouseMove;
            e.mouse_move.x    = event.motion.x;
            e.mouse_move.y    = event.motion.y;
            e.mouse_move.xrel = event.motion.xrel;
            e.mouse_move.yrel = event.motion.yrel;
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            e.type = event_type::MouseWheel;
            e.mouse_wheel.x = event.wheel.x;
            e.mouse_wheel.y = event.wheel.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            e.type = event_type::MouseButtonDown;
            e.mouse_button.button  = translate_mouse_button(event.button.button);
            e.mouse_button.x       = event.button.x;
            e.mouse_button.y       = event.button.y;
            e.mouse_button.pressed = true;
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            e.type = event_type::MouseButtonUp;
            e.mouse_button.button  = translate_mouse_button(event.button.button);
            e.mouse_button.x       = event.button.x;
            e.mouse_button.y       = event.button.y;
            e.mouse_button.pressed = false;
            break;
        case SDL_EVENT_QUIT:
            e.type = event_type::Quit;
            should_close_internal = true;
            break;
        default:
            e = input_event{};
            break;
        }
    }

    bool SDLWindow::should_close() const
    {
        return should_close_internal;
    }

    std::vector<const char*> SDLWindow::get_required_instance_extensions() const
    {
        uint32_t count           = 0;
        const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
        std::vector<const char*> extensions(names, names + count);
        return extensions;
    }

    bool SDLWindow::create_vulkan_surface(VkInstance instance, VkSurfaceKHR* surface) const
    {
        return SDL_Vulkan_CreateSurface(window, instance, nullptr, surface);
    }

    void SDLWindow::get_extent(int& width, int& height) const
    {
        SDL_GetWindowSize(window, &width, &height);
    }

    float SDLWindow::get_aspect_ratio() const
    {
        int width = 0;
        int height = 0;
        get_extent(width, height);
        return static_cast<float>(width) / static_cast<float>(height);
    }

    key_code SDLWindow::translate_key_code(SDL_Keycode key)
    {
        switch (key)
        {
        case SDLK_W:
            return key_code::W;
        case SDLK_A:
            return key_code::A;
        case SDLK_S:
            return key_code::S;
        case SDLK_D:
            return key_code::D;
        case SDLK_Q:
            return key_code::Q;
        case SDLK_E:
            return key_code::E;
        case SDLK_F:
            return key_code::F;
        case SDLK_SPACE:
            return key_code::Space;
        case SDLK_LSHIFT:
            return key_code::LShift;
        case SDLK_LCTRL:
            return key_code::LCtrl;
        case SDLK_ESCAPE:
            return key_code::Escape;
        case SDLK_UP:
            return key_code::Up;
        case SDLK_DOWN:
            return key_code::Down;
        case SDLK_LEFT:
            return key_code::Left;
        case SDLK_RIGHT:
            return key_code::Right;
        default:
            return key_code::Unknown;
        }
    }

    mouse_button SDLWindow::translate_mouse_button(uint8_t button)
    {
        switch (button)
        {
        case SDL_BUTTON_LEFT:
            return mouse_button::Left;
        case SDL_BUTTON_MIDDLE:
            return mouse_button::Middle;
        case SDL_BUTTON_RIGHT:
            return mouse_button::Right;
        default:
            return mouse_button::Unknown;
        }
    }

} // namespace interface
