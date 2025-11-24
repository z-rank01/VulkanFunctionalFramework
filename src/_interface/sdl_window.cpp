#include "sdl_window.h"

#include <iostream>


namespace interface
{

SDLWindow::SDLWindow() { }

SDLWindow::~SDLWindow()
{
    Shutdown();
}

bool SDLWindow::Initialize(const WindowConfig& config)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    window_ = SDL_CreateWindow(config.title.c_str(), config.width, config.height, window_flags);

    if (!window_)
    {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    return true;
}

void SDLWindow::Shutdown()
{
    if (window_)
    {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

void SDLWindow::PollEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_EVENT_QUIT)
        {
            should_close_ = true;
            if (event_callback_)
            {
                InputEvent e;
                e.type = EventType::Quit;
                event_callback_(e);
            }
        }
        else if (event_callback_)
        {
            InputEvent e;
            bool handled = false;

            switch (event.type)
            {
            case SDL_EVENT_WINDOW_RESIZED:
                e.type          = EventType::Resize;
                e.resize.width  = event.window.data1;
                e.resize.height = event.window.data2;
                handled         = true;
                break;
            case SDL_EVENT_KEY_DOWN:
                e.type    = EventType::KeyDown;
                e.key.key = translate_key_code(event.key.key);
                handled   = true;
                break;
            case SDL_EVENT_KEY_UP:
                e.type    = EventType::KeyUp;
                e.key.key = translate_key_code(event.key.key);
                handled   = true;
                break;
            case SDL_EVENT_MOUSE_MOTION:
                e.type            = EventType::MouseMove;
                e.mouse_move.x    = event.motion.x;
                e.mouse_move.y    = event.motion.y;
                e.mouse_move.xrel = event.motion.xrel;
                e.mouse_move.yrel = event.motion.yrel;
                handled           = true;
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                e.type          = EventType::MouseWheel;
                e.mouse_wheel.x = event.wheel.x;
                e.mouse_wheel.y = event.wheel.y;
                handled         = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                e.type                 = EventType::MouseButtonDown;
                e.mouse_button.button  = translate_mouse_button(event.button.button);
                e.mouse_button.x       = event.button.x;
                e.mouse_button.y       = event.button.y;
                e.mouse_button.pressed = true;
                handled                = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                e.type                 = EventType::MouseButtonUp;
                e.mouse_button.button  = translate_mouse_button(event.button.button);
                e.mouse_button.x       = event.button.x;
                e.mouse_button.y       = event.button.y;
                e.mouse_button.pressed = false;
                handled                = true;
                break;
            }

            if (handled)
            {
                event_callback_(e);
            }
        }
    }
}

bool SDLWindow::ShouldClose() const
{
    return should_close_;
}

std::vector<const char*> SDLWindow::GetRequiredInstanceExtensions() const
{
    uint32_t count           = 0;
    const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
    std::vector<const char*> extensions(names, names + count);
    return extensions;
}

bool SDLWindow::CreateSurface(VkInstance instance, VkSurfaceKHR* surface) const
{
    return SDL_Vulkan_CreateSurface(window_, instance, nullptr, surface);
}

void SDLWindow::GetExtent(int& width, int& height) const
{
    SDL_GetWindowSize(window_, &width, &height);
}

float SDLWindow::GetAspectRatio() const
{
    int width, height;
    GetExtent(width, height);
    return (float)width / (float)height;
}

void SDLWindow::SetEventCallback(EventCallback callback)
{
    event_callback_ = callback;
}

KeyCode SDLWindow::translate_key_code(SDL_Keycode key)
{
    switch (key)
    {
    case SDLK_W:
        return KeyCode::W;
    case SDLK_A:
        return KeyCode::A;
    case SDLK_S:
        return KeyCode::S;
    case SDLK_D:
        return KeyCode::D;
    case SDLK_Q:
        return KeyCode::Q;
    case SDLK_E:
        return KeyCode::E;
    case SDLK_F:
        return KeyCode::F;
    case SDLK_SPACE:
        return KeyCode::Space;
    case SDLK_LSHIFT:
        return KeyCode::LShift;
    case SDLK_LCTRL:
        return KeyCode::LCtrl;
    case SDLK_ESCAPE:
        return KeyCode::Escape;
    case SDLK_UP:
        return KeyCode::Up;
    case SDLK_DOWN:
        return KeyCode::Down;
    case SDLK_LEFT:
        return KeyCode::Left;
    case SDLK_RIGHT:
        return KeyCode::Right;
    default:
        return KeyCode::Unknown;
    }
}

MouseButton SDLWindow::translate_mouse_button(uint8_t button)
{
    switch (button)
    {
    case SDL_BUTTON_LEFT:
        return MouseButton::Left;
    case SDL_BUTTON_MIDDLE:
        return MouseButton::Middle;
    case SDL_BUTTON_RIGHT:
        return MouseButton::Right;
    default:
        return MouseButton::Unknown;
    }
}

} // namespace interface
