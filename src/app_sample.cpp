#include "app_sample.h"
#include "_interface/sdl_window.h" // For default implementation
#include "_interface/simple_camera.h"
#include <stdexcept>

void app_sample::initialize()
{
    // initialize sdl window
    window_ = std::make_unique<interface::SDLWindow>();
    interface::WindowConfig config;
    config.title = window_config_.title;
    config.width = window_config_.width;
    config.height = window_config_.height;
    if (!window_->open(config))
    {
        throw std::runtime_error("Failed to open window.");
    }
}

void app_sample::tick()
{

}
