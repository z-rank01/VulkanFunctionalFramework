#include "app_sample.h"
#include "_interface/sdl_window.h" // For default implementation
#include "_interface/simple_camera.h"
#include <stdexcept>

void AppSample::initialize()
{
    window_ = std::make_unique<interface::SDLWindow>();
    interface::WindowConfig config;
    config.title = window_config_.title_;
    config.width = window_config_.width_;
    config.height = window_config_.height_;

    if (!window_->Initialize(config))
    {
        throw std::runtime_error("Failed to create window.");
    }

    window_->SetEventCallback([this](const interface::InputEvent& event) { this->camera_->tick(event); });
}
