#include "app_sample.h"

#include <stdexcept>

#include "_interface/sdl_window.h" // For default implementation
#include "_interface/simple_camera.h"


app_sample::app_sample(SEngineConfig config) : general_config(std::move(config))
{
    vulkan_sample = std::make_unique<VulkanSample>(general_config);
}

void app_sample::initialize()
{
    // initialize sdl window
    window = std::make_unique<interface::SDLWindow>();
    interface::WindowConfig win_config;
    win_config.title  = general_config.window_config.title;
    win_config.width  = general_config.window_config.width;
    win_config.height = general_config.window_config.height;
    if (!window->open(win_config))
    {
        throw std::runtime_error("Failed to open window.");
    }

    // initialize camera
    camera = std::make_unique<interface::simple_camera>();

    // setup vulkan sample
    vulkan_sample->SetWindow(window.get());
    vulkan_sample->SetCamera(camera.get());
    vulkan_sample->Initialize();
}

void app_sample::tick()
{
    interface::input_event e{};
    while (!window->should_close())
    {
        window->tick(e);
        camera->tick(e);
        vulkan_sample->Tick();
    }
}

void app_sample::get_vertex_index_data(std::vector<gltf::PerDrawCallData> per_draw_call_data,
                                    std::vector<uint32_t> indices,
                                    std::vector<gltf::Vertex> vertices)
{
    vulkan_sample->GetVertexIndexData(std::move(per_draw_call_data), std::move(indices), std::move(vertices));
}

void app_sample::get_mesh_list(const std::vector<gltf::PerMeshData>& mesh_list)
{
    vulkan_sample->GetMeshList(mesh_list);
}
