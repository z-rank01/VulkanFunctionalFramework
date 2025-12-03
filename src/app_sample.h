#pragma once
#include <memory>
#include <string>
#include <vector>

#include "_interface/camera.h"
#include "_interface/window.h"
#include "vulkan_sample.h"
#include "_interface/camera_component.h"
#include "_interface/camera_system.h"

class app_sample
{
public:
    app_sample(SEngineConfig config);
    // core public function
    void initialize();
    void tick();

    void get_vertex_index_data(std::vector<gltf::PerDrawCallData> per_draw_call_data, std::vector<uint32_t> indices, std::vector<gltf::Vertex> vertices);
    void get_mesh_list(const std::vector<gltf::PerMeshData>& mesh_list);

private:
    // core member
    std::unique_ptr<interface::Window> window;
    // std::unique_ptr<interface::camera> camera;
    std::unique_ptr<VulkanSample> vulkan_sample;

    // data-oriented camera
    size_t camera_entity_index = 0;
    dod_camera::camera_container camera_container;
    dod_camera::camera_update_context camera_update_context;

    // delta time tracking
    std::chrono::high_resolution_clock::time_point last_frame_time;
    float delta_time;
    

    SEngineConfig general_config;
};
