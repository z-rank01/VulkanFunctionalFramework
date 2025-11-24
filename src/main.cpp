#include <vma/vk_mem_alloc.h>

#include <algorithm>
#include <iostream>

#include "_gltf/gltf_loader.h"
#include "_gltf/gltf_parser.h"
#include "utility/config_reader.h"
#include "utility/logger.h"
#include "vulkan_sample.h"


int main()
{
    std::cout << "Hello, World!" << '\n';
    std::cout << "This is a Vulkan Sample" << '\n';

    // read gltf file

    auto loader = gltf::GltfLoader();
    auto asset  = loader(R"(E:\Assets\Sponza\SponzaBase\NewSponza_Main_glTF_003.gltf)");
    // auto asset = loader("E:\\Assets\\Sponza\\SponzaCurtains\\NewSponza_Curtains_glTF.gltf");

    // parse gltf file

    gltf::GltfParser parser;
    auto mesh_list = parser(asset, gltf::RequestMeshList{});
    auto draw_call_data_list = parser(asset, gltf::RequestDrawCallList{});
    // Transform vertex positions using the draw call's transform matrix (functional expression)
    std::ranges::for_each(draw_call_data_list,
                          [](gltf::PerDrawCallData& primitive)
                          {
                              const glm::mat4& transform = primitive.transform;
                              std::ranges::for_each(primitive.vertices,
                                                    [&](gltf::Vertex& vertex)
                                                    {
                                                        glm::vec4 transformed_position =
                                                            transform * glm::vec4(vertex.position, 1.0F);
                                                        vertex.position = glm::vec3(transformed_position);
                                                    });
                          });

    // collect all indices

    std::vector<uint32_t> indices;
    std::vector<gltf::Vertex> vertices;

    // reserve memory

    auto index_capacity  = std::accumulate(draw_call_data_list.begin(),
                                          draw_call_data_list.end(),
                                          0,
                                          [&](const auto& sum, const auto& draw_call_data)
                                          { return sum + draw_call_data.indices.size(); });
    auto vertex_capacity = std::accumulate(draw_call_data_list.begin(),
                                           draw_call_data_list.end(),
                                           0,
                                           [&](const auto& sum, const auto& draw_call_data)
                                           { return sum + draw_call_data.vertices.size(); });
    indices.reserve(index_capacity);
    vertices.reserve(vertex_capacity);

    // collect all indices and vertices

    for (const auto& draw_call_data : draw_call_data_list)
    {
        indices.insert(indices.end(), draw_call_data.indices.begin(), draw_call_data.indices.end());
        vertices.insert(vertices.end(), draw_call_data.vertices.begin(), draw_call_data.vertices.end());
    }

    // window config

    SWindowConfig window_config;
    const auto kWindowWidth  = 1280;
    const auto kWindowHeight = 720;
    const auto* window_name  = "Vulkan Engine";

    window_config.width  = kWindowWidth;
    window_config.height = kWindowHeight;
    window_config.title  = window_name;

    // general config

    ConfigReader config_reader(R"(D:\Repository\VulkanFunctionalFramework\config\win64\app_config.json)");
    SGeneralConfig general_config;
    if (!config_reader.TryParseGeneralConfig(general_config))
    {
        Logger::LogError("Failed to parse general config");
        return -1;
    }

    // engine config

    SEngineConfig config;
    config.window_config         = window_config;
    config.general_config        = general_config;
    config.frame_count           = 3;
    config.use_validation_layers = true;

    // main loop

    VulkanSample sample(config);
    sample.GetVertexIndexData(draw_call_data_list, indices, vertices);
    sample.GetMeshList(mesh_list);
    sample.Initialize();
    sample.Run();

    std::cout << "Goodbye" << '\n';

    return 0;
}
