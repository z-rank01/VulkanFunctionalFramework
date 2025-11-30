#pragma once

#include <VkBootstrap.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <unordered_set>

#include "_gltf/gltf_data.h"
#include "_interface/sdl_window.h" // For default implementation
#include "_interface/simple_camera.h"
#include "_interface/window.h"
#include "_old/vulkan_commandbuffer.h"
#include "_old/vulkan_framebuffer.h"
#include "_old/vulkan_pipeline.h"
#include "_old/vulkan_renderpass.h"
#include "_old/vulkan_shader.h"
#include "_old/vulkan_synchronization.h"
#include "_templates/common.hpp"
#include "_vra/vra.h"
#include "utility/config_reader.h"

enum class EWindowState : std::uint8_t
{
    kInitialized, // Engine is initialized but not running
    kRunning,     // Engine is running
    kStopped      // Engine is stopped
};

enum class ERenderState : std::uint8_t
{
    kTrue, // Render is enabled
    kFalse // Render is disabled
};

struct window_config
{
    int width;
    int height;
    std::string title;

    [[nodiscard]] constexpr auto Validate() const -> bool { return width > 0 && height > 0; }
};

struct SEngineConfig
{
    window_config window_config;
    SGeneralConfig general_config;
    uint8_t frame_count;
    bool use_validation_layers;
};

struct SOutputFrame
{
    uint32_t image_index;
    std::string queue_id;
    std::string command_buffer_id;
    std::string image_available_semaphore_id;
    std::string render_finished_semaphore_id;
    std::string fence_id;
};

struct SMvpMatrix
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

struct SCamera
{
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up;
    float yaw;
    float pitch;
    float movement_speed;
    float wheel_speed;
    float mouse_sensitivity;
    float zoom;

    // 聚焦点相关
    glm::vec3 focus_point;
    bool has_focus_point;
    float focus_distance;
    float min_focus_distance;
    float max_focus_distance;

    // Add focus constraint enabled flag
    bool focus_constraint_enabled_;

    SCamera(glm::vec3 pos       = glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3 up        = glm::vec3(0.0f, 1.0f, 0.0f),
            float initial_yaw   = -90.0f,
            float initial_pitch = 0.0f)
        : position(pos), world_up(up), yaw(initial_yaw), pitch(initial_pitch), movement_speed(2.5f), wheel_speed(0.01f),
          mouse_sensitivity(0.1f), zoom(45.0f),
          // Initialize focus constraint enabled flag
          focus_constraint_enabled_(true) // Default to enabled
    {
        UpdateCameraVectors();
    }

    void UpdateCameraVectors()
    {
        // if (pitch > 89.0f) pitch = 89.0f;
        // if (pitch < -89.0f) pitch = -89.0f;

        // 在Vulkan坐标系中计算相机方向：+X向右，+Y向上，+Z向屏幕外
        glm::vec3 new_front;
        new_front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        new_front.y = sin(glm::radians(pitch)); // Y轴向上
        new_front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front       = glm::normalize(new_front);

        // 计算右向量和上向量
        right = glm::normalize(glm::cross(front, world_up));
        up    = glm::normalize(glm::cross(right, front));
    }
};

class VulkanSample
{
public:
    VulkanSample() = delete;
    VulkanSample(SEngineConfig config);
    ~VulkanSample();

    void Run();
    void Draw();

    static VulkanSample& GetInstance();
    void Initialize();
    void GetVertexIndexData(std::vector<gltf::PerDrawCallData> per_draw_call_data,
                            std::vector<uint32_t> indices,
                            std::vector<gltf::Vertex> vertices);
    void GetMeshList(const std::vector<gltf::PerMeshData>& mesh_list);

private:
#define FRAME_INDEX_TO_UNIFORM_BUFFER_ID(frame_index) (frame_index + 4)
    // engine members
    uint8_t frame_index_ = 0;
    bool resize_request_ = false;
    EWindowState engine_state_;
    ERenderState render_state_;
    SEngineConfig engine_config_;
    SCamera camera_;
    std::vector<SOutputFrame> output_frames_;

    // mesh data members
    std::vector<gltf::PerMeshData> mesh_list_;
    std::unordered_map<std::string, std::vector<vra::ResourceId>> mesh_vertex_resource_ids_;
    std::unordered_map<std::string, std::vector<vra::ResourceId>> mesh_index_resource_ids_;
    std::unordered_map<std::string, std::vector<VkDeviceSize>> mesh_vertex_offsets_;
    std::unordered_map<std::string, std::vector<VkDeviceSize>> mesh_index_offsets_;

    // vra and vma members
    VmaAllocator vma_allocator_;
    VmaAllocation local_buffer_allocation_;
    VmaAllocation staging_buffer_allocation_;
    VmaAllocation uniform_buffer_allocation_;
    VmaAllocationInfo local_buffer_allocation_info_;
    VmaAllocationInfo staging_buffer_allocation_info_;
    VmaAllocationInfo uniform_buffer_allocation_info_;

    std::unique_ptr<vra::VraDataBatcher> vra_data_batcher_;
    std::map<vra::BatchId, vra::VraDataBatcher::VraBatchHandle> vertex_index_staging_batch_handle_;
    std::map<vra::BatchId, vra::VraDataBatcher::VraBatchHandle> uniform_batch_handle_;
    vra::ResourceId vertex_data_id_;
    vra::ResourceId index_data_id_;
    vra::ResourceId staging_vertex_data_id_;
    vra::ResourceId staging_index_data_id_;
    std::vector<vra::ResourceId> uniform_buffer_id_;

    // vulkan native members
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    vk::Buffer local_buffer_;
    vk::Buffer staging_buffer_;
    vk::Buffer uniform_buffer_;
    vk::DescriptorPool descriptor_pool_;
    vk::DescriptorSetLayout descriptor_set_layout_;
    std::vector<vk::DescriptorSet> descriptor_sets_;
    vk::VertexInputBindingDescription vertex_input_binding_description_;
    vk::VertexInputAttributeDescription vertex_input_attribute_position_;
    vk::VertexInputAttributeDescription vertex_input_attribute_color_;

    // vulkan helper members
    std::unique_ptr<interface::Window> window_;
    std::unique_ptr<VulkanShaderHelper> vk_shader_helper_;
    std::unique_ptr<VulkanRenderpassHelper> vk_renderpass_helper_;
    std::unique_ptr<VulkanPipelineHelper> vk_pipeline_helper_;
    std::unique_ptr<VulkanCommandBufferHelper> vk_command_buffer_helper_;
    std::unique_ptr<VulkanFrameBufferHelper> vk_frame_buffer_helper_;
    std::unique_ptr<VulkanSynchronizationHelper> vk_synchronization_helper_;

    // uniform data
    std::vector<SMvpMatrix> mvp_matrices_;
    void* uniform_buffer_mapped_data_;

    // Input handling members
    std::unordered_set<interface::key_code> pressed_keys_;
    float last_x_ = 0.0F;
    float last_y_ = 0.0F;
    // Rename camera_rotation_mode_ to free_look_mode_
    bool free_look_mode_  = false; // 相机自由查看模式标志（右键按住）
    bool camera_pan_mode_ = false; // 相机平移模式标志（中键）
    float orbit_distance_ = 0.0F;  // 轨道旋转时与中心的距离

    void initialize_vulkan_hpp();
    void initialize_window();
    void initialize_vulkan();
    void initialize_camera();

    // --- Vulkan Initialization Steps ---
    void generate_frame_structs();
    bool create_instance();
    bool create_surface();
    bool create_physical_device();
    bool create_logical_device();
    bool create_swapchain();
    bool create_depth_resources();
    bool create_frame_buffer();
    bool create_pipeline();
    bool create_command_pool();
    bool create_and_write_descriptor_relatives();
    bool create_vma_vra_objects();
    void create_drawcall_list_buffer();
    bool create_uniform_buffers();

    bool allocate_per_frame_command_buffer();
    bool create_synchronization_objects();
    // ------------------------------------

    // --- Vulkan Draw Steps ---
    void draw_frame();
    void resize_swapchain();
    bool record_command(uint32_t image_index, const std::string& command_buffer_id);
    void update_uniform_buffer(uint32_t current_frame_index);
    // -------------------------

    // --- camera control ---
    std::unique_ptr<interface::camera> simple_camera_;
    
    // --- Common Templates Test ---
    vk::Instance comm_vk_instance_;
    vk::PhysicalDevice comm_vk_physical_device_;
    vk::Device comm_vk_logical_device_;
    vk::Queue comm_vk_graphics_queue_;
    vk::Queue comm_vk_transfer_queue_;
    vk::SwapchainKHR comm_vk_swapchain_;
    templates::common::CommVkInstanceContext comm_vk_instance_context_;
    templates::common::CommVkPhysicalDeviceContext comm_vk_physical_device_context_;
    templates::common::CommVkLogicalDeviceContext comm_vk_logical_device_context_;
    templates::common::CommVkSwapchainContext comm_vk_swapchain_context_;

    // --- test function and data ---
    std::vector<gltf::PerDrawCallData> per_draw_call_data_list_;
    std::vector<uint32_t> indices_;
    std::vector<gltf::Vertex> vertices_;

    vk::Buffer test_local_buffer_;
    vk::Buffer test_staging_buffer_;
    vk::VertexInputBindingDescription test_vertex_input_binding_description_;
    std::vector<vk::VertexInputAttributeDescription> test_vertex_input_attributes_;
    VmaAllocation test_local_buffer_allocation_;
    VmaAllocation test_staging_buffer_allocation_;
    VmaAllocationInfo test_local_buffer_allocation_info_;
    VmaAllocationInfo test_staging_buffer_allocation_info_;

    vra::ResourceId test_vertex_buffer_id_;
    vra::ResourceId test_index_buffer_id_;
    vra::ResourceId test_staging_vertex_buffer_id_;
    vra::ResourceId test_staging_index_buffer_id_;

    std::map<vra::BatchId, vra::VraDataBatcher::VraBatchHandle> test_local_host_batch_handle_;
    std::map<vra::BatchId, vra::VraDataBatcher::VraBatchHandle> test_uniform_batch_handle_;

    // 深度资源相关成员
    vk::Image depth_image_          = VK_NULL_HANDLE;
    vk::DeviceMemory depth_memory_  = VK_NULL_HANDLE;
    vk::ImageView depth_image_view_ = VK_NULL_HANDLE;
    vk::Format depth_format_        = vk::Format::eD32Sfloat;

    // 创建深度资源
    vk::Format find_supported_depth_format();
};
