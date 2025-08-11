#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include "vulkan_sample.h"

#include <cmath>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <thread>

#include "_callable/callable.h"
#include "_templates/common.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

using namespace templates;

static VulkanSample* instance = nullptr;

VulkanSample& VulkanSample::GetInstance()
{
    return *instance;
}

VulkanSample::VulkanSample(SEngineConfig config) : engine_config_(std::move(config))
{
    // only one engine initialization is allowed with the application.
    assert(instance == nullptr);
    instance = this;
}

void VulkanSample::Initialize()
{
    // initialize SDL, vulkan, and camera
    initialize_vulkan_hpp();
    initialize_sdl();
    initialize_camera();
    initialize_vulkan();
}

void VulkanSample::GetVertexIndexData(std::vector<gltf::PerDrawCallData> per_draw_call_data,
                                      std::vector<uint32_t> indices,
                                      std::vector<gltf::Vertex> vertices)
{
    per_draw_call_data_list_ = std::move(per_draw_call_data);
    indices_                 = std::move(indices);
    vertices_                = std::move(vertices);
}

void VulkanSample::GetMeshList(const std::vector<gltf::PerMeshData>& mesh_list)
{
    mesh_list_ = mesh_list;
}

VulkanSample::~VulkanSample()
{
    // 等待设备空闲，确保没有正在进行的操作
    vkDeviceWaitIdle(comm_vk_logical_device_);

    // 销毁深度资源
    if (depth_image_view_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(comm_vk_logical_device_, depth_image_view_, nullptr);
        depth_image_view_ = VK_NULL_HANDLE;
    }

    if (depth_image_ != VK_NULL_HANDLE)
    {
        vkDestroyImage(comm_vk_logical_device_, depth_image_, nullptr);
        depth_image_ = VK_NULL_HANDLE;
    }

    if (depth_memory_ != VK_NULL_HANDLE)
    {
        vkFreeMemory(comm_vk_logical_device_, depth_memory_, nullptr);
        depth_memory_ = VK_NULL_HANDLE;
    }

    // 销毁描述符相关资源
    if (descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(comm_vk_logical_device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }

    if (descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(comm_vk_logical_device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }

    

    // destroy vma relatives
    if (uniform_buffer_ != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(vma_allocator_, uniform_buffer_, uniform_buffer_allocation_);
        uniform_buffer_ = VK_NULL_HANDLE;
    }
    if (test_local_buffer_ != VK_NULL_HANDLE) 
    {
        vmaDestroyBuffer(vma_allocator_, test_local_buffer_, test_local_buffer_allocation_);
        test_local_buffer_ = VK_NULL_HANDLE;
    }
    if (test_staging_buffer_ != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(vma_allocator_, test_staging_buffer_, test_staging_buffer_allocation_);
        test_staging_buffer_ = VK_NULL_HANDLE;
    }
    if (vma_allocator_ != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(vma_allocator_);
        vma_allocator_ = VK_NULL_HANDLE;
    }

    // destroy swapchain related resources

    for (auto *image_view : comm_vk_swapchain_context_.swapchain_image_views_)
    {
        vkDestroyImageView(comm_vk_logical_device_, image_view, nullptr);
    }
    vkDestroySwapchainKHR(comm_vk_logical_device_, comm_vk_swapchain_, nullptr);

    // release unique pointer

    vk_shader_helper_.reset();
    vk_window_helper_.reset();
    vk_renderpass_helper_.reset();
    vk_pipeline_helper_.reset();
    vk_frame_buffer_helper_.reset();
    vk_command_buffer_helper_.reset();
    vk_synchronization_helper_.reset();

    // destroy comm test data
    
    vkDestroyDevice(comm_vk_logical_device_, nullptr);
    vkDestroyInstance(comm_vk_instance_, nullptr);

    // 重置单例指针
    instance = nullptr;
}


void VulkanSample::initialize_vulkan_hpp()
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
}

// Initialize the engine
void VulkanSample::initialize_sdl()
{
    vk_window_helper_ = std::make_unique<VulkanSDLWindowHelper>();
    if (!vk_window_helper_->GetWindowBuilder()
             .SetWindowName(engine_config_.window_config.title)
             .SetWindowSize(engine_config_.window_config.width, engine_config_.window_config.height)
             .SetWindowFlags(SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN)
             .SetInitFlags(SDL_INIT_VIDEO | SDL_INIT_EVENTS)
             .Build())
    {
        throw std::runtime_error("Failed to create SDL window.");
    }

    // SDL Creation
}

void VulkanSample::initialize_vulkan()
{
    generate_frame_structs();

    if (!create_instance())
    {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }

    if (!create_surface())
    {
        throw std::runtime_error("Failed to create Vulkan surface.");
    }

    if (!create_physical_device())
    {
        throw std::runtime_error("Failed to create Vulkan physical device.");
    }

    if (!create_logical_device())
    {
        throw std::runtime_error("Failed to create Vulkan logical device.");
    }

    if (!create_swapchain())
    {
        throw std::runtime_error("Failed to create Vulkan swap chain.");
    }

    if (!create_vma_vra_objects())
    {
        throw std::runtime_error("Failed to create Vulkan vra and vma objects.");
    }

    create_drawcall_list_buffer();

    if (!create_uniform_buffers())
    {
        throw std::runtime_error("Failed to create Vulkan uniform buffers.");
    }

    if (!create_and_write_descriptor_relatives())
    {
        throw std::runtime_error("Failed to create Vulkan descriptor relatives.");
    }

    if (!create_pipeline())
    {
        throw std::runtime_error("Failed to create Vulkan pipeline.");
    }

    if (!create_frame_buffer())
    {
        throw std::runtime_error("Failed to create Vulkan frame buffer.");
    }

    if (!create_command_pool())
    {
        throw std::runtime_error("Failed to create Vulkan command pool.");
    }

    if (!allocate_per_frame_command_buffer())
    {
        throw std::runtime_error("Failed to allocate Vulkan command buffer.");
    }

    if (!create_synchronization_objects())
    {
        throw std::runtime_error("Failed to create Vulkan synchronization objects.");
    }
}

void VulkanSample::initialize_camera()
{
    // initialize mvp matrices
    mvp_matrices_ =
        std::vector<SMvpMatrix>(engine_config_.frame_count, {.model=glm::mat4(1.0F), .view=glm::mat4(1.0F), .projection=glm::mat4(1.0F)});

    // initialize camera
    camera_.position          = glm::vec3(0.0F, 0.0F, 10.0F); // 更远的初始距离
    camera_.yaw               = -90.0F;                       // look at origin
    camera_.pitch             = 0.0F;                         // horizontal view
    camera_.wheel_speed       = 0.1F;                         // 降低滚轮速度，避免变化太剧烈
    camera_.movement_speed    = 5.0F;                         // 调整移动速度
    camera_.mouse_sensitivity = 0.2F;                         // 降低鼠标灵敏度
    camera_.zoom              = 45.0F;
    camera_.world_up          = glm::vec3(0.0F, 1.0F, 0.0F); // Y-axis is up in Vulkan

    // initialize camera vectors
    camera_.front = glm::vec3(0.0F, 0.0F, -1.0F); // look at -z direction
    camera_.right = glm::vec3(1.0F, 0.0F, 0.0F);  // right direction is +x
    camera_.up    = glm::vec3(0.0F, 1.0F, 0.0F);  // up direction is +y (because Y-axis is up in Vulkan)

    // initialize focus point related parameters
    camera_.focus_point        = glm::vec3(0.0F); // default focus on origin
    camera_.has_focus_point    = true;            // default enable focus point
    camera_.focus_distance     = 10.0F;           // 增加默认焦距
    camera_.min_focus_distance = 0.5F;            // minimum focus distance
    camera_.max_focus_distance = 10000.0F;        // maximum focus distance
}

// Main loop
void VulkanSample::Run()
{
    engine_state_ = EWindowState::kRunning;

    SDL_Event event;

    Uint64 last_time = SDL_GetTicks();
    float delta_time = 0.0F;

    // main loop
    while (engine_state_ != EWindowState::kStopped)
    {
        // calculate the time difference between frames
        Uint64 current_time = SDL_GetTicks();
        delta_time          = (current_time - last_time) / 1000.0F; // convert to seconds
        last_time           = current_time;

        // handle events on queue
        while (SDL_PollEvent(&event))
        {
            process_input(event);

            // close the window when user alt-f4s or clicks the X button
            if (event.type == SDL_EVENT_QUIT)
            {
                engine_state_ = EWindowState::kStopped;
            }

            if (event.window.type == SDL_EVENT_WINDOW_SHOWN)
            {
                if (event.window.type == SDL_EVENT_WINDOW_MINIMIZED)
                {
                    render_state_ = ERenderState::kFalse;
                }
                if (event.window.type == SDL_EVENT_WINDOW_RESTORED)
                {
                    render_state_ = ERenderState::kTrue;
                }
            }
        }

        // process keyboard input to update camera
        process_keyboard_input(delta_time);

        // do not draw if we are minimized
        if (render_state_ == ERenderState::kFalse)
        {
            // throttle the speed to avoid the endless spinning
            constexpr auto kSleepDurationMs = 100;
            std::this_thread::sleep_for(std::chrono::milliseconds(kSleepDurationMs));
            continue;
        }

        if (resize_request_)
        {
            resize_swapchain();
        }

        // update the view matrix
        update_uniform_buffer(frame_index_);

        // render a frame
        Draw();
    }

    // wait until the GPU is completely idle before cleaning up
    vkDeviceWaitIdle(comm_vk_logical_device_);
}

void VulkanSample::process_input(SDL_Event& event)
{
    // key down event
    if (event.type == SDL_EVENT_KEY_DOWN)
    {
        // ESC key to exit
        if (event.key.key == SDLK_ESCAPE)
        {
            engine_state_ = EWindowState::kStopped;
        }
        // Toggle focus constraint with 'F' key
        if (event.key.key == SDLK_F)
        {
            camera_.focus_constraint_enabled_ = !camera_.focus_constraint_enabled_;
            Logger::LogInfo(camera_.focus_constraint_enabled_ ? "Focus constraint enabled"
                                                              : "Focus constraint disabled");
        }
    }

    // mouse button down event
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
    {
        float mouse_x = NAN;
        float mouse_y = NAN;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        last_x_ = mouse_x;
        last_y_ = mouse_y;

        if (event.button.button == SDL_BUTTON_RIGHT)
        {
            // Use free_look_mode_ instead of camera_rotation_mode_
            free_look_mode_ = true;
            // save the current mouse position
            last_x_ = mouse_x;
            last_y_ = mouse_y;
        }
        else if (event.button.button == SDL_BUTTON_MIDDLE)
        {
            camera_pan_mode_ = true;
        }
    }

    // mouse button up event
    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP)
    {
        if (event.button.button == SDL_BUTTON_RIGHT)
        {
            // Use free_look_mode_ instead of camera_rotation_mode_
            free_look_mode_ = false;
        }
        else if (event.button.button == SDL_BUTTON_MIDDLE)
        {
            camera_pan_mode_ = false;
        }
    }

    // mouse motion event
    if (event.type == SDL_EVENT_MOUSE_MOTION)
    {
        // Only process mouse motion for camera control if a mode is active
        if (!free_look_mode_ && !camera_pan_mode_)
        {
            return; // Do nothing if no camera control mode is active
        }

        float x_pos    = event.motion.x;
        float y_pos    = event.motion.y;
        float x_offset = x_pos - last_x_;
        float y_offset = last_y_ - y_pos; // Invert y-offset as screen y increases downwards
        last_x_        = x_pos;
        last_y_        = y_pos;

        if (free_look_mode_)
        {
            // Calculate mouse sensitivity scale based on distance to focus point if
            // constraint is enabled
            float sensitivity_scale = 1.0F;
            if (camera_.has_focus_point && camera_.focus_constraint_enabled_)
            {
                float current_distance = glm::length(camera_.position - camera_.focus_point);
                float distance_factor  = glm::clamp(current_distance / camera_.focus_distance,
                                                   camera_.min_focus_distance / camera_.focus_distance,
                                                   camera_.max_focus_distance / camera_.focus_distance);
                sensitivity_scale      = distance_factor; // Slower when closer to focus point
            }

            // Apply sensitivity and scale
            float actual_x_offset = x_offset * camera_.mouse_sensitivity * sensitivity_scale;
            float actual_y_offset = y_offset * camera_.mouse_sensitivity * sensitivity_scale;

            // Update the yaw and pitch (Unity-style tracking mode)
            camera_.yaw += actual_x_offset;   // Changed from -= to +=
            camera_.pitch += actual_y_offset; // Changed from -= to +=

            // limit the pitch angle
            camera_.pitch = std::min(camera_.pitch, 89.0F);
            camera_.pitch = std::max(camera_.pitch, -89.0F);

            // calculate the new camera direction and update vectors
            camera_.UpdateCameraVectors();
        }

        if (camera_pan_mode_)
        {
            // calculate the current distance to the focus point
            float current_distance =
                camera_.has_focus_point ? glm::length(camera_.position - camera_.focus_point) : camera_.focus_distance;

            // calculate the distance scale based on the distance
            float distance_scale = glm::clamp(current_distance / camera_.focus_distance,
                                              camera_.min_focus_distance / camera_.focus_distance,
                                              camera_.max_focus_distance / camera_.focus_distance);

            float pan_speed_multiplier = 0.005F;
            // Apply distance scaling to pan speed if focus constraint is enabled
            float actual_pan_speed_multiplier =
                camera_.focus_constraint_enabled_ ? pan_speed_multiplier / distance_scale : pan_speed_multiplier;

            // calculate the target movement amount
            float target_x_offset = x_offset * camera_.movement_speed * actual_pan_speed_multiplier;
            float target_y_offset = y_offset * camera_.movement_speed * actual_pan_speed_multiplier;

            // Apply movement using camera's right and up vectors
            camera_.position -= camera_.right * target_x_offset; // Pan left/right
            camera_.position += camera_.up * target_y_offset;    // Pan up/down
        }
    }

    // mouse wheel event
    if (event.type == SDL_EVENT_MOUSE_WHEEL)
    {
        float zoom_factor = camera_.wheel_speed;
        float distance    = glm::length(camera_.position);

        if (event.wheel.y > 0)
        {
            if (distance > 0.5F)
            {
                camera_.position *= (1.0F - zoom_factor);
            }
        }
        else if (event.wheel.y < 0)
        {
            camera_.position *= (1.0F + zoom_factor);
        }

        process_mouse_scroll(event.wheel.y);
    }
}

void VulkanSample::process_keyboard_input(float delta_time)
{
    // get the keyboard state
    const bool* keyboard_state = SDL_GetKeyboardState(nullptr);

    float velocity = camera_.movement_speed * delta_time;

    // If free look mode is enabled, move in world space based on camera
    // orientation
    if (free_look_mode_)
    {
        // Calculate movement speed scale based on distance to focus point if
        // constraint is enabled
        float distance_scale = 1.0F;
        if (camera_.has_focus_point && camera_.focus_constraint_enabled_)
        {
            float current_distance = glm::length(camera_.position - camera_.focus_point);
            distance_scale         = glm::clamp(current_distance / camera_.focus_distance,
                                        camera_.min_focus_distance / camera_.focus_distance,
                                        camera_.max_focus_distance / camera_.focus_distance);
        }
        float current_velocity = velocity / distance_scale; // Slower when closer

        // move in the screen space
        glm::vec3 movement(0.0F);

        // move front/back (Z-axis relative to camera)
        if (keyboard_state[SDL_SCANCODE_W] || keyboard_state[SDL_SCANCODE_UP])
        {
            movement += camera_.front * current_velocity;
        }
        if (keyboard_state[SDL_SCANCODE_S] || keyboard_state[SDL_SCANCODE_DOWN])
        {
            movement -= camera_.front * current_velocity;
        }

        // move left/right (X-axis relative to camera)
        if (keyboard_state[SDL_SCANCODE_A] || keyboard_state[SDL_SCANCODE_LEFT])
        {
            movement -= camera_.right * current_velocity;
        }
        if (keyboard_state[SDL_SCANCODE_D] || keyboard_state[SDL_SCANCODE_RIGHT])
        {
            movement += camera_.right * current_velocity;
        }

        // move up/down (Y-axis relative to world or camera up)
        if (keyboard_state[SDL_SCANCODE_Q])
        {
            movement += camera_.up * current_velocity; // Using camera's up vector for local up/down
        }
        if (keyboard_state[SDL_SCANCODE_E])
        {
            movement -= camera_.up * current_velocity; // Using camera's up vector for local up/down
        }

        // apply the movement
        camera_.position += movement;
    }
    else // Original screen space movement logic
    {
        // move in the screen space
        glm::vec3 movement(0.0F);

        // move up (Y-axis)
        if (keyboard_state[SDL_SCANCODE_W] || keyboard_state[SDL_SCANCODE_UP])
        {
            movement.y += velocity; // move up (Y-axis positive direction)
        }
        if (keyboard_state[SDL_SCANCODE_S] || keyboard_state[SDL_SCANCODE_DOWN])
        {
            movement.y -= velocity; // move down (Y-axis negative direction)
        }

        // move left (l-axis)t (X-axis)
        if (keyboard_state[SDL_SCANCODE_A] || keyboard_state[SDL_SCANCODE_LEFT])
        {
            movement.x -= velocity; // move left (l-axis negative direction)X-axis
                                    // negative direction)
        }
        if (keyboard_state[SDL_SCANCODE_D] || keyboard_state[SDL_SCANCODE_RIGHT])
        {
            movement.x += velocity; // move right (X-axis positive direction)
        }

        // move front (Z-axis)
        if (keyboard_state[SDL_SCANCODE_Q])
        {
            movement.z += velocity; // move back (Z-axis negative direction)
        }
        if (keyboard_state[SDL_SCANCODE_E])
        {
            movement.z -= velocity; // move front (Z-axis positive direction)
        }

        // apply the smooth movement (removed smooth factor for simplicity in free
        // look, could add back if needed) float smooth_factor = 0.1f; // smooth
        // factor
        camera_.position += movement; // * smooth_factor;
    }
}

void VulkanSample::process_mouse_scroll(float yoffset)
{
    if (camera_.has_focus_point && camera_.focus_constraint_enabled_)
    {
        // Zoom by moving along the camera's front vector
        float zoom_step = camera_.movement_speed * 0.5F; // Adjust zoom speed

        // Calculate distance scale
        float current_distance = glm::length(camera_.position - camera_.focus_point);
        float distance_scale   = glm::clamp(current_distance / camera_.focus_distance,
                                          camera_.min_focus_distance / camera_.focus_distance,
                                          camera_.max_focus_distance / camera_.focus_distance);
        zoom_step /= distance_scale; // Smaller steps when closer

        if (yoffset > 0)
        {
            // Zoom in (move towards focus point)
            camera_.position += camera_.front * zoom_step;
        }
        else if (yoffset < 0)
        {
            // Zoom out (move away from focus point)
            camera_.position -= camera_.front * zoom_step;
        }
        // Update camera vectors after changing position for orbit-like feel
        camera_.UpdateCameraVectors();
    }
    else
    {
        // Original FOV zoom logic when focus constraint is disabled
        camera_.zoom -= yoffset;
        camera_.zoom = std::max(camera_.zoom, 1.0F);
        camera_.zoom = std::min(camera_.zoom, 45.0F);
    }
}

// Main render loop
void VulkanSample::Draw()
{
    draw_frame();
}

// -------------------------------------
// private function to create the engine
// -------------------------------------

void VulkanSample::generate_frame_structs()
{
    output_frames_.resize(engine_config_.frame_count);
    for (int i = 0; i < engine_config_.frame_count; ++i)
    {
        output_frames_[i].image_index                  = i;
        output_frames_[i].queue_id                     = "graphic_queue";
        output_frames_[i].command_buffer_id            = "graphic_command_buffer_" + std::to_string(i);
        output_frames_[i].image_available_semaphore_id = "image_available_semaphore_" + std::to_string(i);
        output_frames_[i].render_finished_semaphore_id = "render_finished_semaphore_" + std::to_string(i);
        output_frames_[i].fence_id                     = "in_flight_fence_" + std::to_string(i);
    }
}

bool VulkanSample::create_instance()
{
    auto extensions     = vk_window_helper_->GetWindowExtensions();
    auto instance_chain = common::instance::create_context() | common::instance::set_application_name("My Vulkan App") |
                          common::instance::set_engine_name("My Engine") |
                          common::instance::set_application_version(1, 3, 0) |
                          common::instance::add_validation_layers({"VK_LAYER_KHRONOS_validation"}) |
                          common::instance::add_extensions(extensions) | common::instance::validate_context() |
                          common::instance::create_vk_instance();

    auto result = instance_chain.evaluate();

    if (!callable::is_ok(result))
    {
        std::string error_msg = std::get<std::string>(result);
        std::cerr << "Failed to create Vulkan instance: " << error_msg << '\n';
        return false;
    }
    auto context      = std::get<templates::common::CommVkInstanceContext>(result);
    comm_vk_instance_ = context.vk_instance_;
    std::cout << "Successfully created Vulkan instance." << '\n';
    return true;
}

bool VulkanSample::create_surface()
{
    return vk_window_helper_->CreateSurface(comm_vk_instance_);
}

bool VulkanSample::create_physical_device()
{
    // vulkan 1.3 features - 用于检查硬件支持
    VkPhysicalDeviceVulkan13Features features_13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features_13.synchronization2 = VK_TRUE;

    auto physical_device_chain = common::physicaldevice::create_physical_device_context(comm_vk_instance_) |
                                 common::physicaldevice::set_surface(vk_window_helper_->GetSurface()) |
                                 common::physicaldevice::require_api_version(1, 3, 0) |
                                 common::physicaldevice::require_features_13(features_13) |
                                 common::physicaldevice::require_queue(VK_QUEUE_GRAPHICS_BIT, 1, true) |
                                 common::physicaldevice::prefer_discrete_gpu() |
                                 common::physicaldevice::select_physical_device();

    auto result = physical_device_chain.evaluate();

    if (!callable::is_ok(result))
    {
        std::string error_msg = std::get<std::string>(result);
        std::cerr << "Failed to create Vulkan physical device: " << error_msg << '\n';
        return false;
    }

    comm_vk_physical_device_context_ = std::get<templates::common::CommVkPhysicalDeviceContext>(result);
    comm_vk_physical_device_         = comm_vk_physical_device_context_.vk_physical_device_;
    std::cout << "Successfully created Vulkan physical device." << '\n';
    return true;
}

bool VulkanSample::create_logical_device()
{
    auto device_chain = common::logicaldevice::create_logical_device_context(comm_vk_physical_device_context_) |
                        common::logicaldevice::require_extensions({VK_KHR_SWAPCHAIN_EXTENSION_NAME}) |
                        common::logicaldevice::add_graphics_queue("main_graphics", vk_window_helper_->GetSurface()) |
                        common::logicaldevice::add_transfer_queue("upload") |
                        common::logicaldevice::validate_device_configuration() |
                        common::logicaldevice::create_logical_device();

    auto result = device_chain.evaluate();
    if (!callable::is_ok(result))
    {
        std::string error_msg = std::get<std::string>(result);
        std::cerr << "Failed to create Vulkan logical device: " << error_msg << '\n';
        return false;
    }
    comm_vk_logical_device_context_ = std::get<common::CommVkLogicalDeviceContext>(result);
    comm_vk_logical_device_         = comm_vk_logical_device_context_.vk_logical_device_;
    comm_vk_graphics_queue_ = common::logicaldevice::get_queue(comm_vk_logical_device_context_, "main_graphics");
    comm_vk_transfer_queue_ = common::logicaldevice::get_queue(comm_vk_logical_device_context_, "upload");
    std::cout << "Successfully created Vulkan logical device." << '\n';
    return true;
}

bool VulkanSample::create_swapchain()
{
    // create swapchain

    auto swapchain_chain =
        common::swapchain::create_swapchain_context(comm_vk_logical_device_context_, vk_window_helper_->GetSurface()) |
        common::swapchain::set_surface_format(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) |
        common::swapchain::set_present_mode(VK_PRESENT_MODE_FIFO_KHR) | 
        common::swapchain::set_image_count(2, 3) |
        common::swapchain::set_desired_extent(static_cast<uint32_t>(engine_config_.window_config.width),
                                              static_cast<uint32_t>(engine_config_.window_config.height)) |
        common::swapchain::query_surface_support() | 
        common::swapchain::select_swapchain_settings() |
        common::swapchain::create_swapchain();

    auto result = swapchain_chain.evaluate();
    if (!callable::is_ok(result))
    {
        std::string error_msg = std::get<std::string>(result);
        std::cerr << "Failed to create Vulkan swapchain: " << error_msg << '\n';
        return false;
    }
    std::cout << "Successfully created Vulkan swapchain." << '\n';
    auto tmp_swapchain_ctx = std::get<common::CommVkSwapchainContext>(result);

    // create swapchain images and image views

    auto swapchain_image_related_chain = callable::make_chain(std::move(tmp_swapchain_ctx)) |
                                         common::swapchain::get_swapchain_images() |
                                         common::swapchain::create_image_views();
    auto swapchain_image_result = swapchain_image_related_chain.evaluate();
    if (!callable::is_ok(swapchain_image_result))
    {
        std::string error_msg = std::get<std::string>(swapchain_image_result);
        std::cerr << "Failed to create Vulkan swapchain image views: " << error_msg << '\n';
        return false;
    }
    std::cout << "Successfully created Vulkan swapchain image views." << '\n';

    // get final swapchain context
    comm_vk_swapchain_context_ = std::get<common::CommVkSwapchainContext>(swapchain_image_result);
    comm_vk_swapchain_         = comm_vk_swapchain_context_.vk_swapchain_;

    return true;
}

bool VulkanSample::create_command_pool()
{
    vk_command_buffer_helper_ = std::make_unique<VulkanCommandBufferHelper>();

    auto queue_family_index =
        common::logicaldevice::find_optimal_queue_family(comm_vk_logical_device_context_, VK_QUEUE_GRAPHICS_BIT);
    if (!queue_family_index.has_value())
    {
        std::cerr << "Failed to find any suitable graphics queue family." << '\n';
        return false;
    }
    return vk_command_buffer_helper_->CreateCommandPool(comm_vk_logical_device_, queue_family_index.value());
}

bool VulkanSample::create_uniform_buffers()
{
    for (int i = 0; i < engine_config_.frame_count; ++i)
    {
        auto current_mvp_matrix   = mvp_matrices_[i];
        VkBufferCreateInfo buffer_create_info = {};
        buffer_create_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_create_info.usage              = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
        buffer_create_info.size               = sizeof(SMvpMatrix);
        vra::VraDataDesc data_desc{
            vra::VraDataMemoryPattern::CPU_GPU, vra::VraDataUpdateRate::Frequent, buffer_create_info};
        vra::VraRawData raw_data{.pData_ = &current_mvp_matrix, .size_ = sizeof(SMvpMatrix)};
        uniform_buffer_id_.push_back(0);
        vra_data_batcher_->Collect(data_desc, raw_data, uniform_buffer_id_.back());
    }

    uniform_batch_handle_ = vra_data_batcher_->Batch();

    // get buffer create info
    const auto& uniform_buffer_create_info =
        uniform_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Frequently].data_desc.GetBufferCreateInfo();

    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage                   = VMA_MEMORY_USAGE_AUTO;
    allocation_create_info.flags = vra_data_batcher_->GetSuggestVmaMemoryFlags(vra::VraDataMemoryPattern::CPU_GPU,
                                                                               vra::VraDataUpdateRate::Frequent);
    allocation_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return Logger::LogWithVkResult(vmaCreateBuffer(vma_allocator_,
                                                   &uniform_buffer_create_info,
                                                   &allocation_create_info,
                                                   &uniform_buffer_,
                                                   &uniform_buffer_allocation_,
                                                   &uniform_buffer_allocation_info_),
                                   "Failed to create uniform buffer",
                                   "Succeeded in creating uniform buffer");
}

bool VulkanSample::create_and_write_descriptor_relatives()
{
    // create descriptor pool
    VkDescriptorType dynamic_uniform_buffer_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;

    VkDescriptorPoolSize pool_size{};
    pool_size.type            = dynamic_uniform_buffer_type;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes    = &pool_size;
    pool_info.maxSets       = 1;

    if (!Logger::LogWithVkResult(vkCreateDescriptorPool(comm_vk_logical_device_, &pool_info, nullptr, &descriptor_pool_),
                                 "Failed to create descriptor pool",
                                 "Succeeded in creating descriptor pool"))
        return false;

    // create descriptor set layout

    VkDescriptorSetLayoutBinding layout_binding{};
    layout_binding.binding         = 0;
    layout_binding.descriptorType  = dynamic_uniform_buffer_type;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings    = &layout_binding;

    if (!Logger::LogWithVkResult(
            vkCreateDescriptorSetLayout(comm_vk_logical_device_, &layout_info, nullptr, &descriptor_set_layout_),
            "Failed to create descriptor set layout",
            "Succeeded in creating descriptor set layout"))
        return false;

    // allocate descriptor set

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool     = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts        = &descriptor_set_layout_;

    if (!Logger::LogWithVkResult(vkAllocateDescriptorSets(comm_vk_logical_device_, &alloc_info, &descriptor_set_),
                                 "Failed to allocate descriptor set",
                                 "Succeeded in allocating descriptor set"))
        return false;

    // write descriptor set
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = uniform_buffer_;
    buffer_info.offset = 0;
    buffer_info.range  = sizeof(SMvpMatrix);

    VkWriteDescriptorSet descriptor_write{};
    descriptor_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet          = descriptor_set_;
    descriptor_write.dstBinding      = 0;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType  = dynamic_uniform_buffer_type;
    descriptor_write.descriptorCount = 1;
    descriptor_write.pBufferInfo     = &buffer_info;

    vkUpdateDescriptorSets(comm_vk_logical_device_, 1, &descriptor_write, 0, nullptr);

    return true;
}

bool VulkanSample::create_vma_vra_objects()
{
    // vra and vma members
    vra_data_batcher_ = std::make_unique<vra::VraDataBatcher>(comm_vk_physical_device_);

    VmaAllocatorCreateInfo allocator_create_info = {};
    allocator_create_info.flags                  = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    allocator_create_info.vulkanApiVersion       = VK_API_VERSION_1_3;
    allocator_create_info.physicalDevice         = comm_vk_physical_device_;
    allocator_create_info.device                 = comm_vk_logical_device_;
    allocator_create_info.instance               = comm_vk_instance_;

    return Logger::LogWithVkResult(vmaCreateAllocator(&allocator_create_info, &vma_allocator_),
                                   "Failed to create Vulkan vra and vma objects",
                                   "Succeeded in creating Vulkan vra and vma objects");
}

// 查找支持的深度格式
VkFormat VulkanSample::find_supported_depth_format()
{
    // 按优先级尝试不同的深度格式
    std::vector<VkFormat> candidates = {
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};

    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(comm_vk_physical_device_, format, &props);

        // 检查该格式是否支持作为深度附件的最佳平铺格式
        if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0U)
        {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported depth format");
}

// 创建深度资源
bool VulkanSample::create_depth_resources()
{
    // 获取深度格式
    depth_format_ = find_supported_depth_format();

    // 创建深度图像
    VkImageCreateInfo image_info{};
    image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType     = VK_IMAGE_TYPE_2D;
    image_info.extent.width  = comm_vk_swapchain_context_.swapchain_info_.extent_.width;
    image_info.extent.height = comm_vk_swapchain_context_.swapchain_info_.extent_.height;
    image_info.extent.depth  = 1;
    image_info.mipLevels     = 1;
    image_info.arrayLayers   = 1;
    image_info.format        = depth_format_;
    image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    // 创建图像
    if (vkCreateImage(comm_vk_logical_device_, &image_info, nullptr, &depth_image_) != VK_SUCCESS)
    {
        Logger::LogError("Failed to create depth image");
        return false;
    }

    // 获取内存需求
    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(comm_vk_logical_device_, depth_image_, &mem_requirements);

    // 分配内存
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;

    // 查找适合的内存类型
    uint32_t memory_type_index = 0;
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(comm_vk_physical_device_, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
    {
        if (((mem_requirements.memoryTypeBits & (1 << i)) != 0U) &&
            ((mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0U))
        {
            memory_type_index = i;
            break;
        }
    }

    alloc_info.memoryTypeIndex = memory_type_index;

    // 分配内存
    if (vkAllocateMemory(comm_vk_logical_device_, &alloc_info, nullptr, &depth_memory_) != VK_SUCCESS)
    {
        Logger::LogError("Failed to allocate depth image memory");
        return false;
    }

    // 绑定内存到图像
    if (vkBindImageMemory(comm_vk_logical_device_, depth_image_, depth_memory_, 0) != VK_SUCCESS)
    {
        Logger::LogError("Failed to bind depth image memory");
        return false;
    }

    // 创建图像视图
    VkImageViewCreateInfo view_info{};
    view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image                           = depth_image_;
    view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format                          = depth_format_;
    view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(comm_vk_logical_device_, &view_info, nullptr, &depth_image_view_) != VK_SUCCESS)
    {
        Logger::LogError("Failed to create depth image view");
        return false;
    }

    return true;
}

bool VulkanSample::create_frame_buffer()
{
    // 创建深度资源
    if (!create_depth_resources())
    {
        throw std::runtime_error("Failed to create depth resources.");
    }

    // 创建帧缓冲
    SVulkanFrameBufferConfig framebuffer_config(
        comm_vk_swapchain_context_.swapchain_info_.extent_, 
        comm_vk_swapchain_context_.swapchain_image_views_, 
        depth_image_view_);

    vk_frame_buffer_helper_ = std::make_unique<VulkanFrameBufferHelper>(comm_vk_logical_device_, framebuffer_config);

    return vk_frame_buffer_helper_->CreateFrameBuffer(vk_renderpass_helper_->GetRenderpass());
}

bool VulkanSample::create_pipeline()
{
    // create shader
    vk_shader_helper_ = std::make_unique<VulkanShaderHelper>(comm_vk_logical_device_);

    std::vector<SVulkanShaderConfig> configs;
    std::string shader_path = engine_config_.general_config.working_directory + "src\\shader\\";
    // std::string vertex_shader_path = shader_path + "triangle.vert.spv";
    // std::string fragment_shader_path = shader_path + "triangle.frag.spv";
    std::string vertex_shader_path   = shader_path + "gltf.vert.spv";
    std::string fragment_shader_path = shader_path + "gltf.frag.spv";
    configs.push_back({.shader_type = EShaderType::kVertexShader, .shader_path = vertex_shader_path.c_str()});
    configs.push_back({.shader_type = EShaderType::kFragmentShader, .shader_path = fragment_shader_path.c_str()});

    for (const auto& config : configs)
    {
        std::vector<uint32_t> shader_code;
        if (!vk_shader_helper_->ReadShaderCode(config.shader_path, shader_code))
        {
            Logger::LogError("Failed to read shader code from " + std::string(config.shader_path));
            return false;
        }

        if (!vk_shader_helper_->CreateShaderModule(comm_vk_logical_device_, shader_code, config.shader_type))
        {
            Logger::LogError("Failed to create shader module for " + std::string(config.shader_path));
            return false;
        }
    }

    // create renderpass
    SVulkanRenderpassConfig renderpass_config{};
    renderpass_config.color_format = comm_vk_swapchain_context_.swapchain_info_.surface_format_.format;
    renderpass_config.depth_format = VK_FORMAT_D32_SFLOAT;  // TODO: Make configurable
    renderpass_config.sample_count = VK_SAMPLE_COUNT_1_BIT; // TODO: Make configurable
    vk_renderpass_helper_            = std::make_unique<VulkanRenderpassHelper>(renderpass_config);
    if (!vk_renderpass_helper_->CreateRenderpass(comm_vk_logical_device_))
    {
        return false;
    }

    // create pipeline
    SVulkanPipelineConfig pipeline_config;
    pipeline_config.swap_chain_extent = comm_vk_swapchain_context_.swapchain_info_.extent_;
    pipeline_config.shader_module_map = {
        {EShaderType::kVertexShader, vk_shader_helper_->GetShaderModule(EShaderType::kVertexShader)},
        {EShaderType::kFragmentShader, vk_shader_helper_->GetShaderModule(EShaderType::kFragmentShader)}};
    pipeline_config.renderpass = vk_renderpass_helper_->GetRenderpass();
    // pipeline_config.vertex_input_binding_description =
    // vertex_input_binding_description_;
    // pipeline_config.vertex_input_attribute_descriptions =
    // {vertex_input_attribute_position_, vertex_input_attribute_color_};
    pipeline_config.vertex_input_binding_description    = test_vertex_input_binding_description_;
    pipeline_config.vertex_input_attribute_descriptions = test_vertex_input_attributes_;
    pipeline_config.descriptor_set_layouts.push_back(descriptor_set_layout_);
    vk_pipeline_helper_ = std::make_unique<VulkanPipelineHelper>(pipeline_config);
    return vk_pipeline_helper_->CreatePipeline(comm_vk_logical_device_);
}

bool VulkanSample::allocate_per_frame_command_buffer()
{
    for (int i = 0; i < engine_config_.frame_count; ++i)
    {
        if (!vk_command_buffer_helper_->AllocateCommandBuffer(
                {.command_buffer_level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .command_buffer_count = 1},
                output_frames_[i].command_buffer_id))
        {
            Logger::LogError("Failed to allocate command buffer for frame " + std::to_string(i));
            return false;
        }
    }
    return true;
}

bool VulkanSample::create_synchronization_objects()
{
    vk_synchronization_helper_ = std::make_unique<VulkanSynchronizationHelper>(comm_vk_logical_device_);
    // create synchronization objects
    for (int i = 0; i < engine_config_.frame_count; ++i)
    {
        if (!vk_synchronization_helper_->CreateVkSemaphore(output_frames_[i].image_available_semaphore_id))
            return false;
        if (!vk_synchronization_helper_->CreateVkSemaphore(output_frames_[i].render_finished_semaphore_id))
            return false;
        if (!vk_synchronization_helper_->CreateFence(output_frames_[i].fence_id))
            return false;
    }
    return true;
}

// ----------------------------------
// private function to draw the frame
// ----------------------------------

void VulkanSample::draw_frame()
{
    // get current resource
    auto current_fence_id                     = output_frames_[frame_index_].fence_id;
    auto current_image_available_semaphore_id = output_frames_[frame_index_].image_available_semaphore_id;
    auto current_render_finished_semaphore_id = output_frames_[frame_index_].render_finished_semaphore_id;
    auto current_command_buffer_id            = output_frames_[frame_index_].command_buffer_id;
    auto current_queue_id                     = output_frames_[frame_index_].queue_id;

    // wait for last frame to finish
    if (!vk_synchronization_helper_->WaitForFence(current_fence_id))
        return;

    // get semaphores
    auto* image_available_semaphore = vk_synchronization_helper_->GetSemaphore(current_image_available_semaphore_id);
    auto* render_finished_semaphore = vk_synchronization_helper_->GetSemaphore(current_render_finished_semaphore_id);
    auto* in_flight_fence           = vk_synchronization_helper_->GetFence(current_fence_id);

    // acquire next image
    uint32_t image_index    = 0;
    VkResult acquire_result = vkAcquireNextImageKHR(comm_vk_logical_device_,
                                                    comm_vk_swapchain_,
                                                    UINT64_MAX,
                                                    image_available_semaphore,
                                                    VK_NULL_HANDLE,
                                                    &image_index);
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR || acquire_result == VK_SUBOPTIMAL_KHR)
    {
        resize_request_ = true;
        return;
    }
    if (acquire_result != VK_SUCCESS)
    {
        Logger::LogWithVkResult(acquire_result, "Failed to acquire next image", "Succeeded in acquiring next image");
        return;
    }

    // reset fence before submitting
    if (!vk_synchronization_helper_->ResetFence(current_fence_id))
        return;

    // record command buffer
    if (!vk_command_buffer_helper_->ResetCommandBuffer(current_command_buffer_id))
        return;
    if (!record_command(image_index, current_command_buffer_id))
        return;

    // submit command buffer
    VkCommandBufferSubmitInfo command_buffer_submit_info{};
    command_buffer_submit_info.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    command_buffer_submit_info.commandBuffer = vk_command_buffer_helper_->GetCommandBuffer(current_command_buffer_id);

    VkSemaphoreSubmitInfo wait_semaphore_info{};
    wait_semaphore_info.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_semaphore_info.semaphore = image_available_semaphore;
    wait_semaphore_info.value     = 1;

    VkSemaphoreSubmitInfo signal_semaphore_info{};
    signal_semaphore_info.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal_semaphore_info.semaphore = render_finished_semaphore;
    signal_semaphore_info.value     = 1;

    VkSubmitInfo2 submit_info{};
    submit_info.sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit_info.commandBufferInfoCount   = 1;
    submit_info.pCommandBufferInfos      = &command_buffer_submit_info;
    submit_info.waitSemaphoreInfoCount   = 1;
    submit_info.pWaitSemaphoreInfos      = &wait_semaphore_info;
    submit_info.signalSemaphoreInfoCount = 1;
    submit_info.pSignalSemaphoreInfos    = &signal_semaphore_info;
    if (!Logger::LogWithVkResult(
            vkQueueSubmit2(comm_vk_graphics_queue_, 1, &submit_info, in_flight_fence),
            "Failed to submit command buffer",
            "Succeeded in submitting command buffer"))
    {
        return;
    }

    // present the image
    VkPresentInfoKHR present_info{};
    present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores    = &render_finished_semaphore;
    present_info.swapchainCount     = 1;
    present_info.pSwapchains        = &comm_vk_swapchain_;
    present_info.pImageIndices      = &image_index;
    VkResult present_result = vkQueuePresentKHR(comm_vk_graphics_queue_, &present_info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR)
    {
        resize_request_ = true;
        return;
    }
    if (present_result != VK_SUCCESS)
    {
        Logger::LogWithVkResult(present_result, "Failed to present image", "Succeeded in presenting image");
        return;
    }

    // update frame index
    frame_index_ = (frame_index_ + 1) % engine_config_.frame_count;
}

void VulkanSample::resize_swapchain()
{
    // wait for the device to be idle
    vkDeviceWaitIdle(comm_vk_logical_device_);

    // destroy old vulkan objects

    for (auto *image_view : comm_vk_swapchain_context_.swapchain_image_views_)
    {
        vkDestroyImageView(comm_vk_logical_device_, image_view, nullptr);
    }
    // Note: Don't destroy swapchain images as they are owned by the swapchain
    vkDestroySwapchainKHR(comm_vk_logical_device_, comm_vk_swapchain_, nullptr);
    vk_frame_buffer_helper_.reset();

    // reset window size
    auto current_extent     = vk_window_helper_->GetCurrentWindowExtent();
    engine_config_.window_config.width  = current_extent.width;
    engine_config_.window_config.height = current_extent.height;

    // create new swapchain
    if (!create_swapchain())
    {
        throw std::runtime_error("Failed to create Vulkan swap chain.");
    }

    // recreate framebuffers
    if (!create_frame_buffer())
    {
        throw std::runtime_error("Failed to create Vulkan frame buffer.");
    }

    resize_request_ = false;
}

bool VulkanSample::record_command(uint32_t image_index, const std::string& command_buffer_id)
{
    // 更新当前帧的 Uniform Buffer
    update_uniform_buffer(image_index);

    // begin command recording
    if (!vk_command_buffer_helper_->BeginCommandBufferRecording(command_buffer_id,
                                                             VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        return false;

    // collect needed objects
    auto *command_buffer = vk_command_buffer_helper_->GetCommandBuffer(command_buffer_id);

    // 从暂存缓冲区复制到本地缓冲区
    VkBufferCopy buffer_copy_info{};
    buffer_copy_info.srcOffset = 0;
    buffer_copy_info.dstOffset = 0;
    buffer_copy_info.size      = test_staging_buffer_allocation_info_.size;
    vkCmdCopyBuffer(command_buffer, test_staging_buffer_, test_local_buffer_, 1, &buffer_copy_info);

    // 设置内存屏障以确保复制完成
    VkBufferMemoryBarrier2 buffer_memory_barrier{};
    buffer_memory_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    buffer_memory_barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    buffer_memory_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    buffer_memory_barrier.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    buffer_memory_barrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    buffer_memory_barrier.buffer        = test_local_buffer_;
    buffer_memory_barrier.offset        = 0;
    buffer_memory_barrier.size          = VK_WHOLE_SIZE;

    VkDependencyInfo dependency_info{};
    dependency_info.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency_info.bufferMemoryBarrierCount = 1;
    dependency_info.pBufferMemoryBarriers    = &buffer_memory_barrier;
    vkCmdPipelineBarrier2(command_buffer, &dependency_info);

    // begin renderpass
    VkClearValue clear_color     = {};
    clear_color.color.float32[0] = 0.1F;
    clear_color.color.float32[1] = 0.1F;
    clear_color.color.float32[2] = 0.1F;
    clear_color.color.float32[3] = 1.0F;

    VkClearValue clear_values[2];
    clear_values[0].color        = {{0.1F, 0.1F, 0.1F, 1.0F}};
    clear_values[1].depthStencil = {.depth = 1.0F, .stencil = 0}; // 设置深度清除值为1.0（远面）

    VkRenderPassBeginInfo renderpass_info{};
    renderpass_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderpass_info.renderPass        = vk_renderpass_helper_->GetRenderpass();
    renderpass_info.framebuffer       = (*vk_frame_buffer_helper_->GetFramebuffers())[image_index];
    renderpass_info.renderArea.offset = {.x = 0, .y = 0};
    renderpass_info.renderArea.extent = comm_vk_swapchain_context_.swapchain_info_.extent_;
    renderpass_info.clearValueCount   = 2;
    renderpass_info.pClearValues      = clear_values;

    vkCmdBeginRenderPass(command_buffer, &renderpass_info, VK_SUBPASS_CONTENTS_INLINE);

    // bind pipeline
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_helper_->GetPipeline());

    // bind descriptor set
    auto offset =
        uniform_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Frequently].offsets[uniform_buffer_id_[frame_index_]];
    auto dynamic_offset = static_cast<uint32_t>(offset);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk_pipeline_helper_->GetPipelineLayout(),
                            0,
                            1,
                            &descriptor_set_,
                            1,
                            &dynamic_offset);

    // dynamic state update
    VkViewport viewport{};
    viewport.x        = 0.0F;
    viewport.y        = 0.0F;
    viewport.width    = static_cast<float>(comm_vk_swapchain_context_.swapchain_info_.extent_.width);
    viewport.height   = static_cast<float>(comm_vk_swapchain_context_.swapchain_info_.extent_.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {.x=0, .y=0};
    scissor.extent = comm_vk_swapchain_context_.swapchain_info_.extent_;
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    // 绑定顶点和索引缓冲区
    vkCmdBindVertexBuffers(
        command_buffer,
        0,
        1,
        &test_local_buffer_,
        &test_local_host_batch_handle_[vra::VraBuiltInBatchIds::GPU_Only].offsets[test_vertex_buffer_id_]);
    vkCmdBindIndexBuffer(
        command_buffer,
        test_local_buffer_,
        test_local_host_batch_handle_[vra::VraBuiltInBatchIds::GPU_Only].offsets[test_index_buffer_id_],
        VK_INDEX_TYPE_UINT32);

    // 遍历每个 mesh 进行绘制
    for (const auto& mesh : mesh_list_)
    {
        // 遍历该 mesh 的所有图元进行绘制
        // if (std::strcmp(mesh.name.c_str(), "arch_stones_02") != 0) {
        //     continue;
        // }
        for (const auto & primitive : mesh.primitives)
        {
            // 绘制当前图元
            vkCmdDrawIndexed(command_buffer,
                             primitive.index_count, // 使用实际的索引数量
                             1,
                             primitive.first_index, // 使用实际的索引偏移量
                             0,
                             0);
        }
    }

    // end renderpass
    vkCmdEndRenderPass(command_buffer);

    // end command recording
    return vk_command_buffer_helper_->EndCommandBufferRecording(command_buffer_id);
}

void VulkanSample::update_uniform_buffer(uint32_t current_frame_index)
{
    // update the model matrix (添加适当的缩放)
    mvp_matrices_[current_frame_index].model = glm::mat4(1.0F);

    // update the view matrix
    mvp_matrices_[current_frame_index].view = glm::lookAt(camera_.position,                 // camera position
                                                          camera_.position + camera_.front, // camera looking at point
                                                          camera_.up                        // camera up direction
    );

    // update the projection matrix
    mvp_matrices_[current_frame_index].projection =
        glm::perspective(glm::radians(camera_.zoom), // FOV
                         static_cast<float>(comm_vk_swapchain_context_.swapchain_info_.extent_.width) /
                             static_cast<float>(comm_vk_swapchain_context_.swapchain_info_.extent_.height), // aspect ratio
                         0.1F,                                                                 // near plane
                         1000.0F // 增加远平面距离，确保能看到远处的物体
        );

    // reverse the Y-axis in Vulkan's NDC coordinate system
    mvp_matrices_[current_frame_index].projection[1][1] *= -1;

    // map vulkan host memory to update the uniform buffer
    uniform_buffer_mapped_data_ = nullptr;
    vmaMapMemory(vma_allocator_, uniform_buffer_allocation_, &uniform_buffer_mapped_data_);

    // get the offset of the current frame in the uniform buffer
    auto offset = uniform_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Frequently]
                      .offsets[uniform_buffer_id_[current_frame_index]];
    uint8_t* data_location = static_cast<uint8_t*>(uniform_buffer_mapped_data_) + offset;

    // copy the data to the mapped memory
    memcpy(data_location, &mvp_matrices_[current_frame_index], sizeof(SMvpMatrix));

    // unmap the memory
    vmaUnmapMemory(vma_allocator_, uniform_buffer_allocation_);
    uniform_buffer_mapped_data_ = nullptr;
}

// add a function to focus on an object
void VulkanSample::focus_on_object(const glm::vec3& object_position, float target_distance)
{
    camera_.focus_point     = object_position;
    camera_.has_focus_point = true;

    // calculate the position the camera should move to
    glm::vec3 direction = glm::normalize(camera_.position - object_position);
    camera_.position    = object_position + direction * target_distance;

    // update the camera direction
    camera_.front = glm::normalize(object_position - camera_.position);
    camera_.right = glm::normalize(glm::cross(camera_.front, camera_.world_up));
    camera_.up    = glm::normalize(glm::cross(camera_.right, camera_.front));

    // update the yaw and pitch
    glm::vec3 front = camera_.front;
    camera_.pitch   = glm::degrees(asin(front.y));
    camera_.yaw     = glm::degrees(atan2(front.z, front.x));
}

void VulkanSample::create_drawcall_list_buffer()
{
    vra::VraRawData vertex_buffer_data{.pData_ = vertices_.data(), .size_ = sizeof(gltf::Vertex) * vertices_.size()};
    vra::VraRawData index_buffer_data{.pData_ = indices_.data(), .size_ = sizeof(uint32_t) * indices_.size()};

    // 顶点缓冲区创建信息
    VkBufferCreateInfo vertex_buffer_create_info{};
    vertex_buffer_create_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vertex_buffer_create_info.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vertex_buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vra::VraDataDesc vertex_buffer_desc{
        vra::VraDataMemoryPattern::GPU_Only, vra::VraDataUpdateRate::RarelyOrNever, vertex_buffer_create_info};

    // 索引缓冲区创建信息
    VkBufferCreateInfo index_buffer_create_info{};
    index_buffer_create_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    index_buffer_create_info.usage       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    index_buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vra::VraDataDesc index_buffer_desc{
        vra::VraDataMemoryPattern::GPU_Only, vra::VraDataUpdateRate::RarelyOrNever, index_buffer_create_info};

    // 暂存缓冲区创建信息
    VkBufferCreateInfo staging_buffer_create_info{};
    staging_buffer_create_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_buffer_create_info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    staging_buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vra::VraDataDesc staging_vertex_buffer_desc{
        vra::VraDataMemoryPattern::CPU_GPU, vra::VraDataUpdateRate::RarelyOrNever, staging_buffer_create_info};
    vra::VraDataDesc staging_index_buffer_desc{
        vra::VraDataMemoryPattern::CPU_GPU, vra::VraDataUpdateRate::RarelyOrNever, staging_buffer_create_info};

    if (!vra_data_batcher_->Collect(vertex_buffer_desc, vertex_buffer_data, test_vertex_buffer_id_))
    {
        Logger::LogError("Failed to collect vertex buffer data");
        return;
    }
    if (!vra_data_batcher_->Collect(index_buffer_desc, index_buffer_data, test_index_buffer_id_))
    {
        Logger::LogError("Failed to collect index buffer data");
        return;
    }
    if (!vra_data_batcher_->Collect(staging_vertex_buffer_desc, vertex_buffer_data, test_staging_vertex_buffer_id_))
    {
        Logger::LogError("Failed to collect staging vertex buffer data");
        return;
    }
    if (!vra_data_batcher_->Collect(staging_index_buffer_desc, index_buffer_data, test_staging_index_buffer_id_))
    {
        Logger::LogError("Failed to collect staging index buffer data");
        return;
    }

    // 执行批处理
    test_local_host_batch_handle_ = vra_data_batcher_->Batch();

    // 创建本地缓冲区
    auto test_local_buffer_create_info =
        test_local_host_batch_handle_[vra::VraBuiltInBatchIds::GPU_Only].data_desc.GetBufferCreateInfo();
    VmaAllocationCreateInfo allocation_create_info{};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(vma_allocator_,
                    &test_local_buffer_create_info,
                    &allocation_create_info,
                    &test_local_buffer_,
                    &test_local_buffer_allocation_,
                    &test_local_buffer_allocation_info_);

    // 创建暂存缓冲区
    auto test_host_buffer_create_info =
        test_local_host_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Rarely].data_desc.GetBufferCreateInfo();
    VmaAllocationCreateInfo staging_allocation_create_info{};
    staging_allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    staging_allocation_create_info.flags = vra_data_batcher_->GetSuggestVmaMemoryFlags(
        vra::VraDataMemoryPattern::CPU_GPU, vra::VraDataUpdateRate::RarelyOrNever);
    vmaCreateBuffer(vma_allocator_,
                    &test_host_buffer_create_info,
                    &staging_allocation_create_info,
                    &test_staging_buffer_,
                    &test_staging_buffer_allocation_,
                    &test_staging_buffer_allocation_info_);

    // 复制数据到暂存缓冲区
    auto consolidate_data = test_local_host_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Rarely].consolidated_data;
    void* data = nullptr;
    vmaInvalidateAllocation(vma_allocator_, test_staging_buffer_allocation_, 0, VK_WHOLE_SIZE);
    vmaMapMemory(vma_allocator_, test_staging_buffer_allocation_, &data);
    memcpy(data, consolidate_data.data(), consolidate_data.size());
    vmaUnmapMemory(vma_allocator_, test_staging_buffer_allocation_);
    vmaFlushAllocation(vma_allocator_, test_staging_buffer_allocation_, 0, VK_WHOLE_SIZE);

    // 设置顶点输入绑定描述
    test_vertex_input_binding_description_.binding   = 0;
    test_vertex_input_binding_description_.stride    = sizeof(gltf::Vertex);
    test_vertex_input_binding_description_.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // 设置顶点属性描述
    test_vertex_input_attributes_.clear();

    // 使用更安全的偏移量计算，确保 offsetof 计算正确
    // position
    test_vertex_input_attributes_.push_back(VkVertexInputAttributeDescription{
        .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(gltf::Vertex, position)});
    // color
    test_vertex_input_attributes_.push_back(VkVertexInputAttributeDescription{
        .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = offsetof(gltf::Vertex, color)});
    // normal
    test_vertex_input_attributes_.push_back(VkVertexInputAttributeDescription{
        .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(gltf::Vertex, normal)});
    // tangent
    test_vertex_input_attributes_.push_back(
        VkVertexInputAttributeDescription{.location = 3,
                                          .binding  = 0,
                                          .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
                                          .offset   = offsetof(gltf::Vertex, tangent)});
    // uv0
    test_vertex_input_attributes_.push_back(VkVertexInputAttributeDescription{
        .location = 4, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(gltf::Vertex, uv0)});
    // uv1
    test_vertex_input_attributes_.push_back(VkVertexInputAttributeDescription{
        .location = 5, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(gltf::Vertex, uv1)});
}
