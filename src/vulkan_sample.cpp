#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_CONSTRUCTORS

#include "vulkan_sample.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "_callable/callable.h"
#include "_templates/common.hpp"

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
    initialize_window();
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
    comm_vk_logical_device_.waitIdle();

    // 销毁深度资源
    if (depth_image_view_ != VK_NULL_HANDLE)
    {
        comm_vk_logical_device_.destroyImageView(depth_image_view_);
        depth_image_view_ = VK_NULL_HANDLE;
    }

    if (depth_image_ != VK_NULL_HANDLE)
    {
        comm_vk_logical_device_.destroyImage(depth_image_);
        depth_image_ = VK_NULL_HANDLE;
    }

    if (depth_memory_ != VK_NULL_HANDLE)
    {
        comm_vk_logical_device_.freeMemory(depth_memory_, nullptr);
        depth_memory_ = VK_NULL_HANDLE;
    }

    // 销毁描述符相关资源
    if (descriptor_pool_ != VK_NULL_HANDLE)
    {
        comm_vk_logical_device_.destroyDescriptorPool(descriptor_pool_);
        descriptor_pool_ = VK_NULL_HANDLE;
    }

    if (descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        comm_vk_logical_device_.destroyDescriptorSetLayout(descriptor_set_layout_);
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

    for (auto image_view : comm_vk_swapchain_context_.swapchain_image_views_)
    {
        comm_vk_logical_device_.destroyImageView(image_view);
    }
    comm_vk_logical_device_.destroySwapchainKHR(comm_vk_swapchain_);

    // release unique pointer

    vk_shader_helper_.reset();
    window_->close();
    window_.reset();
    vk_renderpass_helper_.reset();
    vk_pipeline_helper_.reset();
    vk_frame_buffer_helper_.reset();
    vk_command_buffer_helper_.reset();
    vk_synchronization_helper_.reset();

    // destroy comm test data

    vkDestroyDevice(comm_vk_logical_device_, nullptr);
    vkDestroySurfaceKHR(comm_vk_instance_, surface_, nullptr);
    vkDestroyInstance(comm_vk_instance_, nullptr);

    // 重置单例指针
    instance = nullptr;
}

void VulkanSample::initialize_vulkan_hpp()
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
}

// Initialize the engine
void VulkanSample::initialize_window()
{
    window_ = std::make_unique<interface::SDLWindow>();
    interface::WindowConfig config;
    config.title  = engine_config_.window_config.title;
    config.width  = engine_config_.window_config.width;
    config.height = engine_config_.window_config.height;

    if (!window_->open(config))
    {
        throw std::runtime_error("Failed to create window.");
    }
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
    mvp_matrices_ = std::vector<SMvpMatrix>(
        engine_config_.frame_count,
        {.model = glm::mat4(1.0F), .view = glm::mat4(1.0F), .projection = glm::mat4(1.0F)});

    // initialize camera
    simple_camera_ = std::make_unique<interface::SimpleCamera>();


}

// Main loop
void VulkanSample::Run()
{
    engine_state_ = EWindowState::kRunning;

    // main loop
    interface::InputEvent e{};
    while (engine_state_ != EWindowState::kStopped)
    {
        // handle events on queue
        window_->tick(e);
        if (window_->should_close())
        {
            engine_state_ = EWindowState::kStopped;
        }
        // tackle I/O event
        // on_event(e);
        simple_camera_->tick(e);
        // process keyboard input to update camera
        // process_keyboard_input(delta_time);

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
    auto extensions     = window_->get_required_instance_extensions();
    auto instance_chain = common::instance::create_context() | common::instance::set_application_name("My Vulkan App") |
                          common::instance::set_engine_name("My Engine") |
                          common::instance::set_api_version(1, 3, 0) |
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
    VULKAN_HPP_DEFAULT_DISPATCHER.init(comm_vk_instance_); // a must for loading all other function pointers!
    std::cout << "Successfully created Vulkan instance." << '\n';
    return true;
}

bool VulkanSample::create_surface()
{
    return window_->create_vulkan_surface(comm_vk_instance_, &surface_);
}

bool VulkanSample::create_physical_device()
{
    // vulkan 1.3 features - 用于检查硬件支持
    vk::PhysicalDeviceVulkan13Features features_13{.synchronization2 = vk::True};

    auto physical_device_chain = common::physicaldevice::create_physical_device_context(comm_vk_instance_) |
                                 common::physicaldevice::set_surface(surface_) |
                                 common::physicaldevice::require_api_version(1, 3, 0) |
                                 common::physicaldevice::require_features_13(features_13) |
                                 common::physicaldevice::require_queue(vk::QueueFlagBits::eGraphics, 1, true) |
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
                        common::logicaldevice::require_extensions({vk::KHRSwapchainExtensionName}) |
                        common::logicaldevice::add_graphics_queue("main_graphics", surface_) |
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
    comm_vk_graphics_queue_         = common::logicaldevice::get_queue(comm_vk_logical_device_context_, "main_graphics");
    comm_vk_transfer_queue_         = common::logicaldevice::get_queue(comm_vk_logical_device_context_, "upload");
    std::cout << "Successfully created Vulkan logical device." << '\n';
    return true;
}

bool VulkanSample::create_swapchain()
{
    // create swapchain

    auto swapchain_chain =
        common::swapchain::create_swapchain_context(comm_vk_logical_device_context_, surface_) |
        common::swapchain::set_surface_format(vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear) |
        common::swapchain::set_present_mode(vk::PresentModeKHR::eFifo) | common::swapchain::set_image_count(2, 3) |
        common::swapchain::set_desired_extent(static_cast<uint32_t>(engine_config_.window_config.width),
                                              static_cast<uint32_t>(engine_config_.window_config.height)) |
        common::swapchain::query_surface_support() | common::swapchain::select_swapchain_settings() |
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
        common::logicaldevice::find_optimal_queue_family(comm_vk_logical_device_context_, vk::QueueFlagBits::eGraphics);
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
        auto current_mvp_matrix = mvp_matrices_[i];
        vk::BufferCreateInfo buffer_create_info;
        buffer_create_info.setSize(sizeof(SMvpMatrix))
            .setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
            .setSharingMode(vk::SharingMode::eExclusive);
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
    allocation_create_info.flags                   = vra_data_batcher_->GetSuggestVmaMemoryFlags(vra::VraDataMemoryPattern::CPU_GPU,
                                                                                                 vra::VraDataUpdateRate::Frequent);
    allocation_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return Logger::LogWithVkResult(vmaCreateBuffer(vma_allocator_,
                                                   &uniform_buffer_create_info,
                                                   &allocation_create_info,
                                                   reinterpret_cast<VkBuffer*>(&uniform_buffer_),
                                                   &uniform_buffer_allocation_,
                                                   &uniform_buffer_allocation_info_),
                                   "Failed to create uniform buffer",
                                   "Succeeded in creating uniform buffer");
}

bool VulkanSample::create_and_write_descriptor_relatives()
{
    vk::DescriptorPoolSize pool_size(vk::DescriptorType::eUniformBufferDynamic, 1);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info;
    descriptor_pool_create_info.setPoolSizes(pool_size).setPoolSizeCount(1).setMaxSets(1);

    descriptor_pool_ = comm_vk_logical_device_.createDescriptorPool(descriptor_pool_create_info, nullptr);
    if (!descriptor_pool_)
        return false;

    // create descriptor set layout

    vk::DescriptorSetLayoutBinding layout_binding;
    layout_binding.setBinding(0)
        .setDescriptorType(vk::DescriptorType::eUniformBufferDynamic)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    vk::DescriptorSetLayoutCreateInfo layout_create_info;
    layout_create_info.setBindingCount(1).setPBindings(&layout_binding);

    descriptor_set_layout_ = comm_vk_logical_device_.createDescriptorSetLayout(layout_create_info, nullptr);
    if (!descriptor_set_layout_)
        return false;

    // allocate descriptor set

    vk::DescriptorSetAllocateInfo alloc_info;
    alloc_info.setDescriptorPool(descriptor_pool_).setDescriptorSetCount(1).setPSetLayouts(&descriptor_set_layout_);

    descriptor_sets_ = comm_vk_logical_device_.allocateDescriptorSets(alloc_info);
    if (descriptor_sets_.empty())
        return false;

    // write descriptor set
    vk::DescriptorBufferInfo descriptor_buffer_info
    {
        .buffer = uniform_buffer_,
        .offset = 0,
        .range = sizeof(SMvpMatrix)
    };
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets(descriptor_sets_.size());
    for (size_t i = 0; i < descriptor_sets_.size(); ++i)
    {
        write_descriptor_sets[i] = {.dstSet = descriptor_sets_[i],
                                    .dstBinding = 0,
                                    .dstArrayElement = 0,
                                    .descriptorCount = 1,
                                    .descriptorType = vk::DescriptorType::eUniformBufferDynamic,
                                    .pBufferInfo = &descriptor_buffer_info};
    }
    comm_vk_logical_device_.updateDescriptorSets(write_descriptor_sets, {});
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
vk::Format VulkanSample::find_supported_depth_format()
{
    // 按优先级尝试不同的深度格式
    std::vector<vk::Format> candidates = {
        vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint};

    for (vk::Format format : candidates)
    {
        vk::FormatProperties props;
        comm_vk_physical_device_.getFormatProperties(format, &props);

        // 检查该格式是否支持作为深度附件的最佳平铺格式
        if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
        {
            // 如果支持，则返回该格式
            Logger::LogInfo("Supported depth format found: " + vk::to_string(format));
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
    vk::ImageCreateInfo image_info;
    image_info.setImageType(vk::ImageType::e2D)
        .setExtent({
            .width = comm_vk_swapchain_context_.swapchain_info_.extent_.width,
            .height = comm_vk_swapchain_context_.swapchain_info_.extent_.height,
            .depth = 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setFormat(depth_format_)
        .setTiling(vk::ImageTiling::eOptimal)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setSharingMode(vk::SharingMode::eExclusive);

    // 创建图像
    depth_image_ = comm_vk_logical_device_.createImage(image_info, nullptr);
    if (!depth_image_)
    {
        Logger::LogError("Failed to create depth image");
        return false;
    }

    // 获取内存需求
    vk::MemoryRequirements mem_requirements = comm_vk_logical_device_.getImageMemoryRequirements(depth_image_);

    // 分配内存
    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.setAllocationSize(mem_requirements.size);

    // 查找适合的内存类型
    uint32_t memory_type_index                        = 0;
    vk::PhysicalDeviceMemoryProperties mem_properties = comm_vk_physical_device_.getMemoryProperties();

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
    {
        if (((mem_requirements.memoryTypeBits & (1 << i)) != 0U) &&
            (mem_properties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
        {
            memory_type_index = i;
            break;
        }
    }

    alloc_info.memoryTypeIndex = memory_type_index;

    // 分配内存
    depth_memory_ = comm_vk_logical_device_.allocateMemory(alloc_info, nullptr);
    if (!depth_memory_)
    {
        Logger::LogError("Failed to allocate depth image memory");
        return false;
    }

    // 绑定内存到图像
    comm_vk_logical_device_.bindImageMemory(depth_image_, depth_memory_, 0);

    // 创建图像视图
    vk::ImageViewCreateInfo view_info{};
    view_info.setImage(depth_image_)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(depth_format_)
        .setSubresourceRange(vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eDepth)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1));

    if (comm_vk_logical_device_.createImageView(&view_info, nullptr, &depth_image_view_) != vk::Result::eSuccess)
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
    SVulkanFrameBufferConfig framebuffer_config(comm_vk_swapchain_context_.swapchain_info_.extent_,
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
    SVulkanRenderpassConfig renderpass_config{
        .color_format = comm_vk_swapchain_context_.swapchain_info_.surface_format_.format,
        .depth_format = vk::Format::eD32Sfloat,     // TODO: Make configurable
        .sample_count = vk::SampleCountFlagBits::e1 // TODO: Make configurable
    };
    vk_renderpass_helper_ = std::make_unique<VulkanRenderpassHelper>(renderpass_config);
    if (!vk_renderpass_helper_->CreateRenderpass(comm_vk_logical_device_))
    {
        return false;
    }

    // create pipeline
    SVulkanPipelineConfig pipeline_config{
        .swap_chain_extent = comm_vk_swapchain_context_.swapchain_info_.extent_,
        .shader_module_map = {{EShaderType::kVertexShader,
                               vk_shader_helper_->GetShaderModule(EShaderType::kVertexShader)},
                              {EShaderType::kFragmentShader,
                               vk_shader_helper_->GetShaderModule(EShaderType::kFragmentShader)}},
        .renderpass = vk_renderpass_helper_->GetRenderpass(),
        .vertex_input_binding_description = test_vertex_input_binding_description_,
        .vertex_input_attribute_descriptions = test_vertex_input_attributes_,
        .descriptor_set_layouts = {descriptor_set_layout_}};
    // pipeline_config.vertex_input_binding_description =
    // vertex_input_binding_description_;
    // pipeline_config.vertex_input_attribute_descriptions =
    // {vertex_input_attribute_position_, vertex_input_attribute_color_};
    vk_pipeline_helper_ = std::make_unique<VulkanPipelineHelper>(pipeline_config);
    return vk_pipeline_helper_->CreatePipeline(comm_vk_logical_device_);
}

bool VulkanSample::allocate_per_frame_command_buffer()
{
    for (int i = 0; i < engine_config_.frame_count; ++i)
    {
        if (!vk_command_buffer_helper_->AllocateCommandBuffer(
            {.command_buffer_level = vk::CommandBufferLevel::ePrimary, .command_buffer_count = 1},
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

    // Create synchronization objects per frame-in-flight
    for (int i = 0; i < engine_config_.frame_count; ++i)
    {
        if (!vk_synchronization_helper_->CreateVkSemaphore(output_frames_[i].image_available_semaphore_id))
            return false;

        // Note: Do NOT create the per-frame render_finished_semaphore here anymore.
        // We will use per-image semaphores instead.
        // if (!vk_synchronization_helper_->CreateVkSemaphore(output_frames_[i].render_finished_semaphore_id))
        //    return false;

        if (!vk_synchronization_helper_->CreateFence(output_frames_[i].fence_id))
            return false;
    }

    // Create render_finished semaphores per swapchain image
    const auto swapchain_images = comm_vk_logical_device_.getSwapchainImagesKHR(comm_vk_swapchain_);
    for (size_t i = 0; i < swapchain_images.size(); ++i)
    {
        // Use a unique ID based on the image index
        std::string id = "render_finished_semaphore_image_" + std::to_string(i);
        if (!vk_synchronization_helper_->CreateVkSemaphore(id))
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
    // REMOVE or IGNORE the per-frame render_finished_semaphore_id
    auto current_command_buffer_id            = output_frames_[frame_index_].command_buffer_id;

    // wait for the last frame to finish
    if (!vk_synchronization_helper_->WaitForFence(current_fence_id))
        return;

    // get semaphores
    auto image_available_semaphore = vk_synchronization_helper_->GetSemaphore(current_image_available_semaphore_id);
    auto in_flight_fence           = vk_synchronization_helper_->GetFence(current_fence_id);

    // acquire next image
    auto acquire_result = comm_vk_logical_device_.acquireNextImageKHR(
        comm_vk_swapchain_,
        UINT64_MAX,
        image_available_semaphore,
        VK_NULL_HANDLE);

    // ... (error checks for acquire_result) ...
    if (acquire_result.result != vk::Result::eSuccess && acquire_result.result != vk::Result::eSuboptimalKHR)
    {
        // Handle resize or error
        if (acquire_result.result == vk::Result::eErrorOutOfDateKHR) resize_request_ = true;
        return;
    }

    // [FIX] Get the semaphore associated with the ACQUIRED IMAGE INDEX
    uint32_t image_index = acquire_result.value;
    std::string render_finished_sem_id = "render_finished_semaphore_image_" + std::to_string(image_index);
    auto render_finished_semaphore = vk_synchronization_helper_->GetSemaphore(render_finished_sem_id);

    // reset fence before submitting
    if (!vk_synchronization_helper_->ResetFence(current_fence_id))
        return;

    // record command buffer
    if (!vk_command_buffer_helper_->ResetCommandBuffer(current_command_buffer_id))
        return;
    if (!record_command(image_index, current_command_buffer_id))
        return;

    // submit command buffer
    vk::CommandBufferSubmitInfo command_buffer_submit_info{
        .commandBuffer = vk_command_buffer_helper_->GetCommandBuffer(current_command_buffer_id)};

    vk::SemaphoreSubmitInfo wait_semaphore_info{.semaphore = image_available_semaphore, .value = 1};

    // Use the per-image semaphore here
    vk::SemaphoreSubmitInfo signal_semaphore_info{.semaphore = render_finished_semaphore, .value = 1};

    vk::SubmitInfo2 submit_info{
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &wait_semaphore_info,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command_buffer_submit_info,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signal_semaphore_info,
    };
    comm_vk_graphics_queue_.submit2(submit_info, in_flight_fence);

    // present the image
    vk::PresentInfoKHR present_info{.waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &render_finished_semaphore, // Use the per-image semaphore here too
                                    .swapchainCount = 1,
                                    .pSwapchains = &comm_vk_swapchain_,
                                    .pImageIndices = &acquire_result.value};

    auto res = comm_vk_graphics_queue_.presentKHR(&present_info);
    if (res == vk::Result::eErrorOutOfDateKHR || res == vk::Result::eSuboptimalKHR)
    {
        resize_request_ = true;
        return;
    }
    if (res != vk::Result::eSuccess)
    {
        Logger::LogError("Failed to present image");
        return;
    }
    // Logger::LogInfo("Succeeded in presenting image");

    // update frame index
    frame_index_ = (frame_index_ + 1) % engine_config_.frame_count;
}

void VulkanSample::resize_swapchain()
{
    // wait for the device to be idle
    comm_vk_logical_device_.waitIdle();

    // destroy old vulkan objects

    for (auto image_view : comm_vk_swapchain_context_.swapchain_image_views_)
    {
        comm_vk_logical_device_.destroyImageView(image_view, nullptr);
    }
    // Note: Don't destroy swapchain images as they are owned by the swapchain
    comm_vk_logical_device_.destroySwapchainKHR(comm_vk_swapchain_, nullptr);
    vk_frame_buffer_helper_.reset();

    // reset window size
    int width, height;
    window_->get_extent(width, height);
    engine_config_.window_config.width  = width;
    engine_config_.window_config.height = height;

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
                                                                vk::CommandBufferUsageFlagBits::eOneTimeSubmit))
        return false;

    // collect needed objects
    auto command_buffer = vk_command_buffer_helper_->GetCommandBuffer(command_buffer_id);

    // 从暂存缓冲区复制到本地缓冲区
    vk::BufferCopy buffer_copy_info{
        .srcOffset = buffer_copy_info.srcOffset = 0,
        .dstOffset = buffer_copy_info.dstOffset = 0,
        .size = buffer_copy_info.size           = test_staging_buffer_allocation_info_.size,
    };
    command_buffer.copyBuffer(test_staging_buffer_, test_local_buffer_, 1, &buffer_copy_info);

    // 设置内存屏障以确保复制完成
    vk::BufferMemoryBarrier2 buffer_memory_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexInput,
        .dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead,
        .buffer = test_local_buffer_,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };

    vk::DependencyInfo dependency_info{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &buffer_memory_barrier};
    command_buffer.pipelineBarrier2(&dependency_info);

    // begin renderpass

    vk::ClearValue clear_values_color{.color = {std::array<float, 4>{0.1F, 0.1F, 0.1F, 1.0F}}};
    vk::ClearValue clear_value_depth{.depthStencil = {.depth = 1.0F, .stencil = 0}};
    std::vector<vk::ClearValue> clear_values = {clear_values_color, clear_value_depth};

    vk::RenderPassBeginInfo renderpass_info{
        .renderPass = vk_renderpass_helper_->GetRenderpass(),
        .framebuffer = (*vk_frame_buffer_helper_->GetFramebuffers())[image_index],
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = comm_vk_swapchain_context_.swapchain_info_.extent_},
        .clearValueCount = static_cast<uint32_t>(clear_values.size()),
        .pClearValues = clear_values.data()};

    command_buffer.beginRenderPass(renderpass_info, vk::SubpassContents::eInline);

    // bind pipeline
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, vk_pipeline_helper_->GetPipeline());

    // bind descriptor set
    auto offset =
        uniform_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Frequently].offsets[uniform_buffer_id_[frame_index_]];
    auto dynamic_offset = static_cast<uint32_t>(offset);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      vk_pipeline_helper_->GetPipelineLayout(),
                                      0,
                                      1,
                                      descriptor_sets_.data(),
                                      1,
                                      &dynamic_offset);

    // dynamic state update
    vk::Viewport viewport{.x = 0.0F,
                          .y = 0.0F,
                          .width = static_cast<float>(comm_vk_swapchain_context_.swapchain_info_.extent_.width),
                          .height = static_cast<float>(comm_vk_swapchain_context_.swapchain_info_.extent_.height),
                          .minDepth = 0.0F,
                          .maxDepth = 1.0F};
    command_buffer.setViewport(0, 1, &viewport);

    vk::Rect2D scissor{.offset = {.x = 0, .y = 0}, .extent = comm_vk_swapchain_context_.swapchain_info_.extent_};
    command_buffer.setScissor(0, 1, &scissor);

    // 绑定顶点和索引缓冲区
    command_buffer.bindVertexBuffers(
        0,
        1,
        &test_local_buffer_,
        &test_local_host_batch_handle_[vra::VraBuiltInBatchIds::GPU_Only].offsets[test_vertex_buffer_id_]);
    command_buffer.bindIndexBuffer(
        test_local_buffer_,
        test_local_host_batch_handle_[vra::VraBuiltInBatchIds::GPU_Only].offsets[test_index_buffer_id_],
        vk::IndexType::eUint32);

    // 遍历每个 mesh 进行绘制
    for (const auto& mesh : mesh_list_)
    {
        // 遍历该 mesh 的所有图元进行绘制
        // if (std::strcmp(mesh.name.c_str(), "arch_stones_02") != 0) {
        //     continue;
        // }
        for (const auto& primitive : mesh.primitives)
        {
            // 绘制当前图元
            command_buffer.drawIndexed(primitive.index_count,
                                       // 使用实际的索引数量
                                       1,
                                       primitive.first_index,
                                       // 使用实际的索引偏移量
                                       0,
                                       0);
        }
    }

    // end renderpass
    command_buffer.endRenderPass();

    // end command recording
    return vk_command_buffer_helper_->EndCommandBufferRecording(command_buffer_id);
}

void VulkanSample::update_uniform_buffer(uint32_t current_frame_index)
{
    mvp_matrices_[current_frame_index].model = simple_camera_->get_matrix(interface::TransformMatrixType::kModel);
    mvp_matrices_[current_frame_index].view  = simple_camera_->get_matrix(interface::TransformMatrixType::kView);
    mvp_matrices_[current_frame_index].projection = simple_camera_->get_matrix(interface::TransformMatrixType::kProjection);
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

void VulkanSample::create_drawcall_list_buffer()
{
    vra::VraRawData vertex_buffer_data{.pData_ = vertices_.data(), .size_ = sizeof(gltf::Vertex) * vertices_.size()};
    vra::VraRawData index_buffer_data{.pData_ = indices_.data(), .size_ = sizeof(uint32_t) * indices_.size()};

    // 顶点缓冲区创建信息
    vk::BufferCreateInfo vertex_buffer_create_info{.usage = vk::BufferUsageFlagBits::eVertexBuffer |
                                                            vk::BufferUsageFlagBits::eTransferDst,
                                                   .sharingMode = vk::SharingMode::eExclusive};
    vra::VraDataDesc vertex_buffer_desc{
        vra::VraDataMemoryPattern::GPU_Only, vra::VraDataUpdateRate::RarelyOrNever, vertex_buffer_create_info};

    // 索引缓冲区创建信息
    vk::BufferCreateInfo index_buffer_create_info{.usage = vk::BufferUsageFlagBits::eIndexBuffer |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                  .sharingMode = vk::SharingMode::eExclusive};
    vra::VraDataDesc index_buffer_desc{
        vra::VraDataMemoryPattern::GPU_Only, vra::VraDataUpdateRate::RarelyOrNever, index_buffer_create_info};

    // 暂存缓冲区创建信息
    vk::BufferCreateInfo staging_buffer_create_info{.usage = vk::BufferUsageFlagBits::eTransferSrc,
                                                    .sharingMode = vk::SharingMode::eExclusive};
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
                    reinterpret_cast<VkBuffer*>(&test_local_buffer_),
                    &test_local_buffer_allocation_,
                    &test_local_buffer_allocation_info_);

    // 创建暂存缓冲区
    auto test_host_buffer_create_info =
        test_local_host_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Rarely].data_desc.GetBufferCreateInfo();
    VmaAllocationCreateInfo staging_allocation_create_info{};
    staging_allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    staging_allocation_create_info.flags = vra_data_batcher_->GetSuggestVmaMemoryFlags(
        vra::VraDataMemoryPattern::CPU_GPU,
        vra::VraDataUpdateRate::RarelyOrNever);
    vmaCreateBuffer(vma_allocator_,
                    &test_host_buffer_create_info,
                    &staging_allocation_create_info,
                    reinterpret_cast<VkBuffer*>(&test_staging_buffer_),
                    &test_staging_buffer_allocation_,
                    &test_staging_buffer_allocation_info_);

    // 复制数据到暂存缓冲区
    auto consolidate_data = test_local_host_batch_handle_[vra::VraBuiltInBatchIds::CPU_GPU_Rarely].consolidated_data;
    void* data            = nullptr;
    vmaInvalidateAllocation(vma_allocator_, test_staging_buffer_allocation_, 0, VK_WHOLE_SIZE);
    vmaMapMemory(vma_allocator_, test_staging_buffer_allocation_, &data);
    memcpy(data, consolidate_data.data(), consolidate_data.size());
    vmaUnmapMemory(vma_allocator_, test_staging_buffer_allocation_);
    vmaFlushAllocation(vma_allocator_, test_staging_buffer_allocation_, 0, VK_WHOLE_SIZE);

    // 设置顶点输入绑定描述
    test_vertex_input_binding_description_.binding   = 0;
    test_vertex_input_binding_description_.stride    = sizeof(gltf::Vertex);
    test_vertex_input_binding_description_.inputRate = vk::VertexInputRate::eVertex;

    // 设置顶点属性描述
    test_vertex_input_attributes_.clear();

    // 使用更安全的偏移量计算，确保 offsetof 计算正确
    // position
    test_vertex_input_attributes_.emplace_back(
        vk::VertexInputAttributeDescription{.location = 0,
                                            .binding = 0,
                                            .format = vk::Format::eR32G32B32Sfloat,
                                            .offset = offsetof(gltf::Vertex, position)});
    // color
    test_vertex_input_attributes_.emplace_back(
        vk::VertexInputAttributeDescription{.location = 1,
                                            .binding = 0,
                                            .format = vk::Format::eR32G32B32A32Sfloat,
                                            .offset = offsetof(gltf::Vertex, color)});
    // normal
    test_vertex_input_attributes_.emplace_back(vk::VertexInputAttributeDescription{
        .location = 2, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(gltf::Vertex, normal)});
    // tangent
    test_vertex_input_attributes_.emplace_back(
        vk::VertexInputAttributeDescription{.location = 3,
                                            .binding = 0,
                                            .format = vk::Format::eR32G32B32A32Sfloat,
                                            .offset = offsetof(gltf::Vertex, tangent)});
    // uv0
    test_vertex_input_attributes_.emplace_back(vk::VertexInputAttributeDescription{
        .location = 4, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(gltf::Vertex, uv0)});
    // uv1
    test_vertex_input_attributes_.emplace_back(vk::VertexInputAttributeDescription{
        .location = 5, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(gltf::Vertex, uv1)});
}
