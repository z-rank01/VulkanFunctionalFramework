#pragma once

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <cassert>
#include <stdexcept>
#include <functional> // Required for std::function
#include <map>        // Required for std::map
#include <memory>
#include <string> // Required for std::string
#include <unordered_map>
#include <vector>

namespace vra
{
// - Resource ID is used to identify the resource
// - actual type: uint64_t
using ResourceId = uint64_t;

// - Batch ID is used to identify the batch
// - actual type: char*
using BatchId = std::string;
class VraBuiltInBatchIds
{
public:
    static const BatchId GPU_Only;
    static const BatchId CPU_GPU_Rarely;
    static const BatchId CPU_GPU_Frequently;
    static const BatchId GPU_CPU_Rarely;
    static const BatchId GPU_CPU_Frequently;
};

// --- Forward Declarations ---
struct VraDataDesc;
struct VraRawData;

// --- Data Memory Pattern ---
enum class VraDataMemoryPattern
{
    // Default
    Default,

    // - Device Local, no CPU access
    GPU_Only,

    // - CPU and GPU sequential access, e.g. UBO update
    CPU_GPU,

    // - CPU and GPU random access;
    // - Always has Host-Cached
    GPU_CPU,

    // - CPU and GPU access with unified memory architecture
    SOC,

    // - ring buffer mode, e.g. indirect draw command;
    // - Always has Host-Cached
    Stream_Ring,
};

// --- Data Update Rate ---
enum class VraDataUpdateRate
{
    // Default
    Default,

    // - Update frequently, e.g. UBO update;
    // - Always has Host-Coherent
    Frequent,

    // - Update rarely or never, e.g. static and uniform data;
    // - Require to flush memory before using if there is a transfer
    RarelyOrNever
};

struct VraRawData
{
    const void* pData_ = nullptr;
    size_t size_       = 0;
};

struct VraDataDesc
{
    // --- Constructor ---

    VraDataDesc() = default;
    VraDataDesc(VraDataMemoryPattern pattern,
                VraDataUpdateRate update_rate,
                VkBufferCreateInfo buffer_create_info = VkBufferCreateInfo{})
        : data_pattern_(pattern), data_update_rate_(update_rate), buffer_create_info_(buffer_create_info)
    {
    }

    // --- Getter ---

    VraDataMemoryPattern GetMemoryPattern() const { return data_pattern_; }
    VraDataUpdateRate GetUpdateRate() const { return data_update_rate_; }
    const VkBufferCreateInfo& GetBufferCreateInfo() const { return buffer_create_info_; }
    VkBufferCreateInfo& GetBufferCreateInfo() { return buffer_create_info_; }

private:
    // --- main members ---

    VraDataMemoryPattern data_pattern_;
    VraDataUpdateRate data_update_rate_;
    VkBufferCreateInfo buffer_create_info_;
};

class ResourceIDGenerator
{
public:
    ResourceIDGenerator()
    {
        // Initialize the atomic counter via unique_ptr
        resource_id_counter_ = std::make_unique<std::atomic<ResourceId>>(0);
    }
    ~ResourceIDGenerator() = default;

    /// @brief generate sequential resource id from a atomic counter
    /// @return resource id type: uint64_t
    ResourceId GenerateID()
    {
        // Access the atomic counter through the unique_ptr
        return resource_id_counter_->fetch_add(1);
    }

    /// @brief reset resource id counter
    void Reset() { resource_id_counter_->store(0); }

private:
    // Use unique_ptr to store the atomic counter
    std::unique_ptr<std::atomic<ResourceId>> resource_id_counter_;
};

class VraDataBatcher
{
public:
    /// @brief VraBatchHandle is a struct that contains a vector of uint8_t, an unordered_map of ResourceId and size_t,
    /// and a VraDataDesc.
    /// @brief VraBatchHandle is the screeshot result of batching current collected data, which means that the batching
    /// result will not be stored internally.
    struct VraBatchHandle
    {
        bool initialized = false;
        std::vector<uint8_t> consolidated_data;
        std::unordered_map<ResourceId, size_t> offsets;
        VraDataDesc data_desc;

        void Clear()
        {
            initialized = false;
            consolidated_data.clear();
            offsets.clear();
            data_desc = VraDataDesc{};
        }
    };

    VraDataBatcher() = delete;
    VraDataBatcher(VkPhysicalDevice physical_device) : physical_device_handle_(physical_device)
    {
        vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties_);
        RegisterDefaultBatcher();
    }
    ~VraDataBatcher();

    // --- Data Process ---

    /// @brief Collects buffer data description and raw data pointer.
    /// @param data_desc Description of the buffer data (usage, memory pattern, etc.).
    /// @param data Raw data pointer and size. The pointer pData_ must remain valid until Execute is called.
    /// @return True if collected successfully, false if ID already exists or max count reached.
    bool Collect(VraDataDesc data_desc, VraRawData data, ResourceId& id);

    /// @brief processes all collected buffer data, grouping them by memory pattern
    /// @return a map of batch id and batch handle
    /// @note 1.this function is not thread safe.
    /// @note 2.batch result as a screenshot will be flushed and cleared after batching.
    /// @note 3.collected data still remains, no need to collect again.
    std::map<BatchId, VraBatchHandle> Batch();

    /// @brief clear all collected data and grouped data
    void Clear();

    // --- Batcher Registration ---

    /// @brief register buffer batch strategy
    /// @param batch_id register batch id
    /// @param predicate predicate before grouping
    /// @param batch_method action when grouping
    void RegisterBatcher(
        const BatchId batch_id,
        std::function<bool(const VraDataDesc&)> predicate,
        std::function<void(ResourceId id, VraBatchHandle& batch, const VraDataDesc& data_desc, const VraRawData& data)>
            batch_method);

    // --- Memory Flags Suggestion ---

    /// @brief get suggest memory flags for vulkan
    /// @param batch_id batch id
    /// @return suggest memory flags
    VkMemoryPropertyFlags GetSuggestMemoryFlags(VraDataMemoryPattern data_pattern, VraDataUpdateRate data_update_rate);

    /// @brief get suggest vma memory flags for vulkan
    /// @param batch_id batch id
    /// @return suggest vma memory flags
    VmaAllocationCreateFlags GetSuggestVmaMemoryFlags(VraDataMemoryPattern data_pattern,
                                                      VraDataUpdateRate data_update_rate);

private:
    /// @brief VraBatcher is a struct that contains a batch id, a predicate, and a batch method.
    /// @brief VraBatcher is used to batch buffer data by memory pattern and update rate according to the predicate with
    /// the batch method
    struct VraBatcher
    {
        const BatchId batch_id;
        std::function<bool(const VraDataDesc&)> predicate;
        std::function<void(ResourceId id, VraBatchHandle& batch, const VraDataDesc& data_desc, const VraRawData& data)>
            batch_method;
        VraBatchHandle batch_handle; // Each strategy instance owns its data
    };

    /// @brief VraDataHandle is a struct that contains a resource id, a data description, and a raw data.
    /// @brief VraDataHandle is used to store data description and raw data pointer, as data input.
    struct VraDataHandle
    {
        ResourceId id;
        VraDataDesc data_desc;
        VraRawData data;
    };

    // --- Vulkan Native Objects Cache ---

    VkPhysicalDevice physical_device_handle_;
    VkPhysicalDeviceProperties physical_device_properties_;

    // --- Buffer specific storage ---

    static constexpr size_t MAX_BUFFER_COUNT = 4096;
    std::vector<VraDataHandle> data_handles_;

    // --- Resource Id ---

    ResourceIDGenerator resource_id_generator_;

    // --- Registered Group Strategies ---

    std::map<BatchId, VraBatcher> registered_batchers_;

    /// @brief clear grouped buffer data
    void ClearBatch();

    /// @brief register default grouping strategies
    void RegisterDefaultBatcher();
};

class VraDescriptorAllocator
{
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDevice device_                  = VK_NULL_HANDLE;

    // constructor
    VraDescriptorAllocator(VkDevice device) : device_(device) { }

    // --- Helper methods ---

    // --- Descriptor methods ---
    VkDescriptorSet AllocateDescriptorSet(VkDescriptorSetLayout layout)
    {
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool              = descriptor_pool_;
        alloc_info.descriptorSetCount          = 1;
        alloc_info.pSetLayouts                 = &layout;

        VkDescriptorSet descriptor_set;
        VkResult result = vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate descriptor set!");
        }

        return descriptor_set;
    }
};

class VRA
{
public:
    VRA();
    ~VRA();
};
} // namespace vra