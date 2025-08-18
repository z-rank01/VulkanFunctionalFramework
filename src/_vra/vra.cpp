#define VMA_IMPLEMENTATION
#include "vra.h"

#include <iostream>  // Required for std::cerr
#include <vma/vk_mem_alloc.h> // Include VMA header AFTER the implementation define


namespace vra
{
const BatchId VraBuiltInBatchIds::GPU_Only           = "GPU_Only";
const BatchId VraBuiltInBatchIds::CPU_GPU_Rarely     = "CPU_GPU_Rarely";
const BatchId VraBuiltInBatchIds::CPU_GPU_Frequently = "CPU_GPU_Frequently";
const BatchId VraBuiltInBatchIds::GPU_CPU_Rarely     = "GPU_CPU_Rarely";
const BatchId VraBuiltInBatchIds::GPU_CPU_Frequently = "GPU_CPU_Frequently";

// -------------------------------------
// --- Data Collector Implementation ---
// -------------------------------------

VraDataBatcher::~VraDataBatcher()
{
    Clear();
}

void VraDataBatcher::RegisterBatcher(
    const BatchId batch_id,
    std::function<bool(const VraDataDesc&)> predicate,
    std::function<void(ResourceId, VraBatchHandle&, const VraDataDesc&, const VraRawData&)> batch_action)
{
    if (registered_batchers_.count(batch_id))
    {
        std::cerr << "Batch id " << batch_id << " already registered" << std::endl;
        return;
    }
    registered_batchers_.emplace(batch_id, VraBatcher{batch_id, std::move(predicate), std::move(batch_action)});
}

void VraDataBatcher::RegisterDefaultBatcher()
{
    RegisterBatcher(
        vra::VraBuiltInBatchIds::GPU_Only,
        [](const VraDataDesc& desc)
        {
            return desc.GetMemoryPattern() == VraDataMemoryPattern::GPU_Only &&
                   desc.GetUpdateRate() == VraDataUpdateRate::RarelyOrNever;
        },
        [](ResourceId id, VraBatchHandle& batch, const VraDataDesc& desc, const VraRawData& raw_data)
        {
            const auto& new_buffer_ci = desc.GetBufferCreateInfo();
            auto& batch_ci_ref        = batch.data_desc.GetBufferCreateInfo();

            if (!batch.initialized)
            {
                batch.data_desc   = desc; // Initialize the batch's entire VraDataDesc
                batch.initialized = true;
            }
            else
            {
                if (new_buffer_ci.usage == 0 || new_buffer_ci.sharingMode != batch_ci_ref.sharingMode)
                {
                    return; // Incompatible
                }

                batch_ci_ref.usage |= new_buffer_ci.usage;
                batch_ci_ref.flags |= new_buffer_ci.flags;
                if (new_buffer_ci.queueFamilyIndexCount > batch_ci_ref.queueFamilyIndexCount &&
                    new_buffer_ci.sharingMode == VK_SHARING_MODE_CONCURRENT)
                {
                    batch_ci_ref.queueFamilyIndexCount = new_buffer_ci.queueFamilyIndexCount;
                    batch_ci_ref.pQueueFamilyIndices   = new_buffer_ci.pQueueFamilyIndices;
                }
            }

            if (raw_data.pData_ != nullptr && raw_data.size_ > 0)
            {
                batch.offsets[id]       = batch.consolidated_data.size();
                const uint8_t* data_ptr = static_cast<const uint8_t*>(raw_data.pData_);
                batch.consolidated_data.insert(batch.consolidated_data.end(), data_ptr, data_ptr + raw_data.size_);
            }
            else
            {
                batch.offsets[id] = batch.consolidated_data.size();
            }

            batch_ci_ref.size = batch.consolidated_data.size();
        });

    // Dynamic Sequential Batch
    RegisterBatcher(
        vra::VraBuiltInBatchIds::CPU_GPU_Rarely,
        [](const VraDataDesc& desc)
        {
            return desc.GetMemoryPattern() == VraDataMemoryPattern::CPU_GPU &&
                   desc.GetUpdateRate() == VraDataUpdateRate::RarelyOrNever;
        },
        [&physical_device_properties = this->physical_device_properties_](
            ResourceId id, VraBatchHandle& batch, const VraDataDesc& desc, const VraRawData& raw_data)
        {
            const auto& new_buffer_ci = desc.GetBufferCreateInfo();
            // Get a modifiable reference to the batch's VkBufferCreateInfo
            auto& batch_ci_ref = batch.data_desc.GetBufferCreateInfo();

            if (!batch.initialized)
            {
                batch.data_desc   = desc; // Initialize the batch's entire VraDataDesc
                batch.initialized = true;
                // batch_ci_ref now refers to the CI within the newly set batch.data_desc
            }
            else if (new_buffer_ci.usage == 0 || new_buffer_ci.sharingMode != batch_ci_ref.sharingMode)
            {
                return; // Incompatible
            }
            else
            {
                // Merge into batch_ci_ref
                batch_ci_ref.usage |= new_buffer_ci.usage;
                batch_ci_ref.flags |= new_buffer_ci.flags;
                if (new_buffer_ci.queueFamilyIndexCount > batch_ci_ref.queueFamilyIndexCount &&
                    new_buffer_ci.sharingMode == VK_SHARING_MODE_CONCURRENT)
                {
                    batch_ci_ref.queueFamilyIndexCount = new_buffer_ci.queueFamilyIndexCount;
                    batch_ci_ref.pQueueFamilyIndices   = new_buffer_ci.pQueueFamilyIndices;
                }
            }

            // Combine data with alignment
            size_t base_offset_for_item    = batch.consolidated_data.size();
            size_t aligned_offset_for_item = base_offset_for_item;

            // Apply alignment if the batch's buffer usage suggests it (e.g., UBO, SSBO)
            if ((batch_ci_ref.usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) != 0)
            {
                size_t alignment_requirement = physical_device_properties.limits.minUniformBufferOffsetAlignment;
                // TODO: Consider other alignment requirements if necessary, e.g., minStorageBufferOffsetAlignment
                if (alignment_requirement > 0)
                {
                    aligned_offset_for_item =
                        (base_offset_for_item + alignment_requirement - 1) & ~(alignment_requirement - 1);
                }
            }

            size_t padding_needed = aligned_offset_for_item - base_offset_for_item;
            if (padding_needed > 0)
            {
                batch.consolidated_data.insert(
                    batch.consolidated_data.end(), padding_needed, (uint8_t)0); // Pad with zeros
            }

            batch.offsets[id] = aligned_offset_for_item; // Store the correctly aligned offset

            if (raw_data.pData_ != nullptr && raw_data.size_ > 0)
            {
                const uint8_t* data_ptr = static_cast<const uint8_t*>(raw_data.pData_);
                batch.consolidated_data.insert(batch.consolidated_data.end(), data_ptr, data_ptr + raw_data.size_);
            }

            batch_ci_ref.size = batch.consolidated_data.size();
        });

    // Dynamic Sequential Batch (Frequently)
    RegisterBatcher(
        vra::VraBuiltInBatchIds::CPU_GPU_Frequently,
        [](const VraDataDesc& desc)
        {
            return desc.GetMemoryPattern() == VraDataMemoryPattern::CPU_GPU &&
                   desc.GetUpdateRate() == VraDataUpdateRate::Frequent;
        },
        [&physical_device_properties = this->physical_device_properties_](
            ResourceId id, VraBatchHandle& batch, const VraDataDesc& desc, const VraRawData& raw_data)
        {
            const auto& new_buffer_ci = desc.GetBufferCreateInfo();
            // Get a modifiable reference to the batch's VkBufferCreateInfo
            auto& batch_ci_ref = batch.data_desc.GetBufferCreateInfo();

            if (!batch.initialized)
            {
                batch.data_desc   = desc; // Initialize the batch's entire VraDataDesc
                batch.initialized = true;
                // batch_ci_ref now refers to the CI within the newly set batch.data_desc
            }
            else if (new_buffer_ci.usage == 0 || new_buffer_ci.sharingMode != batch_ci_ref.sharingMode)
            {
                return; // Incompatible
            }
            else
            {
                // Merge into batch_ci_ref
                batch_ci_ref.usage |= new_buffer_ci.usage;
                batch_ci_ref.flags |= new_buffer_ci.flags;
                if (new_buffer_ci.queueFamilyIndexCount > batch_ci_ref.queueFamilyIndexCount &&
                    new_buffer_ci.sharingMode == VK_SHARING_MODE_CONCURRENT)
                {
                    batch_ci_ref.queueFamilyIndexCount = new_buffer_ci.queueFamilyIndexCount;
                    batch_ci_ref.pQueueFamilyIndices   = new_buffer_ci.pQueueFamilyIndices;
                }
            }

            // Combine data with alignment
            size_t base_offset_for_item    = batch.consolidated_data.size();
            size_t aligned_offset_for_item = base_offset_for_item;

            // Apply alignment if the batch's buffer usage suggests it (e.g., UBO, SSBO)
            if ((batch_ci_ref.usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) != 0)
            {
                size_t alignment_requirement = physical_device_properties.limits.minUniformBufferOffsetAlignment;
                // TODO: Consider other alignment requirements if necessary, e.g., minStorageBufferOffsetAlignment
                if (alignment_requirement > 0)
                {
                    aligned_offset_for_item =
                        (base_offset_for_item + alignment_requirement - 1) & ~(alignment_requirement - 1);
                }
            }

            size_t padding_needed = aligned_offset_for_item - base_offset_for_item;
            if (padding_needed > 0)
            {
                batch.consolidated_data.insert(
                    batch.consolidated_data.end(), padding_needed, (uint8_t)0); // Pad with zeros
            }

            batch.offsets[id] = aligned_offset_for_item; // Store the correctly aligned offset

            if (raw_data.pData_ != nullptr && raw_data.size_ > 0)
            {
                const uint8_t* data_ptr = static_cast<const uint8_t*>(raw_data.pData_);
                batch.consolidated_data.insert(batch.consolidated_data.end(), data_ptr, data_ptr + raw_data.size_);
            }

            batch_ci_ref.size = batch.consolidated_data.size();
        });
}

bool VraDataBatcher::Collect(VraDataDesc desc, VraRawData data, ResourceId& id)
{
    // check buffer conditions
    if (desc.GetBufferCreateInfo().usage == 0 || data.pData_ == nullptr || data.size_ == 0)
    {
        return false;
    }

    // store buffer data
    id = resource_id_generator_.GenerateID();
    VraDataHandle data_handle;
    data_handle.id        = id;
    data_handle.data_desc = desc;
    data_handle.data      = data;
    data_handles_.push_back(data_handle);

    return true;
}

std::map<BatchId, VraDataBatcher::VraBatchHandle> VraDataBatcher::Batch()
{
    ClearBatch();

    // Optional: Estimate sizes and reserve capacity
    for (const auto& data_handle : data_handles_)
    {
        const VraRawData& raw_data = data_handle.data;
        const VraDataDesc& desc    = data_handle.data_desc;

        for (auto& p : registered_batchers_)
        {
            auto& batcher = p.second;
            if (batcher.predicate(desc))
            {
                // Basic estimation, doesn't account for padding in dynamic batches yet.
                // For a more accurate estimation, the predicate or a dedicated estimation function
                // in the batcher would be needed if padding is significant.
                batcher.batch_handle.consolidated_data.reserve(raw_data.size_);
                break;
            }
        }
    }
    for (auto& p : registered_batchers_)
    {
        auto& batcher = p.second;
        if (batcher.batch_handle.consolidated_data.size() > 0)
        { // Only reserve if there's an estimate
            batcher.batch_handle.consolidated_data.reserve(batcher.batch_handle.consolidated_data.size());
        }
    }

    // Batch data
    for (const auto& data_handle : data_handles_)
    {
        ResourceId id              = data_handle.id;
        const VraRawData& raw_data = data_handle.data;
        const VraDataDesc& desc    = data_handle.data_desc;

        for (auto& p : registered_batchers_)
        {
            auto& batcher = p.second;
            if (batcher.predicate(desc))
            {
                batcher.batch_method(id, batcher.batch_handle, desc, raw_data);
                break; // Assume buffer belongs to only one batch based on predicate
            }
        }
    }

    // Shrink to fit for non-dynamic batches (as per original logic)
    for (auto& p : registered_batchers_)
    {
        auto& batcher              = p.second;
        const auto& memory_pattern = batcher.batch_handle.data_desc.GetMemoryPattern();
        if (memory_pattern == vra::VraDataMemoryPattern::GPU_Only)
        {
            batcher.batch_handle.consolidated_data.shrink_to_fit();
        }
    }

    std::map<BatchId, VraDataBatcher::VraBatchHandle> batch_handles_;
    for (const auto& p : registered_batchers_)
    {
        auto& batcher                    = p.second;
        batch_handles_[batcher.batch_id] = batcher.batch_handle;
    }
    return batch_handles_;
}

void VraDataBatcher::Clear()
{
    data_handles_.clear();
    ClearBatch();
}

VkMemoryPropertyFlags VraDataBatcher::GetSuggestMemoryFlags(VraDataMemoryPattern data_pattern,
                                                            VraDataUpdateRate data_update_rate)
{
    switch (data_pattern)
    {
    case VraDataMemoryPattern::GPU_Only:
        return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    case VraDataMemoryPattern::CPU_GPU:
        if (data_update_rate == VraDataUpdateRate::Frequent)
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    case VraDataMemoryPattern::GPU_CPU:
        if (data_update_rate == VraDataUpdateRate::Frequent)
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    case VraDataMemoryPattern::SOC:
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
               VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    case VraDataMemoryPattern::Stream_Ring:
        if (data_update_rate == VraDataUpdateRate::Frequent)
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    default:
        return VkMemoryPropertyFlagBits();
    }
}

VmaAllocationCreateFlags VraDataBatcher::GetSuggestVmaMemoryFlags(VraDataMemoryPattern data_pattern,
                                                                  VraDataUpdateRate data_update_rate)
{
    switch (data_pattern)
    {
    case VraDataMemoryPattern::GPU_Only:
        return VmaAllocationCreateFlags();

    case VraDataMemoryPattern::CPU_GPU:
        if (data_update_rate == VraDataUpdateRate::Frequent)
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    case VraDataMemoryPattern::GPU_CPU:
        if (data_update_rate == VraDataUpdateRate::Frequent)
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    case VraDataMemoryPattern::SOC:
        return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;

    case VraDataMemoryPattern::Stream_Ring:
        if (data_update_rate == VraDataUpdateRate::Frequent)
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    default:
        return VmaAllocationCreateFlags();
    }
}

void VraDataBatcher::ClearBatch()
{
    for (auto& p : registered_batchers_)
    {
        auto& batcher = p.second;
        batcher.batch_handle.Clear();
    }
}

// -------------------------------------
// --- VRA Implementation ---
// -------------------------------------

VRA::VRA() { }

VRA::~VRA() { }
} // namespace vra
