use {
    super::{vertex::Vertex, EngineData, Mat4},
    anyhow::{anyhow, Result},
    std::{mem::size_of, ptr::copy_nonoverlapping},
    vulkanalia::prelude::v1_0::{
        vk, Device, DeviceV1_0, Handle, HasBuilder, Instance, InstanceV1_0,
    },
};

pub unsafe fn begin_single_time_commands(
    device: &Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);
    let command_buffer = device.allocate_command_buffers(&info)?[0];
    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

pub unsafe fn end_single_time_commands(
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(graphics_queue)?;
    device.free_command_buffers(command_pool, &[command_buffer]);

    Ok(())
}

pub unsafe fn get_memory_type_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(physical_device);
    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
}

pub unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = device.create_buffer(&buffer_info, None)?;
    let requirements = device.get_buffer_memory_requirements(buffer);
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            physical_device,
            properties,
            requirements,
        )?);
    let buffer_memory = device.allocate_memory(&memory_info, None)?;
    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

pub unsafe fn copy_buffer(
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);
    end_single_time_commands(device, graphics_queue, command_pool, command_buffer)?;

    Ok(())
}

// TODO - Check if we can have a method for any "staged" buffer copy (source -> staging -> destination)
pub unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    vertices: &[Vertex],
    size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy to staging buffer
    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    copy_nonoverlapping(vertices.as_ptr(), memory.cast(), vertices.len());
    device.unmap_memory(staging_buffer_memory);

    // Create vertex buffer
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        // XXX - Split buffers that will be used later (and probably with data copied using memcpy) and buffers used to store data passed as arg here (using device local memory)
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy to vertex buffer
    copy_buffer(
        device,
        graphics_queue,
        command_pool,
        staging_buffer,
        vertex_buffer,
        size,
    )?;

    // Cleanup
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((vertex_buffer, vertex_buffer_memory))
}

// TODO - Check if we can have a method for any "staged" buffer copy (source -> staging -> destination)
pub unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    indices: &[u32],
    size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    copy_nonoverlapping(indices.as_ptr(), memory.cast(), indices.len());
    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        // TODO - Split buffers that will be used later (and probably with data copied using memcpy) and buffers used to store data passed as arg here (using device local memory)
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    copy_buffer(
        device,
        graphics_queue,
        command_pool,
        staging_buffer,
        index_buffer,
        size,
    )?;

    // Cleanup
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((index_buffer, index_buffer_memory))
}

// Vulkan expects data to be aligned in memory in a specific way.
// https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/chap14.html#interfaces-resources-layout
// Here the structure is properly aligned.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub view: Mat4,
    pub proj: Mat4,
}

pub unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut EngineData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data.physical_device,
            size_of::<UniformBufferObject>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}
