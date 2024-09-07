use super::{vertex::Vertex, Mat4};
use crate::command_buffers::{begin_single_time_commands, end_single_time_commands};
use anyhow::{anyhow, Result};
use std::ptr::copy_nonoverlapping;
use vulkanalia::prelude::v1_3::{
    vk, Device, DeviceV1_0, DeviceV1_2, HasBuilder, Instance, InstanceV1_0,
};

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

#[derive(Default, Debug)]
pub struct BufferAllocation {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    address: Option<vk::DeviceAddress>,
}

impl BufferAllocation {
    /// Constructor
    pub unsafe fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&buffer_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let mut memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type_index(
                instance,
                physical_device,
                properties,
                requirements,
            )?);

        let shader_device_address = usage & vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            == vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let memory = if shader_device_address {
            let mut flags = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            memory_info = memory_info.push_next(&mut flags);
            device.allocate_memory(&memory_info, None)?
        } else {
            device.allocate_memory(&memory_info, None)?
        };

        device.bind_buffer_memory(buffer, memory, 0)?;

        let address = if shader_device_address {
            let info = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
            Some(device.get_buffer_device_address(&info))
        } else {
            None
        };

        Ok(Self {
            buffer,
            memory,
            address,
        })
    }

    /// Return the buffer device address
    ///
    /// # Panics
    /// Panics if the buffer was not allocated with [vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS]
    pub fn address(&self) -> vk::DeviceAddress {
        self.address
            .expect("Buffer allocated without `SHADER_DEVICE_ADDRESS` flag")
    }

    /// Destroy the buffer and release the associated memory.
    #[inline]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None)
    }
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

// TODO: To do on a dedicated thread
pub unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    vertices: &[Vertex],
    size: vk::DeviceSize,
) -> Result<BufferAllocation> {
    let staging_buffer = BufferAllocation::new(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy to staging buffer
    let memory = device.map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
    copy_nonoverlapping(vertices.as_ptr(), memory.cast(), vertices.len());
    device.unmap_memory(staging_buffer.memory);

    // Create vertex buffer
    let vertex_buffer = BufferAllocation::new(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        // TODO: Split buffers that will be used later (and probably with data copied using memcpy) and buffers used to store data passed as arg here (using device local memory)
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy to vertex buffer
    copy_buffer(
        device,
        graphics_queue,
        command_pool,
        staging_buffer.buffer,
        vertex_buffer.buffer,
        size,
    )?;

    // Cleanup
    staging_buffer.destroy(device);

    Ok(vertex_buffer)
}

// TODO: To do on a dedicated thread
pub unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    indices: &[u32],
    size: vk::DeviceSize,
) -> Result<BufferAllocation> {
    let staging_buffer = BufferAllocation::new(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
    copy_nonoverlapping(indices.as_ptr(), memory.cast(), indices.len());
    device.unmap_memory(staging_buffer.memory);

    let index_buffer = BufferAllocation::new(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        // TODO: Split buffers that will be used later (and probably with data copied using memcpy) and buffers used to store data passed as arg here (using device local memory)
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    copy_buffer(
        device,
        graphics_queue,
        command_pool,
        staging_buffer.buffer,
        index_buffer.buffer,
        size,
    )?;

    // Cleanup
    staging_buffer.destroy(device);

    Ok(index_buffer)
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
