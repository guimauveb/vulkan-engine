use super::QueueFamilyIndices;
use anyhow::Result;
use vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0, Handle, HasBuilder, Instance};

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

pub unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(instance, physical_device, surface)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}
