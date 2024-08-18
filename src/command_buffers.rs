use super::{EngineData, QueueFamilyIndices};
use anyhow::Result;
use vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0, HasBuilder, Instance};

pub unsafe fn create_command_buffers(device: &Device, data: &mut EngineData) -> Result<()> {
    for image_index in 0..data.swapchain_images.len() {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.framebuffers_command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
        data.command_buffers.push(command_buffer);
    }

    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

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
