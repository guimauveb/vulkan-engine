use {
    super::{buffer::UniformBufferObject, EngineData},
    anyhow::Result,
    std::mem::size_of,
    vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0, HasBuilder},
};

pub unsafe fn create_descriptor_sets(device: &Device, data: &mut EngineData) -> Result<()> {
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);
    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i].buffer())
            .offset(0)
            // Could use vk::WHOLE_SIZE here
            .range(size_of::<UniformBufferObject>() as vk::DeviceSize);
        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);
        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);
        let image_info = &[info];
        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info);

        device.update_descriptor_sets(&[ubo_write, sampler_write], &[] as &[vk::CopyDescriptorSet]);
    }

    Ok(())
}

pub unsafe fn create_descriptor_pool(device: &Device, data: &mut EngineData) -> Result<()> {
    let image_count = data.swapchain_images.len() as u32;
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(image_count);
    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(image_count);
    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(image_count);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}
