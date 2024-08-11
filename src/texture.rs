use {
    super::{
        buffer::{begin_single_time_commands, end_single_time_commands, BufferAllocation},
        image::{copy_buffer_to_image, create_image, create_image_view, transition_image_layout},
        EngineData,
    },
    anyhow::{anyhow, Result},
    std::{fs::File, ptr::copy_nonoverlapping},
    vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0, HasBuilder, Instance, InstanceV1_0},
};

// It should be noted that it is uncommon in practice to generate the mipmap levels at runtime anyway.
// Usually they are pregenerated and stored in the texture file alongside the base level to improve loading speed.

// TODO: Implementing resizing in software and loading multiple levels from a file is left as an exercise to the reader.
unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &EngineData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!(
            "Texture image format does not support linear blitting"
        ));
    }
    let command_buffer = begin_single_time_commands(device, data.command_pool)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);
    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);
    let (mut mip_width, mut mip_height) = (width, height);

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);
        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }
        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(
        device,
        data.graphics_queue,
        data.command_pool,
        command_buffer,
    )?;

    Ok(())
}

// All of the helper functions that submit commands so far have been set up to execute synchronously by waiting for the queue to become idle.
// For practical applications it is recommended to combine these operations in a single command buffer and execute them asynchronously for higher throughput,
// especially the transitions and copy in the create_texture_image function.

// TODO: Try to experiment with this by creating a setup_command_buffer that the helper functions record commands into,
// and add a flush_setup_commands to execute the commands that have been recorded so far.
// It's best to do this after the texture mapping works to check if the texture resources are still set up correctly.
pub unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut EngineData,
) -> Result<()> {
    let image = File::open("resources/viking_room.png")?;
    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let size = reader.info().raw_bytes();
    let mut pixels = vec![0; size];
    reader.next_frame(&mut pixels)?;

    let (width, height) = reader.info().size();
    if width != 1024 || height != 1024 || reader.info().color_type != png::ColorType::Rgba {
        return Err(anyhow!("Invalid texture image"));
    }

    let size = size as vk::DeviceSize;
    // Create staging buffer
    let staging_buffer = BufferAllocation::new(
        instance,
        device,
        data.physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy to staging buffer
    let memory = device.map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
    copy_nonoverlapping(pixels.as_ptr(), memory.cast(), pixels.len());
    device.unmap_memory(staging_buffer.memory);

    data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data.physical_device,
        width,
        height,
        data.mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    // Transition + copy
    transition_image_layout(
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
    )?;
    copy_buffer_to_image(
        device,
        data,
        staging_buffer.buffer,
        data.texture_image,
        width,
        height,
    )?;

    staging_buffer.destroy(device);

    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )?;

    Ok(())
}

pub unsafe fn create_texture_image_view(device: &Device, data: &mut EngineData) -> Result<()> {
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )?;
    Ok(())
}

pub unsafe fn create_texture_sampler(device: &Device, data: &mut EngineData) -> Result<()> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0) // Optional
        .max_lod(data.mip_levels as f32)
        .mip_lod_bias(0.0); // Optional
    data.texture_sampler = device.create_sampler(&info, None)?;

    Ok(())
}
