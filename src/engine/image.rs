use super::{commands::ImmediateSubmit, memory::AllocatedImage, VulkanInterface, ALLOCATOR};
use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;
use vulkanalia::{
    prelude::v1_3::{vk, Device, DeviceV1_0, HasBuilder},
    vk::DeviceV1_3,
};

/// RGB to RGBA compute shader push constants
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RgbToRgbaPushConstants {
    pub image_width: u32,
    pub image_height: u32,
    pub in_rgb: vk::DeviceAddress,
}

impl RgbToRgbaPushConstants {
    #[inline]
    pub fn new(image_width: u32, image_height: u32, in_rgb: vk::DeviceAddress) -> Self {
        Self {
            image_width,
            image_height,
            in_rgb,
        }
    }
}

/// Write pixels into an [`AllocatedImage`]
pub fn write_pixels_to_image<T>(
    interface: &VulkanInterface,
    immediate_submit: &ImmediateSubmit,
    pixels: *const T,
    allocated_image: &AllocatedImage,
    channels: u32,
) -> Result<()> {
    let extent = allocated_image.image_extent;
    let data_size: u64 = (extent.depth * extent.width * extent.height * channels).into();
    let staging_info = vk::BufferCreateInfo::builder()
        .size(data_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let staging_alloc = ALLOCATOR.allocate_buffer(
        interface,
        staging_info,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    debug_assert!(
        staging_alloc.size >= data_size,
        "Staging buffer too small for pixel data: {data_size} > {}",
        staging_alloc.size
    );
    let memory = staging_alloc.mapped_memory::<T>(&interface.device)?;
    unsafe {
        memcpy(pixels, memory, data_size as usize / size_of::<T>());
    }
    staging_alloc.unmap_memory(&interface.device);

    let copy = |cmd_buffer: vk::CommandBuffer| {
        transition_image(
            &interface.device,
            cmd_buffer,
            allocated_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let image_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let copy_region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(image_subresource)
            .image_extent(extent)
            .build();
        unsafe {
            interface.device.cmd_copy_buffer_to_image(
                cmd_buffer,
                staging_alloc.buffer,
                allocated_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        }
        transition_image(
            &interface.device,
            cmd_buffer,
            allocated_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        Ok(())
    };
    immediate_submit.execute(&interface.device, copy)?;
    ALLOCATOR.deallocate(&interface.device, &staging_alloc);

    Ok(())
}

pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range)
        .build();

    Ok(unsafe { device.create_image_view(&info, None)? })
}

pub fn copy_image_to_image(
    device: &Device,
    cmd_buffer: vk::CommandBuffer,
    source: vk::Image,
    destination: vk::Image,
    src_size: vk::Extent2D,
    dst_size: vk::Extent2D,
    dst_offsets: Option<[vk::Offset3D; 2]>,
) -> Result<()> {
    let src_offsets = [
        vk::Offset3D { x: 0, y: 0, z: 0 },
        vk::Offset3D {
            x: src_size.width as i32,
            y: src_size.height as i32,
            z: 1,
        },
    ];
    let dst_offsets = if let Some(dst_offsets) = dst_offsets {
        dst_offsets
    } else {
        [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: dst_size.width as i32,
                y: dst_size.height as i32,
                z: 1,
            },
        ]
    };

    let src_subresources = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0)
        .build();
    let dst_subresources = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0)
        .build();

    let blit_regions = &[vk::ImageBlit2::builder()
        .src_offsets(src_offsets)
        .dst_offsets(dst_offsets)
        .src_subresource(src_subresources)
        .dst_subresource(dst_subresources)
        .build()];
    let blit_info = vk::BlitImageInfo2::builder()
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(destination)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(blit_regions)
        .build();

    unsafe { device.cmd_blit_image2(cmd_buffer, &blit_info) };

    Ok(())
}

/// See [synchronization examples](https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples)
/// for optimal ways of using pipeline barriers.
pub fn transition_image(
    device: &Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };
    let subresource_range = image_subresource_range(aspect_mask);
    let image_barrier = vk::ImageMemoryBarrier2::builder()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(subresource_range)
        .image(image)
        .build();
    let image_barriers = &[image_barrier];
    let dep_info = vk::DependencyInfo::builder()
        .image_memory_barriers(image_barriers)
        .build();
    unsafe {
        device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);
    }
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
        .build()
}
