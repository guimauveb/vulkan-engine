use super::{
    buffer::{create_index_buffer, create_vertex_buffer, BufferAllocation},
    command_buffers::create_command_pool,
    image::{create_image, create_image_view},
};
use anyhow::{anyhow, Result};
use bytemuck::bytes_of;
use egui::{
    epaint::{ahash::AHashMap, image::ImageDelta, Primitive},
    ClippedPrimitive, Context, FontDefinitions, FullOutput, ImageData, Pos2, Style, TextureId,
    TexturesDelta,
};
use egui_winit::{
    winit::{event::WindowEvent, window::Window},
    EventResponse, State,
};
use log::error;
use raw_window_handle::HasRawDisplayHandle;
use std::{
    mem::{size_of, size_of_val},
    ptr::copy_nonoverlapping,
};
use vulkanalia::{
    prelude::v1_3::{vk, Device, DeviceV1_0, Handle, HasBuilder, Instance},
    vk::KhrSwapchainExtension,
};

#[derive(Default, Debug, PartialEq, Eq)]
pub enum EguiTheme {
    #[default]
    Dark,
    Light,
}

fn insert_image_memory_barrier(
    device: &Device,
    cmd_buff: vk::CommandBuffer,
    image: vk::Image,
    src_q_family_index: u32,
    dst_q_family_index: u32,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    old_image_layout: vk::ImageLayout,
    new_image_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    subresource_range: impl vk::Cast<Target = vk::ImageSubresourceRange>,
) {
    let image_memory_barrier = vk::ImageMemoryBarrier::builder()
        .src_queue_family_index(src_q_family_index)
        .dst_queue_family_index(dst_q_family_index)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_image_layout)
        .new_layout(new_image_layout)
        .image(image)
        .subresource_range(subresource_range);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd_buff,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::BY_REGION,
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[image_memory_barrier],
        );
    }
}

unsafe fn create_descriptor_pool(
    device: &Device,
    descriptor_count: u32,
) -> Result<vk::DescriptorPool> {
    Ok(device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(descriptor_count)
            .pool_sizes(&[vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(descriptor_count)]),
        None,
    )?)
}

unsafe fn create_descriptor_set_layout(
    device: &Device,
    image_count: u32,
) -> Result<Vec<vk::DescriptorSetLayout>> {
    Ok((0..image_count)
        .map(|_| {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                    vk::DescriptorSetLayoutBinding::builder()
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .binding(0)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                ]),
                None,
            )
        })
        .collect::<Result<Vec<_>, _>>()?)
}

unsafe fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass> {
    Ok(device.create_render_pass(
        &vk::RenderPassCreateInfo::builder()
            .attachments(&[vk::AttachmentDescription::builder()
                .format(format)
                .samples(vk::SampleCountFlags::_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)])
            .subpasses(&[vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&[vk::AttachmentReference::builder()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])])
            .dependencies(&[vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)]),
        None,
    )?)
}

unsafe fn create_pipeline_layout(
    device: &Device,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
) -> Result<vk::PipelineLayout> {
    Ok(device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(&[
                vk::PushConstantRange::builder()
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .offset(0)
                    .size(size_of::<f32>() as u32 * 2), // screen size
            ]),
        None,
    )?)
}

unsafe fn create_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline> {
    let bindings = [vk::VertexInputBindingDescription::builder()
        .binding(0)
        .input_rate(vk::VertexInputRate::VERTEX)
        .stride(4 * size_of::<f32>() as u32 + 4 * size_of::<u8>() as u32)];

    let attributes = [
        // Position
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .offset(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT),
        // UV
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .offset(8)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT),
        // Color
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .offset(16)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM),
    ];

    let vertex_shader_module = {
        let bytes_code = include_bytes!("../shaders/gui/spv/vert.spv");
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: bytes_code.len(),
            code: bytes_code.as_ptr().cast(),
            ..Default::default()
        };
        device.create_shader_module(&shader_module_create_info, None)?
    };
    let fragment_shader_module = {
        let bytes_code = include_bytes!("../shaders/gui/spv/frag.spv");
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: bytes_code.len(),
            code: bytes_code.as_ptr().cast(),
            ..Default::default()
        };
        device.create_shader_module(&shader_module_create_info, None)?
    };
    let pipeline_shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(b"main\0"),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(b"main\0"),
    ];

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0);
    let stencil_op = vk::StencilOpState::builder()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS);
    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .front(stencil_op)
        .back(stencil_op);
    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)];
    let color_blend_info =
        vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&attributes)
        .vertex_binding_descriptions(&bindings);
    let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::_1);

    let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
        .stages(&pipeline_shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterization_info)
        .multisample_state(&multisample_info)
        .depth_stencil_state(&depth_stencil_info)
        .color_blend_state(&color_blend_info)
        .dynamic_state(&dynamic_state_info)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0)];

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_info, None)?
        .0[0];
    device.destroy_shader_module(vertex_shader_module, None);
    device.destroy_shader_module(fragment_shader_module, None);

    Ok(pipeline)
}

unsafe fn create_sampler(device: &Device) -> Result<vk::Sampler> {
    Ok(device.create_sampler(
        &vk::SamplerCreateInfo::builder()
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE),
        None,
    )?)
}

unsafe fn create_framebuffers_image_views(
    device: &Device,
    swapchain_images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    Ok(swapchain_images
        .iter()
        .map(|swapchain_image| unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(*swapchain_image)
                    .view_type(vk::ImageViewType::_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )
        })
        .collect::<Result<Vec<_>, _>>()?)
}

unsafe fn create_framebuffers(
    device: &Device,
    image_views: &[vk::ImageView],
    render_pass: vk::RenderPass,
    width: u32,
    height: u32,
) -> Result<Vec<vk::Framebuffer>> {
    Ok(image_views
        .iter()
        .map(|&image_view| unsafe {
            let attachments = &[image_view];
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(attachments)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )
        })
        .collect::<Result<Vec<_>, _>>()?)
}

pub struct Integration {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    width: u32,
    height: u32,
    scale_factor: f64,
    state: State,
    graphics_queue: vk::Queue,
    queue_family_index: u32,
    // TODO: Remove when buffer creation has been updated
    command_pools: Vec<vk::CommandPool>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    render_pass: vk::RenderPass,
    framebuffer_color_image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    vertex_buffers: Vec<BufferAllocation>,
    index_buffers: Vec<BufferAllocation>,
    texture_desc_sets: AHashMap<TextureId, vk::DescriptorSet>,
    texture_images: AHashMap<TextureId, vk::Image>,
    texture_image_infos: AHashMap<TextureId, vk::ImageCreateInfo>,
    texture_allocations: AHashMap<TextureId, vk::DeviceMemory>,
    texture_image_views: AHashMap<TextureId, vk::ImageView>,
    user_texture_layout: vk::DescriptorSetLayout,
    user_textures: Vec<Option<vk::DescriptorSet>>,
}

impl Integration {
    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    fn index_buffer_size() -> u64 {
        1024 * 1024 * 2
    }

    pub unsafe fn new<H: HasRawDisplayHandle>(
        surface: vk::SurfaceKHR,
        format: vk::Format,
        physical_device: vk::PhysicalDevice,
        device: Device,
        instance: Instance,
        graphics_queue: vk::Queue,
        queue_family_index: u32,
        swapchain: vk::SwapchainKHR,
        display_target: &H,
        width: u32,
        height: u32,
        scale_factor: f64,
        font_definitions: FontDefinitions,
        style: Style,
    ) -> Result<Self> {
        let context = Context::default();
        context.set_fonts(font_definitions);
        context.set_style(style);
        let viewport_id = context.viewport_id();
        let state = State::new(
            context,
            viewport_id,
            display_target,
            Some(scale_factor as f32),
            None,
        );

        let swapchain_images = device.get_swapchain_images_khr(swapchain)?;
        let image_count = swapchain_images.len() as u32;
        let descriptor_pool = create_descriptor_pool(&device, image_count)?;
        let descriptor_set_layouts = create_descriptor_set_layout(&device, image_count)?;
        let render_pass = create_render_pass(&device, format)?;
        let pipeline_layout = create_pipeline_layout(&device, &descriptor_set_layouts)?;
        let pipeline = create_pipeline(&device, pipeline_layout, render_pass)?;
        let sampler = create_sampler(&device)?;
        let framebuffer_color_image_views =
            create_framebuffers_image_views(&device, &swapchain_images, format)?;
        let framebuffers = create_framebuffers(
            &device,
            &framebuffer_color_image_views,
            render_pass,
            width,
            height,
        )?;

        // Only necessary because we use create_vertex_buffer / create_index_buffer
        let command_pools = swapchain_images
            .iter()
            .map(|_| create_command_pool(&instance, &device, physical_device, surface))
            .collect::<Result<Vec<_>, _>>()?;

        // Create vertex buffer and index buffer
        let (mut vertex_buffers, mut index_buffers) = (Vec::new(), Vec::new());
        for command_pool in command_pools.iter().take(framebuffers.len()) {
            // TODO: Use create_buffer instead
            let vertex_buffer = create_vertex_buffer(
                &instance,
                &device,
                physical_device,
                graphics_queue,
                *command_pool,
                &[], // Buffer will be used later, thus we don't copy anything to it now
                Self::vertex_buffer_size(),
            )?;
            vertex_buffers.push(vertex_buffer);
            // TODO: Use create_buffer instead
            let index_buffer = create_index_buffer(
                &instance,
                &device,
                physical_device,
                graphics_queue,
                *command_pool,
                &[], // Buffer will be used later, thus we don't copy anything to it now
                Self::index_buffer_size(),
            )?;
            index_buffers.push(index_buffer);
        }

        let user_texture_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .binding(0)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ]),
            None,
        )?;

        Ok(Self {
            physical_device,
            instance,
            command_pools,
            width,
            height,
            scale_factor,
            state,
            device,
            queue_family_index,
            graphics_queue,
            descriptor_pool,
            descriptor_set_layouts,
            pipeline_layout,
            pipeline,
            sampler,
            render_pass,
            framebuffer_color_image_views,
            framebuffers,
            vertex_buffers,
            index_buffers,
            texture_desc_sets: AHashMap::new(),
            texture_images: AHashMap::new(),
            texture_image_infos: AHashMap::new(),
            texture_allocations: AHashMap::new(),
            texture_image_views: AHashMap::new(),
            user_texture_layout,
            user_textures: Vec::new(),
        })
    }

    pub fn handle_event(&mut self, window: &Window, winit_event: &WindowEvent) -> EventResponse {
        self.state.on_window_event(window, winit_event)
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.state.take_egui_input(window);
        self.state.egui_ctx().begin_frame(raw_input);
    }

    pub fn end_frame(&mut self, window: &Window) -> FullOutput {
        let output = self.state.egui_ctx().end_frame();
        self.state
            .handle_platform_output(window, output.platform_output.clone());
        output
    }

    pub fn context(&self) -> Context {
        self.state.egui_ctx().clone()
    }

    // TODO: Refactoring
    unsafe fn update_texture(&mut self, texture_id: TextureId, delta: ImageDelta) -> Result<()> {
        // Extract pixel data from egui
        let data: Vec<u8> = match &delta.image {
            ImageData::Color(image) => {
                if image.width() * image.height() != image.pixels.len() {
                    return Err(anyhow!("Mismatch between texture size and texel count"));
                }
                image
                    .pixels
                    .iter()
                    .flat_map(|color| color.to_array())
                    .collect()
            }
            ImageData::Font(image) => image
                .srgba_pixels(None)
                .flat_map(|color| color.to_array())
                .collect(),
        };

        let cmd_pool = {
            let info =
                vk::CommandPoolCreateInfo::builder().queue_family_index(self.queue_family_index);
            self.device.create_command_pool(&info, None)?
        };
        let cmd_buff = {
            let info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            self.device.allocate_command_buffers(&info)?[0]
        };
        let cmd_buff_fence = {
            let info = vk::FenceCreateInfo::builder();
            self.device.create_fence(&info, None)?
        };

        // TODO: Check if we can have a method like create_vertex/index_buffer for any "staged" buffer copy (source -> staging -> destination)
        let size = data.len() as vk::DeviceSize;
        let staging_buffer = {
            BufferAllocation::new(
                &self.instance,
                &self.device,
                self.physical_device,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?
        };
        // Copy to staging buffer
        let memory =
            self.device
                .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
        copy_nonoverlapping(data.as_ptr(), memory.cast(), data.len());

        let (texture_image, texture_image_memory) = create_image(
            &self.instance,
            &self.device,
            self.physical_device,
            delta.image.width() as u32,
            delta.image.height() as u32,
            1,
            vk::SampleCountFlags::_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // TODO: Update create_image or use a specific method to retrieve the ImageCreateInfo.
        // self.texture_image_infos.insert(texture_id, info);
        let texture_image_view = create_image_view(
            &self.device,
            texture_image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        // Begin command buffer
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(cmd_buff, &info)?;
        // Transition texture image for transfer dst
        insert_image_memory_barrier(
            &self.device,
            cmd_buff,
            texture_image,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::empty(), // NONE_KHR
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .base_mip_level(0)
                .level_count(1),
        );
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(delta.image.width() as u32)
            .buffer_image_height(delta.image.height() as u32)
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            });
        self.device.cmd_copy_buffer_to_image(
            cmd_buff,
            staging_buffer.buffer,
            texture_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        insert_image_memory_barrier(
            &self.device,
            cmd_buff,
            texture_image,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .base_mip_level(0)
                .level_count(1),
        );
        self.device.end_command_buffer(cmd_buff)?;
        let cmd_buffs = [cmd_buff];
        let submit_infos = [vk::SubmitInfo::builder().command_buffers(&cmd_buffs)];
        self.device
            .queue_submit(self.graphics_queue, &submit_infos, cmd_buff_fence)?;
        self.device
            .wait_for_fences(&[cmd_buff_fence], true, u64::MAX)?;

        // Texture is now in GPU memory, now we need to decide whether we should register it as new or update existing.
        if let Some(pos) = delta.pos {
            // Blit texture data to existing texture if delta pos exists (e.g. font changed)
            if let Some(existing_texture) = self.texture_images.get(&texture_id) {
                let info = self.texture_image_infos.get(&texture_id).unwrap();
                self.device
                    .reset_command_pool(cmd_pool, vk::CommandPoolResetFlags::empty())?;
                // Begin command buffer
                let command_buffer_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(cmd_buff, &command_buffer_info)?;

                // Transition existing image for transfer dst
                insert_image_memory_barrier(
                    &self.device,
                    cmd_buff,
                    *existing_texture,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .base_mip_level(0)
                        .level_count(1),
                );
                // Transition new image for transfer src
                insert_image_memory_barrier(
                    &self.device,
                    cmd_buff,
                    texture_image,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .base_mip_level(0)
                        .level_count(1),
                );
                let top_left = vk::Offset3D {
                    x: pos[0] as i32,
                    y: pos[1] as i32,
                    z: 0,
                };
                let bottom_right = vk::Offset3D {
                    x: pos[0] as i32 + delta.image.width() as i32,
                    y: pos[1] as i32 + delta.image.height() as i32,
                    z: 1,
                };

                let region = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: info.extent.width as i32,
                            y: info.extent.height as i32,
                            z: info.extent.depth as i32,
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [top_left, bottom_right],
                };
                self.device.cmd_blit_image(
                    cmd_buff,
                    texture_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    *existing_texture,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                    vk::Filter::NEAREST,
                );
                // Transition existing image for shader read
                insert_image_memory_barrier(
                    &self.device,
                    cmd_buff,
                    *existing_texture,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .base_mip_level(0)
                        .level_count(1),
                );
                self.device.end_command_buffer(cmd_buff)?;
                let cmd_buffs = [cmd_buff];
                let submit_infos = [vk::SubmitInfo::builder().command_buffers(&cmd_buffs)];
                self.device
                    .queue_submit(self.graphics_queue, &submit_infos, cmd_buff_fence)?;
                self.device
                    .wait_for_fences(&[cmd_buff_fence], true, u64::MAX)?;

                // Destroy texture_image and view
                self.device.destroy_image(texture_image, None);
                self.device.destroy_image_view(texture_image_view, None);
                self.device.free_memory(texture_image_memory, None);
            } else {
                return Ok(());
            }
        } else {
            // Otherwise save the newly created texture

            // Update descriptor set
            let descriptor_set = {
                let set_layouts = &[self.descriptor_set_layouts[0]];
                let info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(set_layouts);
                self.device.allocate_descriptor_sets(&info)?[0]
            };
            let image_info = vk::DescriptorImageInfo::builder()
                .image_view(texture_image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(self.sampler);
            let image_info = &[image_info];
            let dsc_writes = [vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_array_element(0)
                .dst_binding(0)
                .image_info(image_info)];
            self.device
                .update_descriptor_sets(&dsc_writes, &[] as &[vk::CopyDescriptorSet]);
            // Register new texture
            self.texture_images.insert(texture_id, texture_image);
            self.texture_allocations
                .insert(texture_id, texture_image_memory);
            self.texture_image_views
                .insert(texture_id, texture_image_view);
            self.texture_desc_sets.insert(texture_id, descriptor_set);
        }

        // Cleanup
        staging_buffer.destroy(&self.device);
        self.device.destroy_command_pool(cmd_pool, None);
        self.device.destroy_fence(cmd_buff_fence, None);

        Ok(())
    }

    // TODO: Refactoring
    pub unsafe fn paint(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        clipped_meshes: Vec<ClippedPrimitive>,
        textures_delta: TexturesDelta,
    ) -> Result<()> {
        for (id, image_delta) in textures_delta.set {
            self.update_texture(id, image_delta)?;
        }

        let mut vertex_buffer_ptr = self
            .device
            .map_memory(
                self.vertex_buffers[image_index].memory,
                0,
                Self::vertex_buffer_size(),
                vk::MemoryMapFlags::empty(),
            )?
            .cast::<u8>();
        let vertex_buffer_end = vertex_buffer_ptr.add(Self::vertex_buffer_size() as usize);

        let mut index_buffer_ptr = self
            .device
            .map_memory(
                self.index_buffers[image_index].memory,
                0,
                Self::index_buffer_size(),
                vk::MemoryMapFlags::empty(),
            )?
            .cast::<u8>();
        let index_buffer_end = index_buffer_ptr.add(Self::index_buffer_size() as usize);

        // Begin render pass
        self.device.cmd_begin_render_pass(
            command_buffer,
            &vk::RenderPassBeginInfo::builder()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index])
                .clear_values(&[])
                .render_area(
                    vk::Rect2D::builder().extent(
                        vk::Extent2D::builder()
                            .width(self.width)
                            .height(self.height),
                    ),
                ),
            vk::SubpassContents::INLINE,
        );

        // Bind resources
        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );
        self.device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.vertex_buffers[image_index].buffer],
            &[0],
        );
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.index_buffers[image_index].buffer,
            0,
            vk::IndexType::UINT32,
        );
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.width as f32)
            .height(self.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        let (width_points, height_points) = (
            self.width as f32 / self.scale_factor as f32,
            self.height as f32 / self.scale_factor as f32,
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytes_of(&width_points),
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            size_of_val(&width_points) as u32,
            bytes_of(&height_points),
        );

        let (mut vertex_base, mut index_base) = (0, 0);
        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_meshes
        {
            let mesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => todo!(),
            };
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            if let egui::TextureId::User(id) = mesh.texture_id {
                if let Some(descriptor_set) = self.user_textures[id as usize] {
                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );
                } else {
                    error!(
                        "This UserTexture has already been unregistered: {:?}",
                        mesh.texture_id
                    );
                    continue;
                }
            } else {
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[*self.texture_desc_sets.get(&mesh.texture_id).unwrap()],
                    &[],
                );
            }
            let v_slice = &mesh.vertices[..];
            let v_size = size_of_val(&v_slice[0]);
            let v_copy_size = v_slice.len() * v_size;

            let i_slice = &mesh.indices[..];
            let i_size = size_of_val(&i_slice[0]);
            let i_copy_size = i_slice.len() * i_size;

            let vertex_buffer_next = vertex_buffer_ptr.add(v_copy_size);
            let index_buffer_next = index_buffer_ptr.add(i_copy_size);

            if vertex_buffer_next >= vertex_buffer_end || index_buffer_next >= index_buffer_end {
                return Err(anyhow!("egui paint out of memory"));
            }

            // Map memory
            copy_nonoverlapping(v_slice.as_ptr(), vertex_buffer_ptr.cast(), v_copy_size);
            copy_nonoverlapping(i_slice.as_ptr(), index_buffer_ptr.cast(), i_copy_size);

            vertex_buffer_ptr = vertex_buffer_next;
            index_buffer_ptr = index_buffer_next;

            // Record draw commands
            let min = clip_rect.min;
            let min = Pos2 {
                x: min.x * self.scale_factor as f32,
                y: min.y * self.scale_factor as f32,
            };
            let min = Pos2 {
                x: f32::clamp(min.x, 0.0, self.width as f32),
                y: f32::clamp(min.y, 0.0, self.height as f32),
            };
            let max = clip_rect.max;
            let max = Pos2 {
                x: max.x * self.scale_factor as f32,
                y: max.y * self.scale_factor as f32,
            };
            let max = Pos2 {
                x: f32::clamp(max.x, min.x, self.width as f32),
                y: f32::clamp(max.y, min.y, self.height as f32),
            };
            self.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D::builder()
                    .offset(
                        vk::Offset2D::builder()
                            .x(min.x.round() as i32)
                            .y(min.y.round() as i32),
                    )
                    .extent(
                        vk::Extent2D::builder()
                            .width((max.x.round() - min.x) as u32)
                            .height((max.y.round() - min.y) as u32),
                    )],
            );
            self.device.cmd_draw_indexed(
                command_buffer,
                mesh.indices.len() as u32,
                1,
                index_base,
                vertex_base,
                0,
            );

            vertex_base += mesh.vertices.len() as i32;
            index_base += mesh.indices.len() as u32;
        }

        for &id in &textures_delta.free {
            self.texture_desc_sets.remove_entry(&id); // descriptor_set is destroyed with descriptor_pool
            self.texture_image_infos.remove_entry(&id);
            if let Some((_, image)) = self.texture_images.remove_entry(&id) {
                self.device.destroy_image(image, None);
            }
            if let Some((_, image_view)) = self.texture_image_views.remove_entry(&id) {
                self.device.destroy_image_view(image_view, None);
            }
            if let Some((_, allocation)) = self.texture_allocations.remove_entry(&id) {
                self.device.free_memory(allocation, None);
            }
        }

        // End render pass
        self.device.cmd_end_render_pass(command_buffer);
        self.device
            .unmap_memory(self.vertex_buffers[image_index].memory);
        self.device
            .unmap_memory(self.index_buffers[image_index].memory);

        Ok(())
    }

    /// Destroy swapchain.
    unsafe fn destroy_swapchain(&mut self) {
        self.device.destroy_render_pass(self.render_pass, None);
        self.device.destroy_pipeline(self.pipeline, None);
        for image_view in self.framebuffer_color_image_views.drain(..) {
            self.device.destroy_image_view(image_view, None);
        }
        for framebuffer in self.framebuffers.drain(..) {
            self.device.destroy_framebuffer(framebuffer, None);
        }
    }

    /// Recreate swapchain.
    pub unsafe fn recreate_swapchain(
        &mut self,
        width: u32,
        height: u32,
        swapchain: vk::SwapchainKHR,
        format: vk::Format,
    ) -> Result<()> {
        self.destroy_swapchain();

        self.width = width;
        self.height = height;
        let swapchain_images = self.device.get_swapchain_images_khr(swapchain)?;
        self.render_pass = create_render_pass(&self.device, format)?;
        self.pipeline = create_pipeline(&self.device, self.pipeline_layout, self.render_pass)?;
        self.framebuffer_color_image_views =
            create_framebuffers_image_views(&self.device, &swapchain_images, format)?;
        self.framebuffers = create_framebuffers(
            &self.device,
            &self.framebuffer_color_image_views,
            self.render_pass,
            width,
            height,
        )?;

        Ok(())
    }

    /// Destroy GUI.
    ///
    /// # Unsafe
    /// This method releases vk objects memory that is not managed by Rust.
    pub unsafe fn destroy(&mut self) {
        self.destroy_swapchain();
        self.device
            .destroy_descriptor_set_layout(self.user_texture_layout, None);
        for command_pool in self.command_pools.drain(..) {
            self.device.destroy_command_pool(command_pool, None);
        }
        for buffer in self.index_buffers.drain(..) {
            buffer.destroy(&self.device);
        }
        for buffer in self.vertex_buffers.drain(..) {
            buffer.destroy(&self.device);
        }
        self.device.destroy_sampler(self.sampler, None);
        self.device
            .destroy_pipeline_layout(self.pipeline_layout, None);
        for descriptor_set_layout in self.descriptor_set_layouts.drain(..) {
            self.device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
        }
        self.device
            .destroy_descriptor_pool(self.descriptor_pool, None);
        for (_texture_id, texture_image) in self.texture_images.drain() {
            self.device.destroy_image(texture_image, None);
        }
        for (_texture_id, texture_image_view) in self.texture_image_views.drain() {
            self.device.destroy_image_view(texture_image_view, None);
        }
        for (_texture_id, texture_allocation) in self.texture_allocations.drain() {
            self.device.free_memory(texture_allocation, None);
        }
    }
}
