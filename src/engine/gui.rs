use super::{
    commands::ImmediateSubmit,
    descriptors::{DescriptorAllocator, DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio},
    image::{copy_image_to_image, transition_image, write_pixels_to_image},
    memory::{AllocatedImage, AllocatedImageInfo, Allocation},
    pipelines::{GraphicsPipelineBuilder, Pipeline},
    swapchain::Swapchain,
    Engine, Vec2, Vec4, VulkanInterface, ALLOCATOR,
};
use anyhow::{anyhow, Result};
use cgmath::{vec2, vec4};
use egui::{
    epaint::{image::ImageDelta, Primitive},
    ClippedPrimitive, Context, FontDefinitions, FullOutput, ImageData, Pos2, Style, TextureId,
    TexturesDelta,
};
use egui_winit::{
    winit::{event::WindowEvent, window::Window},
    EventResponse, State,
};
use log::error;
use std::{
    collections::HashMap,
    mem::{size_of, size_of_val},
    ptr::{addr_of, copy_nonoverlapping as memcpy},
    slice::from_raw_parts,
};
use vulkanalia::{
    prelude::v1_3::{vk, DeviceV1_0, HasBuilder},
    vk::DeviceV1_3,
    Device,
};

impl Engine {
    /// Update [`egui`] integration
    pub fn draw_gui(&mut self, cmd_buffer: vk::CommandBuffer, image_index: usize) -> Result<()> {
        if let (Some(gui), Some(theme)) = (self.gui.as_mut(), self.gui_theme.as_mut()) {
            match theme {
                EguiTheme::Dark => gui.context().set_visuals(egui::style::Visuals::dark()),
                EguiTheme::Light => gui.context().set_visuals(egui::style::Visuals::light()),
            }

            gui.begin_frame(&self.window);
            egui::SidePanel::left("my_side_panel").show(gui.context(), |ui| {
                ui.heading("vulkan-engine");
                ui.label("v0.1.0");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Theme");
                    let id = ui.make_persistent_id("theme_combo_box_side");
                    egui::ComboBox::from_id_source(id)
                        .selected_text(format!("{theme:?}"))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(theme, EguiTheme::Dark, "Dark");
                            ui.selectable_value(theme, EguiTheme::Light, "Light");
                        });
                });
                ui.separator();
            });
            egui::Window::new("Window 1")
                .resizable(true)
                .scroll2([true, true])
                .show(gui.context(), |ui| {
                    ui.heading("vulkan-engine");
                    ui.label("v0.1.0");
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Theme");
                        let id = ui.make_persistent_id("theme_combo_box_window");
                        egui::ComboBox::from_id_source(id)
                            .selected_text(format!("{theme:?}"))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(theme, EguiTheme::Dark, "Dark");
                                ui.selectable_value(theme, EguiTheme::Light, "Light");
                            });
                    });
                    ui.separator();
                });
            let output = gui.end_frame(&self.window);
            let clipped_meshes = gui
                .context()
                .tessellate(output.shapes, self.window.scale_factor() as f32);

            // Begin a render pass connected to the swapchain image
            let color_attachments = &[vk::RenderingAttachmentInfo::builder()
                .image_view(self.swapchain.views[image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build()];
            let rendering_info = vk::RenderingInfo::builder()
                .render_area(
                    vk::Rect2D::builder()
                        .offset(vk::Offset2D { x: 0, y: 0 })
                        .extent(gui.extent)
                        .build(),
                )
                .color_attachments(color_attachments)
                .layer_count(1)
                .build();
            unsafe {
                self.interface
                    .device
                    .cmd_begin_rendering(cmd_buffer, &rendering_info);
            }
            gui.draw(
                &self.interface,
                cmd_buffer,
                image_index,
                clipped_meshes,
                output.textures_delta,
            )?;
            unsafe {
                self.interface.device.cmd_end_rendering(cmd_buffer);
            }
        }

        Ok(())
    }
}

/// Default vertex buffer size
const VERTEX_BUFFER_SIZE: u64 = 1024 * 1024 * 4;
/// Default index buffer size
const INDEX_BUFFER_SIZE: u64 = 1024 * 1024 * 2;

#[derive(Default, Debug, PartialEq, Eq)]
pub enum EguiTheme {
    #[default]
    Dark,
    Light,
}

/// [`egui`] integration into the [`Engine`]
pub struct Integration {
    state: State,
    extent: vk::Extent2D,
    scale_factor: f32,
    pipeline: Pipeline,
    sampler: vk::Sampler,
    immediate_submit: ImmediateSubmit,
    descriptor_allocator: DescriptorAllocator,
    descriptor_writer: DescriptorWriter,
    allocations: Vec<MeshAllocations>,
    // TODO: Group texture data into a single struct
    texture_descriptor_sets: HashMap<TextureId, vk::DescriptorSet>,
    texture_images: HashMap<TextureId, AllocatedImage>,
    texture_image_infos: HashMap<TextureId, AllocatedImageInfo>,
    texture_layout: vk::DescriptorSetLayout,
    user_textures: Vec<Option<vk::DescriptorSet>>,
}

impl Integration {
    /// Constructor
    pub fn new(
        interface: &VulkanInterface,
        graphics_queue_index: u32,
        graphics_queue: vk::Queue,
        swapchain: &Swapchain,
        window: &Window,
    ) -> Result<Self> {
        let mut builder =
            IntegrationBuilder::new(interface, graphics_queue_index, graphics_queue, window);

        builder.set_extent(swapchain.extent);
        builder.set_commands()?;
        builder.set_sync_structures()?;
        builder.set_descriptors()?;
        builder.set_mesh_pipeline()?;
        builder.create_mesh_allocations(swapchain.images.len())?;
        builder.set_sampler()?;

        Ok(builder.build())
    }

    /// Handle window event
    pub fn handle_event(&mut self, window: &Window, winit_event: &WindowEvent) -> EventResponse {
        self.state.on_window_event(window, winit_event)
    }

    /// Begin frame
    fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.state.take_egui_input(window);
        self.state.egui_ctx().begin_frame(raw_input);
    }

    /// End frame
    fn end_frame(&mut self, window: &Window) -> FullOutput {
        let output = self.state.egui_ctx().end_frame();
        self.state
            .handle_platform_output(window, output.platform_output.clone());
        output
    }

    // Get [`egui`] [`Context`]
    fn context(&self) -> &Context {
        self.state.egui_ctx()
    }

    /// Draw the GUI
    fn draw(
        &mut self,
        interface: &VulkanInterface,
        cmd_buffer: vk::CommandBuffer,
        image_index: usize,
        clipped_meshes: Vec<ClippedPrimitive>,
        textures_delta: TexturesDelta,
    ) -> Result<()> {
        for (id, image_delta) in textures_delta.set {
            self.update_texture(interface, cmd_buffer, id, image_delta)?;
        }

        let allocations = self.allocations[image_index];
        unsafe {
            interface.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
        }
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.extent.width as f32)
            .height(self.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();
        unsafe {
            interface
                .device
                .cmd_set_viewport(cmd_buffer, 0, &[viewport]);
            interface.device.cmd_bind_index_buffer(
                cmd_buffer,
                allocations.indices.buffer,
                0,
                vk::IndexType::UINT32,
            );
        }
        let (width_points, height_points) = (
            self.extent.width as f32 / self.scale_factor,
            self.extent.height as f32 / self.scale_factor,
        );
        let push_constants = DrawPushConstants::new(
            vec2(width_points, height_points),
            allocations.vertices.device_address()?,
        );
        unsafe {
            interface.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                from_raw_parts(
                    addr_of!(push_constants).cast::<u8>(),
                    size_of::<DrawPushConstants>(),
                ),
            );
        }
        self.draw_meshes(interface, cmd_buffer, clipped_meshes, allocations)?;

        for &id in &textures_delta.free {
            // Descriptor set is destroyed with the descriptor pool
            self.texture_descriptor_sets.remove_entry(&id);
            self.texture_image_infos.remove_entry(&id);
            if let Some((_, image)) = self.texture_images.remove_entry(&id) {
                ALLOCATOR.deallocate_image(&interface.device, image);
            }
        }

        Ok(())
    }

    fn draw_meshes(
        &self,
        interface: &VulkanInterface,
        cmd_buffer: vk::CommandBuffer,
        clipped_meshes: Vec<ClippedPrimitive>,
        allocations: MeshAllocations,
    ) -> Result<()> {
        let mut staging_memory = allocations.get_staging_memory(&interface.device)?;
        let (mut vertex_size, mut index_size) = (0, 0);
        let (mut vertex_base, mut index_base) = (0, 0);
        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_meshes
        {
            let mesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => {
                    return Err(anyhow!(
                        "`Primitive::Callback(_)` primitive not implemented"
                    ))
                }
            };
            let (vertices, indices) = (
                mesh.vertices
                    .into_iter()
                    .map(Into::into)
                    .collect::<Vec<Vertex>>(),
                mesh.indices,
            );
            if vertices.is_empty() || indices.is_empty() {
                continue;
            }

            vertex_size += size_of_val(&vertices[..]);
            index_size += size_of_val(&indices[..]);

            let descriptor_set = if let TextureId::User(id) = mesh.texture_id {
                if let Some(descriptor_set) = self.user_textures[id as usize] {
                    descriptor_set
                } else {
                    error!(
                        "`UserTexture` with id '{:?}' has already been unregistered",
                        mesh.texture_id
                    );
                    continue;
                }
            } else {
                *self
                    .texture_descriptor_sets
                    .get(&mesh.texture_id)
                    .ok_or_else(|| {
                        anyhow!(
                            "Descriptor set for texture id '{:?}` not found",
                            mesh.texture_id
                        )
                    })?
            };
            unsafe {
                interface.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
            }

            let mesh_data_size = vertex_size + index_size;
            debug_assert!(
                allocations.staging.size >= u64::try_from(mesh_data_size)?,
                "Staging buffer too small for mesh data: {mesh_data_size} > {}",
                allocations.staging.size,
            );
            // Copy vertices and indices into the staging buffer
            unsafe {
                memcpy(vertices.as_ptr(), staging_memory.vertices, vertices.len());
                staging_memory.vertices = staging_memory.vertices.add(vertices.len());
                memcpy(indices.as_ptr(), staging_memory.indices, indices.len());
                staging_memory.indices = staging_memory.indices.add(indices.len());
            }

            // Record draw commands
            let min = clip_rect.min;
            let min = Pos2 {
                x: min.x * self.scale_factor,
                y: min.y * self.scale_factor,
            };
            let min = Pos2 {
                x: f32::clamp(min.x, 0.0, self.extent.width as f32),
                y: f32::clamp(min.y, 0.0, self.extent.height as f32),
            };
            let max = clip_rect.max;
            let max = Pos2 {
                x: max.x * self.scale_factor,
                y: max.y * self.scale_factor,
            };
            let max = Pos2 {
                x: f32::clamp(max.x, min.x, self.extent.width as f32),
                y: f32::clamp(max.y, min.y, self.extent.height as f32),
            };
            unsafe {
                interface.device.cmd_set_scissor(
                    cmd_buffer,
                    0,
                    &[vk::Rect2D::builder()
                        .offset(
                            vk::Offset2D::builder()
                                .x(min.x.round() as i32)
                                .y(min.y.round() as i32)
                                .build(),
                        )
                        .extent(
                            vk::Extent2D::builder()
                                .width((max.x.round() - min.x) as u32)
                                .height((max.y.round() - min.y) as u32)
                                .build(),
                        )],
                );
                interface.device.cmd_draw_indexed(
                    cmd_buffer,
                    indices.len() as u32,
                    1,
                    index_base,
                    vertex_base,
                    0,
                );
            }

            vertex_base += vertices.len() as i32;
            index_base += indices.len() as u32;
        }

        // Copy from staging to destination buffers
        let copy = |cmd_buffer: vk::CommandBuffer| {
            let vertex_copy = vk::BufferCopy::builder()
                .size(vertex_size.try_into()?)
                .build();
            unsafe {
                interface.device.cmd_copy_buffer(
                    cmd_buffer,
                    allocations.staging.buffer,
                    allocations.vertices.buffer,
                    &[vertex_copy],
                );
            }
            let index_copy = vk::BufferCopy::builder()
                .src_offset(VERTEX_BUFFER_SIZE)
                .size(index_size.try_into()?)
                .build();
            unsafe {
                interface.device.cmd_copy_buffer(
                    cmd_buffer,
                    allocations.staging.buffer,
                    allocations.indices.buffer,
                    &[index_copy],
                );
            }

            Ok(())
        };
        self.immediate_submit.execute(&interface.device, copy)?;
        allocations.staging.unmap_memory(&interface.device);

        Ok(())
    }

    fn update_texture(
        &mut self,
        interface: &VulkanInterface,
        cmd_buffer: vk::CommandBuffer,
        texture_id: TextureId,
        delta: ImageDelta,
    ) -> Result<()> {
        // Extract pixel data from egui
        let pixels: Vec<u8> = match &delta.image {
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

        let new_texture_info = AllocatedImageInfo::default()
            .extent(
                vk::Extent3D::builder()
                    .width(delta.image.width().try_into()?)
                    .height(delta.image.height().try_into()?)
                    .depth(1)
                    .build(),
            )
            .format(vk::Format::R8G8B8A8_SRGB)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .mipmapped(false);
        let new_texture = ALLOCATOR.allocate_image(interface, new_texture_info)?;
        write_pixels_to_image(
            interface,
            &self.immediate_submit,
            pixels.as_ptr(),
            &new_texture,
        )?;
        // Texture is now in GPU memory, now we need to decide whether we should register it as new or update existing.
        // TODO: Test this branch
        if let Some(position) = delta.pos {
            // Blit new texture data to existing texture if delta pos exists (e.g. font changed)
            if let Some(existing_texture) = self.texture_images.get(&texture_id) {
                let existing_texture_info = self
                    .texture_image_infos
                    .get(&texture_id)
                    .ok_or_else(|| anyhow!("Missing texture image info"))?;
                blit_new_texture(
                    interface,
                    cmd_buffer,
                    existing_texture,
                    existing_texture_info,
                    &new_texture,
                    &new_texture_info,
                    position,
                    delta,
                )?;
                self.texture_image_infos
                    .insert(texture_id, new_texture_info);
                // Destroy texture image
                ALLOCATOR.deallocate_image(&interface.device, new_texture);
            } else {
                return Ok(());
            }
        // Otherwise save the newly created texture
        } else {
            // Update descriptor set
            let descriptor_set = self
                .descriptor_allocator
                .allocate(&interface.device, self.texture_layout)?;
            self.descriptor_writer.write_image(
                0,
                new_texture.image_view,
                self.sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );
            self.descriptor_writer
                .update_set(&interface.device, descriptor_set);
            self.descriptor_writer.clear();
            // Register new texture
            self.texture_images.insert(texture_id, new_texture);
            self.texture_descriptor_sets
                .insert(texture_id, descriptor_set);
        }

        Ok(())
    }

    /// Resize the GUI
    pub fn resize(&mut self, extent: vk::Extent2D) {
        self.extent = extent;
    }

    /// Destroy the GUI
    pub fn destroy(&mut self, interface: &VulkanInterface) {
        // Destroy pipeline
        self.pipeline.cleanup(&interface.device);
        // Destroy descriptor resources
        self.descriptor_allocator.destroy_pools(&interface.device);
        unsafe {
            interface
                .device
                .destroy_descriptor_set_layout(self.texture_layout, None);
        }
        // Destroy immediate submit resources
        self.immediate_submit.cleanup(&interface.device);
        // Destroy vertex and index buffers
        for allocations in self.allocations.drain(..) {
            allocations.cleanup(&interface.device);
        }
        // Destroy sampler
        unsafe {
            interface.device.destroy_sampler(self.sampler, None);
        }
        // Destroy images
        for (_, texture_image) in self.texture_images.drain() {
            ALLOCATOR.deallocate_image(&interface.device, texture_image);
        }
    }
}

/// Blit new texture data to existing texture
#[allow(clippy::too_many_arguments)]
fn blit_new_texture(
    interface: &VulkanInterface,
    cmd_buffer: vk::CommandBuffer,
    existing_texture: &AllocatedImage,
    existing_texture_info: &AllocatedImageInfo,
    new_texture: &AllocatedImage,
    new_texture_info: &AllocatedImageInfo,
    position: [usize; 2],
    delta: ImageDelta,
) -> Result<()> {
    // Transition existing image for transfer dst
    transition_image(
        &interface.device,
        cmd_buffer,
        existing_texture.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    // Transition new image for transfer src
    transition_image(
        &interface.device,
        cmd_buffer,
        new_texture.image,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
    let top_left = vk::Offset3D {
        x: position[0] as i32,
        y: position[1] as i32,
        z: 0,
    };
    let bottom_right = vk::Offset3D {
        x: position[0] as i32 + delta.image.width() as i32,
        y: position[1] as i32 + delta.image.height() as i32,
        z: 1,
    };
    // Copy new texture to existing one
    copy_image_to_image(
        &interface.device,
        cmd_buffer,
        new_texture.image,
        existing_texture.image,
        vk::Extent2D {
            width: new_texture_info.extent.width,
            height: new_texture_info.extent.height,
        },
        vk::Extent2D {
            width: existing_texture_info.extent.width,
            height: existing_texture_info.extent.height,
        },
        Some([top_left, bottom_right]),
    )?;
    // Transition existing image for shader read
    transition_image(
        &interface.device,
        cmd_buffer,
        existing_texture.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    Ok(())
}

struct IntegrationBuilder<'v> {
    interface: &'v VulkanInterface,
    graphics_queue_index: u32,
    graphics_queue: vk::Queue,
    state: State,
    extent: vk::Extent2D,
    scale_factor: f32,
    texture_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    pipeline: Pipeline,
    descriptor_allocator: DescriptorAllocator,
    descriptor_writer: DescriptorWriter,
    immediate_fence: vk::Fence,
    immediate_command_buffer: vk::CommandBuffer,
    immediate_command_pool: vk::CommandPool,
    allocations: Vec<MeshAllocations>,
}

impl<'v> IntegrationBuilder<'v> {
    /// Initialize the builder
    fn new(
        interface: &'v VulkanInterface,
        graphics_queue_index: u32,
        graphics_queue: vk::Queue,
        window: &Window,
    ) -> Self {
        let context = Context::default();
        context.set_fonts(FontDefinitions::default());
        context.set_style(Style::default());
        let viewport_id = context.viewport_id();
        let scale_factor = window.scale_factor() as f32;
        let state = State::new(context, viewport_id, window, Some(scale_factor), None);

        Self {
            interface,
            graphics_queue_index,
            graphics_queue,
            state,
            extent: vk::Extent2D::default(),
            scale_factor,
            texture_layout: vk::DescriptorSetLayout::default(),
            sampler: vk::Sampler::default(),
            pipeline: Pipeline::default(),
            descriptor_allocator: DescriptorAllocator::default(),
            descriptor_writer: DescriptorWriter::default(),
            immediate_fence: vk::Fence::default(),
            immediate_command_buffer: vk::CommandBuffer::default(),
            immediate_command_pool: vk::CommandPool::default(),
            allocations: Vec::default(),
        }
    }

    /// Build an [`Integration`]
    fn build(self) -> Integration {
        Integration {
            state: self.state,
            extent: self.extent,
            scale_factor: self.scale_factor,
            pipeline: self.pipeline,
            sampler: self.sampler,
            descriptor_allocator: self.descriptor_allocator,
            descriptor_writer: self.descriptor_writer,
            allocations: self.allocations,
            immediate_submit: ImmediateSubmit::new(
                self.immediate_command_pool,
                self.immediate_command_buffer,
                self.immediate_fence,
                self.graphics_queue,
            ),
            texture_descriptor_sets: HashMap::new(),
            texture_images: HashMap::new(),
            texture_image_infos: HashMap::new(),
            texture_layout: self.texture_layout,
            user_textures: Vec::new(),
        }
    }

    // Set extent
    fn set_extent(&mut self, extent: vk::Extent2D) {
        self.extent = extent;
    }

    /// Initialize commands
    fn set_commands(&mut self) -> Result<()> {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.graphics_queue_index)
            .build();
        // Immediate submit commands
        self.immediate_command_pool = unsafe {
            self.interface
                .device
                .create_command_pool(&pool_info, None)?
        };
        let cmd_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.immediate_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .build();
        self.immediate_command_buffer = unsafe {
            self.interface
                .device
                .allocate_command_buffers(&cmd_alloc_info)?[0]
        };

        Ok(())
    }

    /// Create the synchronization structures
    fn set_sync_structures(&mut self) -> Result<()> {
        // Immediate submit fence
        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();
        self.immediate_fence = unsafe { self.interface.device.create_fence(&fence_info, None)? };

        Ok(())
    }

    /// Initialize resource descriptors
    fn set_descriptors(&mut self) -> Result<()> {
        let sizes = vec![PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 1.0)];
        self.descriptor_allocator = DescriptorAllocator::new(&self.interface.device, 10, sizes)?;

        let mut layout_builder = DescriptorLayoutBuilder::default();
        layout_builder.add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        self.texture_layout = layout_builder.build(
            &self.interface.device,
            vk::ShaderStageFlags::FRAGMENT,
            Option::<vk::DescriptorSetLayoutBindingFlagsCreateInfo>::None,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;

        Ok(())
    }

    /// Create the mesh graphics pipeline
    fn set_mesh_pipeline(&mut self) -> Result<()> {
        let mut builder = GraphicsPipelineBuilder::new(&self.interface.device);
        let push_constant_ranges = &[vk::PushConstantRange::builder()
            .offset(0)
            .size(size_of::<DrawPushConstants>().try_into()?)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()];
        let layouts = &[self.texture_layout];
        let layout = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(push_constant_ranges)
            .set_layouts(layouts)
            .build();
        builder.set_layout(&layout)?;
        builder.set_shaders(
            include_bytes!("../../shaders/gui/spv/vert.spv"),
            include_bytes!("../../shaders/gui/spv/frag.spv"),
        )?;
        builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        builder.set_polygon_mode(vk::PolygonMode::FILL);
        // No backface culling
        builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE);
        // No multisampling
        builder.disable_multisampling();
        // Color blending
        builder.enable_blending_alphablend();
        // Depth test
        builder.enable_depthtest(vk::TRUE, vk::CompareOp::ALWAYS);
        // Connect the image format we will draw into
        builder.set_color_attachment_format(vk::Format::R8G8B8A8_SRGB);
        builder.set_depth_format(vk::Format::D32_SFLOAT);
        builder.set_viewport_state();
        builder.set_dynamic_states(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        self.pipeline = builder.build()?;

        Ok(())
    }

    /// Create mesh allocations
    fn create_mesh_allocations(&mut self, image_count: usize) -> Result<()> {
        self.allocations = (0..image_count)
            .map(|_| MeshAllocations::new(self.interface))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    /// Initialize the sampler
    fn set_sampler(&mut self) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .build();
        self.sampler = unsafe { self.interface.device.create_sampler(&sampler_info, None)? };

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    color: Vec4,
    position: Vec2,
    uv: Vec2,
}

impl From<egui::epaint::Vertex> for Vertex {
    fn from(value: egui::epaint::Vertex) -> Self {
        // TODO: Pass colors to the shader as RGBA (0-255 range) to avoid unnecessary conversions?
        let color = value.color.to_normalized_gamma_f32();

        Self {
            color: vec4(color[0], color[1], color[2], color[3]),
            position: vec2(value.pos.x, value.pos.y),
            uv: vec2(value.uv.x, value.uv.y),
        }
    }
}

/// Push constants for mesh object draws
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct DrawPushConstants {
    screen_size: Vec2,
    vertex_buffer: vk::DeviceAddress,
}

impl DrawPushConstants {
    /// Constructor
    #[inline]
    fn new(screen_size: Vec2, vertex_buffer: vk::DeviceAddress) -> Self {
        Self {
            screen_size,
            vertex_buffer,
        }
    }
}

/// Meshes are written into GPU memory through a single staging buffer
#[derive(Copy, Clone, Debug)]
struct MeshAllocations {
    staging: Allocation,
    vertices: Allocation,
    indices: Allocation,
}

impl MeshAllocations {
    /// Constructor
    fn new(interface: &VulkanInterface) -> Result<Self> {
        // Staging buffers
        let staging_info = vk::BufferCreateInfo::builder()
            .size(VERTEX_BUFFER_SIZE + INDEX_BUFFER_SIZE)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .build();
        let staging_alloc = ALLOCATOR.allocate_buffer(
            interface,
            staging_info,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        // Destination buffers
        let vertex_info = vk::BufferCreateInfo::builder()
            .size(VERTEX_BUFFER_SIZE)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .build();
        let vertex_alloc = ALLOCATOR.allocate_buffer(
            interface,
            vertex_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let index_info = vk::BufferCreateInfo::builder()
            .size(INDEX_BUFFER_SIZE)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .build();
        let index_alloc = ALLOCATOR.allocate_buffer(
            interface,
            index_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Ok(Self {
            staging: staging_alloc,
            vertices: vertex_alloc,
            indices: index_alloc,
        })
    }

    /// Get mesh allocations staging memory pointers
    fn get_staging_memory(&self, device: &Device) -> Result<MeshStagingMemory> {
        const MESH_BUFFER_SIZE: u64 = VERTEX_BUFFER_SIZE + INDEX_BUFFER_SIZE;

        let staging = self.staging.get_mapped_memory(device)?;
        debug_assert!(
            self.staging.size >= u64::try_from(MESH_BUFFER_SIZE)?,
            "Staging buffer too small for mesh data: {MESH_BUFFER_SIZE} > {}",
            self.staging.size,
        );
        let vertices: *mut Vertex = staging.cast();
        let indices: *mut u32 = unsafe {
            // Index buffer starts at VERTEX_BUFFER_SIZE offset
            staging.add(VERTEX_BUFFER_SIZE.try_into()?).cast()
        };

        Ok(MeshStagingMemory { vertices, indices })
    }

    /// Cleanup resources
    fn cleanup(&self, device: &Device) {
        ALLOCATOR.deallocate(device, &self.staging);
        ALLOCATOR.deallocate(device, &self.vertices);
        ALLOCATOR.deallocate(device, &self.indices);
    }
}

/// Mesh staging memory pointers
#[derive(Copy, Clone, Debug)]
struct MeshStagingMemory {
    vertices: *mut Vertex,
    indices: *mut u32,
}
