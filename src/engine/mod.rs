pub mod engine_data;
pub mod error;

use super::{
    buffer::{create_index_buffer, create_vertex_buffer, UniformBufferObject},
    camera::Camera,
    command_buffers::create_command_pool,
    frame_rate_limiter::FrameRateLimiter,
    gui::{EguiTheme, Integration as GuiIntegration},
    mesh::{gltf::GltfLoader, DrawPushConstants, MeshBuffers},
    texture::{create_texture_image, create_texture_image_view, create_texture_sampler},
    vertex::Vertex,
    Mat4,
};
use anyhow::{anyhow, Result};
use cgmath::{point3, vec3, Deg};
use egui::{FontDefinitions, Style};
use engine_data::EngineData;
use std::{
    mem::size_of,
    path::Path,
    ptr::{addr_of, copy_nonoverlapping},
    slice,
};
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    prelude::v1_3::{
        vk::{self, ExtDebugUtilsExtension, Handle, KhrSurfaceExtension, KhrSwapchainExtension},
        Device, DeviceV1_0, Entry, HasBuilder, Instance, InstanceV1_0,
    },
    Version,
};
use winit::window::Window;

/// Whether validation layers should be enabled.
pub const ENABLE_VALIDATION_LAYER: bool = cfg!(debug_assertions);

/// Names of the validation layer.
pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// Vulkan SDK version that started requiring the portability subset extension for macOS.
pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

/// Maximum number of frames that can be processed concurrently.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Vulkan engine.
pub struct Engine {
    _entry: Entry,
    pub instance: Instance,
    pub data: EngineData,
    pub device: Device,
    pub frame: usize,
    pub resized: bool,
    pub camera: Camera,
    pub frame_rate_limiter: FrameRateLimiter,
    // Gui
    pub gui_integration: Option<GuiIntegration>,
    pub gui_theme: Option<EguiTheme>,
}

impl Engine {
    /// Initialize the Vulkan engine.
    pub unsafe fn new(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|err| anyhow!("{err}"))?;
        let mut data = EngineData::default();
        let instance = data.create_instance(window, &entry)?;
        data.create_surface(&instance, window, window)?;

        data.pick_physical_device(&instance)?;
        let device = data.create_logical_device(&entry, &instance)?;

        data.create_swapchain(window, &instance, &device)?;
        data.create_swapchain_image_views(&device)?;
        data.create_render_pass(&instance, &device)?;
        data.create_descriptor_set_layout(&device)?;
        data.create_graphics_pipeline(&device)?;

        // Command pools
        // Global
        data.command_pool =
            create_command_pool(&instance, &device, data.physical_device, data.surface)?;
        // Per framebuffer
        for _ in 0..data.swapchain_images.len() {
            let command_pool =
                create_command_pool(&instance, &device, data.physical_device, data.surface)?;
            data.framebuffers_command_pools.push(command_pool);
        }

        data.create_color_objects(&instance, &device)?;
        data.create_depth_objects(&instance, &device)?;
        data.create_framebuffers(&device)?;

        // TODO: Move to EngineData
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &mut data)?;

        data.create_uniform_buffers(&instance, &device)?;
        data.create_descriptor_pool(&device)?;
        data.create_descriptor_sets(&device)?;
        data.create_command_buffers(&device)?;
        data.create_sync_objects(&device)?;

        let camera = Camera::new(
            point3(0.0, 0.5, 2.5),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 1.0, 0.0),
            data.swapchain_extent.width,
            data.swapchain_extent.height,
        );

        // TODO: Pass data instead of individual fields
        let gui_integration = GuiIntegration::new(
            data.surface,
            data.swapchain_format,
            data.physical_device,
            device.clone(),
            instance.clone(),
            data.graphics_queue,
            data.graphics_queue_family_index,
            data.swapchain,
            window,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            window.scale_factor(),
            FontDefinitions::default(),
            Style::default(),
        )?;

        let mut engine = Self {
            _entry: entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            camera,
            frame_rate_limiter: FrameRateLimiter::new(Some(1.0 / 60.0)),
            gui_integration: Some(gui_integration),
            gui_theme: Some(EguiTheme::Dark),
        };

        // TODO: Avoid having to pass engine as ref and then settings the result back to
        // egnine.data.meshes
        let meshes = engine
            .data
            .mesh_loader
            .load_gltf(&engine, Path::new("./assets/basicmesh.glb"))?;
        engine.data.meshes = meshes;

        Ok(engine)
    }

    /// Update uniform buffer.
    pub unsafe fn update_uniform_buffer(&mut self, image_index: usize) -> Result<()> {
        self.camera.on_render();
        let view = self.camera.view();

        // Correction matrix used to map OpenGL depth range to Vulkan depth range ([-1, 1.0] to [0.0, 1.0])
        // and flips the Y axis.
        #[rustfmt::skip]
        let correction = Mat4::new(
            1.0,  0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0,  0.0, 0.5, 0.0,
            0.0,  0.0, 0.5, 1.0,
        );
        let proj = correction
            * cgmath::perspective(
                Deg(45.0),
                self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
                0.1,
                100.0,
            );
        let ubo = UniformBufferObject { view, proj };

        let memory = self.device.map_memory(
            self.data.uniform_buffers[image_index].memory,
            0,
            size_of::<UniformBufferObject>() as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )?;
        copy_nonoverlapping(&ubo, memory.cast(), 1);
        self.device
            .unmap_memory(self.data.uniform_buffers[image_index].memory);

        Ok(())
    }

    /// Create a [MeshBuffers]
    pub unsafe fn create_mesh(&self, indices: &[u32], vertices: &[Vertex]) -> Result<MeshBuffers> {
        let (index_buffer_size, vertex_buffer_size) = (
            std::mem::size_of_val(indices) as vk::DeviceSize,
            std::mem::size_of_val(vertices) as vk::DeviceSize,
        );

        let vertex_buffer = create_vertex_buffer(
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.graphics_queue,
            self.data.command_pool,
            vertices,
            vertex_buffer_size,
        )?;
        let index_buffer = create_index_buffer(
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.graphics_queue,
            self.data.command_pool,
            indices,
            index_buffer_size,
        )?;

        Ok(MeshBuffers::new(index_buffer, vertex_buffer))
    }

    /// Update secondary command buffers.
    pub unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        let command_buffers = &mut self.data.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.framebuffers_command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);
            let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }
        let command_buffer = command_buffers[model_index];

        let world_matrix = Mat4::from_angle_y(Deg(-90.0));
        let push_constants = DrawPushConstants::new(
            world_matrix,
            self.data.meshes[2].mesh_buffers.vertex_buffer.address(),
        );
        let push_constants_bytes = slice::from_raw_parts(
            addr_of!(push_constants).cast::<u8>(),
            size_of::<DrawPushConstants>(),
        );

        // Commands
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.graphics_pipeline,
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.graphics_pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            push_constants_bytes,
        );
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.data.meshes[2].mesh_buffers.index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.graphics_pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );

        self.device.cmd_draw_indexed(
            command_buffer,
            self.data.meshes[2].surfaces[0].count,
            1,
            self.data.meshes[2].surfaces[0].start_index,
            0,
            0,
        );

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    /// Update command buffer.
    pub unsafe fn update_command_buffer(
        &mut self,
        image_index: usize,
        window: &Window,
    ) -> Result<()> {
        // Reset
        let command_pool = self.data.framebuffers_command_pools[image_index];
        self.device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        // Commands
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = &[color_clear_value, depth_clear_value];

        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device.cmd_begin_render_pass(
            command_buffer,
            &info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );
        let secondary_command_buffers = &[self.update_secondary_command_buffer(image_index, 0)?];
        self.device
            .cmd_execute_commands(command_buffer, secondary_command_buffers);
        self.device.cmd_end_render_pass(command_buffer);

        self.update_gui(command_buffer, image_index, window)?;

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    /// Render a frame.
    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        if !self.frame_rate_limiter.render() {
            return Ok(());
        }

        let in_flight_fence = self.data.in_flight_fences[self.frame];
        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let image_index = match self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        ) {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(err) => return Err(anyhow!(err)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;
        self.update_command_buffer(image_index, window)?;
        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let images_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(images_indices);

        let present_khr = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = present_khr == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || present_khr == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(err) = present_khr {
            return Err(anyhow!(err));
        }

        if self.frame == MAX_FRAMES_IN_FLIGHT - 1 {
            self.frame = 0;
        } else {
            self.frame += 1;
        }

        Ok(())
    }

    /// Destroy the engine.
    ///
    /// # Unsafe
    /// This method releases vk objects memory that is not managed by Rust.
    pub unsafe fn destroy(&mut self) -> Result<()> {
        self.device.device_wait_idle()?;
        if let Some(gui) = self.gui_integration.as_mut() {
            gui.destroy();
        }
        self.destroy_swapchain();
        for command_pool in self.data.framebuffers_command_pools.drain(..) {
            self.device.destroy_command_pool(command_pool, None);
        }
        for fence in self.data.in_flight_fences.drain(..) {
            self.device.destroy_fence(fence, None);
        }
        for semaphore in self.data.render_finished_semaphores.drain(..) {
            self.device.destroy_semaphore(semaphore, None);
        }
        for semaphore in self.data.image_available_semaphores.drain(..) {
            self.device.destroy_semaphore(semaphore, None);
        }

        for mesh in self.data.meshes.drain(..) {
            self.device
                .destroy_buffer(mesh.mesh_buffers.index_buffer.buffer, None);
            self.device
                .free_memory(mesh.mesh_buffers.index_buffer.memory, None);
            self.device
                .destroy_buffer(mesh.mesh_buffers.vertex_buffer.buffer, None);
            self.device
                .free_memory(mesh.mesh_buffers.vertex_buffer.memory, None);
        }

        self.data.index_buffer.destroy(&self.device);
        self.data.vertex_buffer.destroy(&self.device);

        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);
        if ENABLE_VALIDATION_LAYER {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);

        Ok(())
    }
}

/// Swapchain related methods
impl Engine {
    /// Recreate swapchain.
    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        self.data
            .create_swapchain(window, &self.instance, &self.device)?;
        self.data.create_swapchain_image_views(&self.device)?;
        self.data.create_render_pass(&self.instance, &self.device)?;
        self.data.create_graphics_pipeline(&self.device)?;
        self.data
            .create_color_objects(&self.instance, &self.device)?;
        self.data
            .create_depth_objects(&self.instance, &self.device)?;
        self.data.create_framebuffers(&self.device)?;
        self.data
            .create_uniform_buffers(&self.instance, &self.device)?;
        self.data.create_descriptor_pool(&self.device)?;
        self.data.create_descriptor_sets(&self.device)?;
        self.data.create_command_buffers(&self.device)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        self.camera.set_dimensions(
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
        );

        if let Some(egui) = self.gui_integration.as_mut() {
            egui.recreate_swapchain(
                self.data.swapchain_extent.width,
                self.data.swapchain_extent.height,
                self.data.swapchain,
                self.data.swapchain_format,
            )?;
        }

        Ok(())
    }

    /// Destroy swapchain.
    pub unsafe fn destroy_swapchain(&mut self) {
        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);
        for buffer in self.data.uniform_buffers.drain(..) {
            buffer.destroy(&self.device);
        }
        self.device
            .destroy_image_view(self.data.color_image_view, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device.destroy_image(self.data.color_image, None);
        self.device
            .destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        for framebuffer in self.data.framebuffers.drain(..) {
            self.device.destroy_framebuffer(framebuffer, None);
        }
        self.device
            .destroy_pipeline(self.data.graphics_pipeline, None);
        self.device
            .destroy_pipeline_layout(self.data.graphics_pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        for image_view in self.data.swapchain_image_views.drain(..) {
            self.device.destroy_image_view(image_view, None);
        }
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

// GUI methods
impl Engine {
    /// Update egui integration.
    pub unsafe fn update_gui(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        window: &Window,
    ) -> Result<()> {
        if let (Some(egui_integration), Some(theme)) =
            (self.gui_integration.as_mut(), self.gui_theme.as_mut())
        {
            match theme {
                EguiTheme::Dark => egui_integration
                    .context()
                    .set_visuals(egui::style::Visuals::dark()),
                EguiTheme::Light => egui_integration
                    .context()
                    .set_visuals(egui::style::Visuals::light()),
            }

            egui_integration.begin_frame(window);
            egui::SidePanel::left("my_side_panel").show(&egui_integration.context(), |ui| {
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
                .show(&egui_integration.context(), |ui| {
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
            let output = egui_integration.end_frame(window);
            let clipped_meshes = egui_integration
                .context()
                .tessellate(output.shapes, window.scale_factor() as f32);
            egui_integration.paint(
                command_buffer,
                image_index,
                clipped_meshes,
                output.textures_delta,
            )?;
        }

        Ok(())
    }
}
