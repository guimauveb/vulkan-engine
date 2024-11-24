mod builder;
mod camera;
mod commands;
mod descriptors;
mod device;
mod frame;
mod gltf;
mod gui;
mod image;
mod material;
mod memory;
mod meshes;
mod pipelines;
mod swapchain;

use anyhow::{anyhow, Result};
use builder::EngineBuilder;
use camera::Camera;
use cgmath::{vec3, vec4, Deg, Matrix4, SquareMatrix, Vector2, Vector3, Vector4};
use commands::ImmediateSubmit;
use descriptors::{DescriptorAllocator, DescriptorWriter};
use device::QueueFamilyIndices;
use frame::Frame;
use gltf::Scene as GltfScene;
use gui::{EguiTheme, Integration as EguiIntegration};
use image::{copy_image_to_image, transition_image, write_pixels_to_image, RgbToRgbaPushConstants};
use log::{debug, error};
use material::{Material, MaterialConstants, MaterialPass, MaterialResources, MetallicRoughness};
use memory::{AllocatedImage, AllocatedImageInfo, Allocation, Allocator};
use meshes::{DrawContext, DrawPushConstants, Renderable, Scene};
use pipelines::Pipeline;
use std::{
    cmp::min,
    collections::HashMap,
    mem::transmute,
    path::Path,
    ptr::{addr_of, copy_nonoverlapping as memcpy, from_ref},
    slice::from_raw_parts,
};
use swapchain::Swapchain;
use vulkanalia::{
    prelude::v1_3::{
        vk::{self, ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension},
        Device, DeviceV1_0, Entry, Handle, HasBuilder, Instance, InstanceV1_0,
    },
    vk::DeviceV1_3,
    Version,
};
use winit::{
    event::{Event, KeyEvent, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::{Key, NamedKey},
    window::Window,
};

/// Whether validation layers should be enabled.
pub const ENABLE_VALIDATION_LAYER: bool = cfg!(debug_assertions);

/// Names of the validation layer.
pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// Vulkan SDK version that started requiring the portability subset extension for macOS.
pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

/// Maximum number of frames that can be processed concurrently.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Global Vulkan memory allocator
static ALLOCATOR: Allocator = Allocator;

/// Correction matrix used to map OpenGL depth range to Vulkan depth range ([-1, 1.0] to [0.0, 1.0])
/// and flip the y axis.
#[rustfmt::skip]
const CORRECTION_MATRIX: Mat4 = Mat4::new(
    1.0,  0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0,  0.0, 0.5, 0.0,
    0.0,  0.0, 0.5, 1.0,
);

pub type Point3 = cgmath::Point3<f32>;
pub type Vec2 = Vector2<f32>;
pub type Vec3 = Vector3<f32>;
pub type Vec4 = Vector4<f32>;
pub type Mat4 = Matrix4<f32>;

/// Holds the objects used to interract with the Vulkan API
pub struct VulkanInterface {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanInterface {
    /// Constructor
    #[inline]
    pub fn new(
        _entry: Entry,
        instance: Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        messenger: vk::DebugUtilsMessengerEXT,
    ) -> Self {
        Self {
            _entry,
            instance,
            physical_device,
            device,
            messenger,
        }
    }

    /// Destroy Vulkan objects
    fn destroy(&self) {
        unsafe {
            self.device.destroy_device(None);
            if ENABLE_VALIDATION_LAYER {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// `vulkan` based 3D engine
pub struct Engine {
    interface: VulkanInterface,
    event_loop: Option<EventLoop<()>>,
    window: Window,
    surface: vk::SurfaceKHR,
    indices: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    _present_queue: vk::Queue,
    swapchain: Swapchain,
    frames: Vec<Frame>,
    current_frame: usize,
    draw_image: AllocatedImage,
    depth_image: AllocatedImage,
    // Default textures
    white_image: AllocatedImage,
    checkerboard_image: AllocatedImage,
    default_sampler_nearest: vk::Sampler,
    default_sampler_linear: vk::Sampler,
    draw_extent: vk::Extent2D,
    descriptor_pool: DescriptorAllocator,
    descriptor_writer: DescriptorWriter,
    draw_image_descriptors: vk::DescriptorSet,
    draw_image_descriptor_layout: vk::DescriptorSetLayout,
    rgb_to_rgba_descriptors: vk::DescriptorSet,
    rgb_to_rgba_descriptor_layout: vk::DescriptorSetLayout,
    gradient_pipeline: Pipeline,
    rgb_to_rgba_pipeline: Pipeline,
    mesh_pipeline: Pipeline,
    immediate_submit: ImmediateSubmit,
    default_material: Material,
    metal_rough_material: MetallicRoughness,
    main_draw_context: DrawContext,
    scene: Scene,
    scene_descriptor_layout: vk::DescriptorSetLayout,
    scenes: HashMap<String, GltfScene>,
    gui: Option<EguiIntegration>,
    gui_theme: Option<EguiTheme>,
    camera: Option<Camera>,
}

impl Engine {
    /// Build an [Engine]
    pub fn new() -> Result<Self> {
        let mut builder = EngineBuilder::new()?;
        builder.set_swapchain()?;
        builder.set_images()?;
        builder.set_commands()?;
        builder.set_sync_structures()?;
        builder.set_descriptors()?;
        builder.set_compute_pipelines()?;
        builder.set_mesh_pipeline()?;
        builder.set_metal_roughness_material_pipeline()?;
        builder.set_samplers()?;
        builder.set_gui()?;
        builder.set_camera();

        Ok(builder.build())
    }

    /// Run main event loop
    pub fn run(mut self) -> Result<()> {
        let (mut stopped, mut minimized) = (false, false);
        let stop_engine =
            |engine: &mut Engine, stopped: &mut bool, target: &EventLoopWindowTarget<()>| {
                if let Err(err) = engine.stop() {
                    error!("Failed to stop engine: {err}");
                }
                *stopped = true;
                target.exit();
            };

        if let Some(event_loop) = self.event_loop.take() {
            event_loop.run(move |event, target| {
                target.set_control_flow(ControlFlow::Poll);
                match event {
                    Event::NewEvents(StartCause::Poll)
                        if !minimized && !stopped && !target.exiting() =>
                    {
                        self.window.request_redraw();
                    }
                    Event::WindowEvent { event, .. } => {
                        if let Some(gui) = self.gui.as_mut() {
                            if gui.handle_event(&self.window, &event).consumed {
                                return;
                            }
                        }
                        match event {
                            WindowEvent::RedrawRequested
                                if !minimized && !stopped && !target.exiting() =>
                            {
                                if let Err(err) = self.draw() {
                                    error!("Cannot render frame: {err}");
                                    stop_engine(&mut self, &mut stopped, target);
                                }
                            }
                            WindowEvent::KeyboardInput { event, .. } => match event {
                                KeyEvent {
                                    logical_key: Key::Named(key),
                                    ..
                                } => {
                                    if key == NamedKey::Escape && !stopped {
                                        stop_engine(&mut self, &mut stopped, target);
                                    }
                                }
                                KeyEvent {
                                    logical_key: key, ..
                                } => {
                                    if let Some(camera) = self.camera.as_mut() {
                                        camera.handle_key_pressed(key);
                                    }
                                }
                            },
                            WindowEvent::CursorMoved { position, .. } => {
                                if let Some(camera) = self.camera.as_mut() {
                                    camera.handle_cursor_motion(position.cast());
                                }
                            }
                            WindowEvent::MouseWheel { delta, .. } => {
                                if let Some(camera) = self.camera.as_mut() {
                                    camera.handle_scroll(delta);
                                }
                            }
                            WindowEvent::Resized(size) => {
                                if size.width == 0 || size.height == 0 {
                                    minimized = true;
                                } else {
                                    minimized = false;
                                    if let Err(err) = self.resize() {
                                        error!("Cannot resize extent: {err}");
                                    }
                                }
                            }
                            WindowEvent::CloseRequested => {
                                if !stopped {
                                    stop_engine(&mut self, &mut stopped, target);
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            })?;
        }

        Ok(())
    }

    /// Draw loop
    pub fn draw(&mut self) -> Result<()> {
        /// 1 second
        const WAIT_FOR_FENCE_TIMEOUT: u64 = 1_000_000_000;
        const ACQUIRE_NEXT_IMAGE_TIMEOUT: u64 = 1_000_000_000;

        self.update_scene()?;

        {
            // Wait until the GPU has finished rendering the last frame.
            let current_frame = &mut self.frames[self.current_frame];
            unsafe {
                self.interface.device.wait_for_fences(
                    &[current_frame.render_fence],
                    true,
                    WAIT_FOR_FENCE_TIMEOUT,
                )?;
            }
            current_frame.deallocate(&self.interface.device);
            current_frame
                .frame_descriptors
                .clear_pools(&self.interface.device)?;
        }

        let current_frame = &self.frames[self.current_frame];
        // Request image from the swapchain
        let image_index = unsafe {
            match self.interface.device.acquire_next_image_khr(
                self.swapchain.vk_swapchain,
                ACQUIRE_NEXT_IMAGE_TIMEOUT,
                current_frame.swapchain_semaphore,
                vk::Fence::null(),
            ) {
                Ok((image_index, _)) => image_index as usize,
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(),
                Err(err) => return Err(err.into()),
            }
        };

        self.draw_extent.width = min(
            self.swapchain.extent.width,
            self.draw_image.image_extent.width,
        );
        self.draw_extent.height = min(
            self.swapchain.extent.height,
            self.draw_image.image_extent.height,
        );

        unsafe {
            self.interface
                .device
                .reset_fences(&[current_frame.render_fence])?;
        }

        let cmd_buffer = current_frame.command_buffer;
        // Now that we are sure that the commands finished executing, we can safely reset the
        // command buffer to begin recording again.
        unsafe {
            self.interface
                .device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())?;
        }

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe {
            self.interface
                .device
                .begin_command_buffer(cmd_buffer, &begin_info)?;
        }
        // Transition our main draw image into general layout so that we can write into it
        transition_image(
            &self.interface.device,
            cmd_buffer,
            self.draw_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );
        // Transition depth image
        transition_image(
            &self.interface.device,
            cmd_buffer,
            self.depth_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        self.draw_background(cmd_buffer);
        self.draw_geometry(cmd_buffer)?;

        // Transition the draw image and the swapchain image to their correct transfer layouts
        transition_image(
            &self.interface.device,
            cmd_buffer,
            self.draw_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let swapchain_image = self.swapchain.images[image_index];
        transition_image(
            &self.interface.device,
            cmd_buffer,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        // Copy the draw image into the swapchain so that it can be presented to the screen
        copy_image_to_image(
            &self.interface.device,
            cmd_buffer,
            self.draw_image.image,
            swapchain_image,
            self.draw_extent,
            self.swapchain.extent,
            None,
        )?;
        // Set the swapchain image layout to attachment optimal so that we can draw it
        transition_image(
            &self.interface.device,
            cmd_buffer,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        self.draw_gui(cmd_buffer, image_index)?;

        // Set the swapchain image layout to `present` so that we can show it on the screen
        transition_image(
            &self.interface.device,
            cmd_buffer,
            swapchain_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        // Finalize command buffer so that it can be executed
        unsafe {
            self.interface.device.end_command_buffer(cmd_buffer)?;
        }

        // Prepare the submission to the queue.
        // We want to wait on `swapchain_sempahore` as it's signaled when the swapchain is ready.
        // We signal `render_sempahore` to signal that rendering has finished.
        let current_frame = &self.frames[self.current_frame];
        let cmd_info = &[vk::CommandBufferSubmitInfo::builder()
            .command_buffer(cmd_buffer)
            .device_mask(0)
            .build()];
        let wait_info = &[vk::SemaphoreSubmitInfo::builder()
            .semaphore(current_frame.swapchain_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .device_index(0)
            .value(1)
            .build()];
        let signal_info = &[vk::SemaphoreSubmitInfo::builder()
            .semaphore(current_frame.render_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .device_index(0)
            .value(1)
            .build()];
        let submit_info = &[vk::SubmitInfo2::builder()
            .wait_semaphore_infos(wait_info)
            .signal_semaphore_infos(signal_info)
            .command_buffer_infos(cmd_info)
            .build()];
        // Submit command buffer to the queue and execute it
        // `render_fence` will now block until the graphic commands finish execution
        unsafe {
            self.interface.device.queue_submit2(
                self.graphics_queue,
                submit_info,
                current_frame.render_fence,
            )?;
        }

        // Put the image we just rendered into the window.
        // We wait for `render_semaphore` as it's necessary that drawing commands have finished
        // before the image is dipslayed to the user.
        let swapchains = &[self.swapchain.vk_swapchain];
        let wait_semaphores = &[current_frame.render_semaphore];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(swapchains)
            .wait_semaphores(wait_semaphores)
            .image_indices(image_indices)
            .build();
        unsafe {
            self.interface
                .device
                .queue_present_khr(self.graphics_queue, &present_info)?;
        }

        self.update_current_frame();

        Ok(())
    }

    /// Draw geometry
    fn draw_geometry(&mut self, cmd_buffer: vk::CommandBuffer) -> Result<()> {
        // Begin a render pass connected to our draw image
        let color_attachments = &[vk::RenderingAttachmentInfo::builder()
            .image_view(self.draw_image.image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];
        let mut depth_clear_value = vk::ClearValue::default();
        depth_clear_value.depth_stencil.depth = 0.0;
        let depth_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(self.depth_image.image_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(depth_clear_value)
            .build();
        let rendering_info = vk::RenderingInfo::builder()
            .render_area(
                vk::Rect2D::builder()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(self.draw_extent),
            )
            .color_attachments(color_attachments)
            .depth_attachment(&depth_attachment)
            .layer_count(1)
            .build();

        // Begin rendering
        unsafe {
            self.interface
                .device
                .cmd_begin_rendering(cmd_buffer, &rendering_info);
            self.interface.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.mesh_pipeline.pipeline,
            );
        }
        // Set dynamic viewport
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.draw_extent.width as f32)
            .height(self.draw_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();
        unsafe {
            self.interface
                .device
                .cmd_set_viewport(cmd_buffer, 0, &[viewport]);
        }
        // Set dynamic scissor
        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: self.draw_extent.width,
                height: self.draw_extent.height,
            })
            .build();
        unsafe {
            self.interface
                .device
                .cmd_set_scissor(cmd_buffer, 0, &[scissor]);
        }
        // Allocate a new uniform buffer for the scene data
        let size = size_of::<Scene>() as vk::DeviceSize;
        let scene_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .build();
        let scene_alloc = ALLOCATOR.allocate_buffer(
            &self.interface,
            scene_info,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Scene buffer will be deallocated on the next call to `draw`
        self.frames[self.current_frame].add_allocation(scene_alloc);

        // Write into the buffer
        debug_assert!(
            scene_alloc.size >= size,
            "Buffer too small for scene data: {size} > {}",
            scene_alloc.size
        );
        let scene_memory = scene_alloc.mapped_memory::<Scene>(&self.interface.device)?;
        unsafe {
            memcpy(from_ref(&self.scene), scene_memory, 1);
        }
        scene_alloc.unmap_memory(&self.interface.device);

        // Create a descriptor set that binds that buffer and updates it
        let global_descriptor = self.frames[self.current_frame]
            .frame_descriptors
            .allocate(&self.interface.device, self.scene_descriptor_layout)?;
        self.descriptor_writer.write_buffer(
            0,
            scene_alloc.buffer,
            size,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        self.descriptor_writer
            .update_set(&self.interface.device, global_descriptor);
        self.descriptor_writer.clear();
        self.draw_scene(cmd_buffer, global_descriptor);

        // End rendering
        unsafe {
            self.interface.device.cmd_end_rendering(cmd_buffer);
        }

        Ok(())
    }

    /// Draw every object in the scene
    fn draw_scene(&self, cmd_buffer: vk::CommandBuffer, global_descriptor: vk::DescriptorSet) {
        for object in &self.main_draw_context.opaque_surfaces {
            unsafe {
                self.interface.device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    object.material.pipeline.pipeline,
                );
                self.interface.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    object.material.pipeline.layout,
                    0,
                    &[global_descriptor],
                    &[],
                );
                self.interface.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    object.material.pipeline.layout,
                    1,
                    &[object.material.material_set],
                    &[],
                );
                self.interface.device.cmd_bind_index_buffer(
                    cmd_buffer,
                    object.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                let push_constants =
                    DrawPushConstants::new(object.transform, object.vertex_buffer_address);
                self.interface.device.cmd_push_constants(
                    cmd_buffer,
                    object.material.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    from_raw_parts(
                        addr_of!(push_constants).cast::<u8>(),
                        size_of::<DrawPushConstants>(),
                    ),
                );
                self.interface.device.cmd_draw_indexed(
                    cmd_buffer,
                    object.index_count,
                    1,
                    object.first_index,
                    0,
                    0,
                );
            }
        }
    }

    /// Draw the image background
    fn draw_background(&self, cmd_buffer: vk::CommandBuffer) {
        unsafe {
            // Bind the gradient drawing compute pipeline
            self.interface.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline.pipeline,
            );
            // Bind the descriptor set containing the draw image for the compute pipeline
            self.interface.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline.layout,
                0,
                &[self.draw_image_descriptors],
                &[],
            );
            // Execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need
            // to divide by it
            self.interface.device.cmd_dispatch(
                cmd_buffer,
                (self.draw_extent.width as f32 / 16.0).ceil() as u32,
                (self.draw_extent.height as f32 / 16.0).ceil() as u32,
                1,
            );
        }
    }

    // FIXME: Does not really work, no validation errors but output colors are messed up.
    /// Convert an `RGB` image to `RGBA` via a compute shader
    fn _rgb_to_rgba(
        &self,
        cmd_buffer: vk::CommandBuffer,
        input_image_alloc: Allocation,
        output_image: AllocatedImage,
        width: u32,
        height: u32,
    ) -> Result<()> {
        transition_image(
            &self.interface.device,
            cmd_buffer,
            output_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        unsafe {
            // Bind the rgb->rgba compute pipeline
            self.interface.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.rgb_to_rgba_pipeline.pipeline,
            );
            // Bind the descriptor set containing the image for the compute pipeline
            self.interface.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.rgb_to_rgba_pipeline.layout,
                0,
                &[self.rgb_to_rgba_descriptors],
                &[],
            );
            let push_constants =
                RgbToRgbaPushConstants::new(width, height, input_image_alloc.device_address()?);
            self.interface.device.cmd_push_constants(
                cmd_buffer,
                self.rgb_to_rgba_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                from_raw_parts(
                    addr_of!(push_constants).cast::<u8>(),
                    size_of::<RgbToRgbaPushConstants>(),
                ),
            );
            // Execute the compute pipeline dispatch.
            self.interface.device.cmd_dispatch(
                cmd_buffer,
                // TODO: Check what workgroup size to chose
                (width as f32 / 16.0).ceil() as u32,
                (height as f32 / 16.0).ceil() as u32,
                1,
            );
        }

        transition_image(
            &self.interface.device,
            cmd_buffer,
            output_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        Ok(())
    }

    /// Update scene data
    fn update_scene(&mut self) -> Result<()> {
        if let Some(camera) = self.camera.as_mut() {
            camera.update();
        }

        self.main_draw_context.opaque_surfaces.clear();
        // Update global scene
        self.scene.view = self.camera.map_or_else(
            || Mat4::from_translation(vec3(0.0, 0.0, -5.0)),
            |camera| camera.view_matrix(),
        );
        // Camera projection
        self.scene.projection = CORRECTION_MATRIX
            * cgmath::perspective(
                Deg(70.0),
                self.draw_extent.width as f32 / self.draw_extent.height as f32,
                10000.0,
                0.1,
            );
        self.scene.view_proj = self.scene.projection * self.scene.view;
        self.scene.ambient_color = vec4(0.1, 0.1, 0.1, 0.1);
        self.scene.sunlight_color = vec4(1.0, 1.0, 1.0, 1.0);
        self.scene.sunlight_direction = vec4(0.0, 1.0, 0.5, 1.0);

        // Render scene
        self.scenes
            .get("structure")
            .ok_or(anyhow!("`structure` scene not found"))?
            .draw(Mat4::identity(), &mut self.main_draw_context)?;

        Ok(())
    }

    /// Load default textures, materials, meshes and scene data
    pub fn load_default_data(&mut self) -> Result<()> {
        self.load_default_textures()?;
        self.load_default_material()?;
        self.load_gltf_scene(Path::new("./assets/structure.glb"))?;

        Ok(())
    }

    /// Load default textures
    fn load_default_textures(&mut self) -> Result<()> {
        debug!("Loading default textures");
        // Like glm::packUnorm4x8, but the wrong way
        let black: u32 = unsafe { transmute([0u8, 0, 0, 0]) };
        let white: u32 = unsafe { transmute([1u8, 1, 1, 1]) };
        let img_info = AllocatedImageInfo::default()
            .extent(
                vk::Extent3D::builder()
                    .width(16)
                    .height(16)
                    .depth(1)
                    .build(),
            )
            .format(vk::Format::R8G8B8A8_UNORM)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .mipmapped(false);
        let white_image = ALLOCATOR.allocate_image(&self.interface, img_info)?;
        write_pixels_to_image(
            &self.interface,
            &self.immediate_submit,
            from_ref(&white),
            &white_image,
            4,
        )?;
        self.white_image = white_image;

        let magenta: u128 = unsafe { transmute([0.77f32, 0.77, 0.77, 0.77]) };

        // Checkerboard image
        let mut pixels = [0u32; 256];
        for x in 0..16 {
            for y in 0..16 {
                pixels[y * 16 + x] = if (x % 2) ^ (y % 2) == 0 {
                    black
                } else {
                    // Truncate as it's only for testing puproses
                    magenta as u32
                };
            }
        }
        let img_info = AllocatedImageInfo::default()
            .extent(
                vk::Extent3D::builder()
                    .width(16)
                    .height(16)
                    .depth(1)
                    .build(),
            )
            .format(vk::Format::R8G8B8A8_UNORM)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .mipmapped(false);
        let image = ALLOCATOR.allocate_image(&self.interface, img_info)?;
        write_pixels_to_image(
            &self.interface,
            &self.immediate_submit,
            pixels.as_ptr(),
            &image,
            4,
        )?;
        self.checkerboard_image = image;

        Ok(())
    }

    /// Load default material
    fn load_default_material(&mut self) -> Result<()> {
        debug!("Loading default material");
        // Set the uniform buffer for the material data
        let material_info = vk::BufferCreateInfo::builder()
            .size(size_of::<MaterialConstants>() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .build();
        let material_alloc = ALLOCATOR.allocate_buffer(
            &self.interface,
            material_info,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Write the buffer
        let scene_uniform_data = unsafe {
            material_alloc
                .mapped_memory::<MaterialConstants>(&self.interface.device)?
                .as_mut()
                .ok_or(anyhow!("`MaterialConstants` mapped memory pointer is null"))?
        };
        scene_uniform_data.color_factor = vec4(1.0, 1.0, 1.0, 1.0);
        scene_uniform_data.roughness_factor = vec4(1.0, 0.5, 0.0, 0.0);

        let material_resources = MaterialResources::new(
            self.white_image,
            self.default_sampler_linear,
            self.white_image,
            self.default_sampler_linear,
            material_alloc,
            0,
        );
        self.default_material = self.metal_rough_material.write_material(
            &self.interface.device,
            MaterialPass::MainColor,
            material_resources,
            &mut self.descriptor_pool,
        )?;

        ALLOCATOR.deallocate(&self.interface.device, &material_alloc);

        Ok(())
    }

    /// Update current frame index
    #[inline]
    fn update_current_frame(&mut self) {
        if self.current_frame == MAX_FRAMES_IN_FLIGHT - 1 {
            self.current_frame = 0;
        } else {
            self.current_frame += 1;
        }
    }

    /// Shut down the engine
    pub fn stop(&mut self) -> Result<()> {
        debug!("Shutting down engine");
        self.destroy()?;
        debug!("Engine stopped");

        Ok(())
    }

    /// destroy the resources
    fn destroy(&mut self) -> Result<()> {
        debug!("Cleaning up engine resources");
        unsafe {
            self.interface.device.device_wait_idle()?;
        }

        // Destroy the GUI
        if let Some(mut gui) = self.gui.take() {
            gui.destroy(&self.interface);
        }
        // destroy frames
        for mut frame in self.frames.drain(..) {
            frame.destroy(&self.interface.device);
        }
        // Destroy material
        self.metal_rough_material.destroy(&self.interface.device);
        // destroy scenes
        for (_, mut scene) in self.scenes.drain() {
            scene.destroy(&self.interface.device);
        }
        // Destroy pipelines
        self.gradient_pipeline.destroy(&self.interface.device);
        self.mesh_pipeline.destroy(&self.interface.device);
        self.rgb_to_rgba_pipeline.destroy(&self.interface.device);
        // Destroy descriptor resources
        unsafe {
            self.descriptor_pool.destroy_pools(&self.interface.device);
            self.interface
                .device
                .destroy_descriptor_set_layout(self.draw_image_descriptor_layout, None);
            self.interface
                .device
                .destroy_descriptor_set_layout(self.scene_descriptor_layout, None);
            self.interface
                .device
                .destroy_descriptor_set_layout(self.rgb_to_rgba_descriptor_layout, None);
        }
        // Destroy immediate submit resources
        self.immediate_submit.destroy(&self.interface.device);
        // Destroy samplers
        unsafe {
            self.interface
                .device
                .destroy_sampler(self.default_sampler_nearest, None);
            self.interface
                .device
                .destroy_sampler(self.default_sampler_linear, None);
        }
        // Destroy images
        ALLOCATOR.deallocate_image(&self.interface.device, self.draw_image);
        ALLOCATOR.deallocate_image(&self.interface.device, self.depth_image);
        ALLOCATOR.deallocate_image(&self.interface.device, self.white_image);
        ALLOCATOR.deallocate_image(&self.interface.device, self.checkerboard_image);
        // Destroy swapchain
        self.swapchain.destroy(&self.interface.device);

        unsafe {
            self.interface
                .instance
                .destroy_surface_khr(self.surface, None);
        }
        // Destroy Vulkan objects
        self.interface.destroy();

        Ok(())
    }

    /// Resize the swapchain and GUI with the new extent size
    fn resize(&mut self) -> Result<()> {
        unsafe {
            self.interface.device.device_wait_idle()?;
        }
        self.swapchain.destroy(&self.interface.device);
        self.recreate_swapchain()?;
        if let Some(gui) = self.gui.as_mut() {
            gui.resize(self.swapchain.extent);
        }

        Ok(())
    }

    /// Recreate the swapchain
    fn recreate_swapchain(&mut self) -> Result<()> {
        self.swapchain = Swapchain::new(&self.interface, self.surface, self.indices, &self.window)?;

        Ok(())
    }
}
