// TODO:- Custom build to automatically compile shaders when compiling the program
//      - Check pub/private code
//      - Docstring

mod buffer;
mod camera;
mod command_buffers;
mod descriptor_sets;
mod device;
mod gui;
mod image;
mod mesh;
mod pipeline;
mod swapchain;
mod texture;
mod vertex;

use {
    anyhow::{anyhow, Result},
    buffer::{create_uniform_buffers, create_vertex_buffer, BufferAllocation, UniformBufferObject},
    camera::Camera,
    cgmath::{point3, vec3, Deg, Matrix4, Vector2, Vector3, Vector4},
    command_buffers::{create_command_buffers, create_command_pool},
    descriptor_sets::{create_descriptor_pool, create_descriptor_sets},
    device::{create_logical_device, pick_physical_device, QueueFamilyIndices},
    egui::{FontDefinitions, Style},
    gui::{EguiTheme, Integration},
    hashbrown::HashSet,
    image::{create_color_objects, create_depth_objects, create_image_view, get_depth_format},
    log::{debug, error, info, trace, warn},
    mesh::{load_gltf_meshes, load_obj_meshes, MeshBuffers},
    minstant::Instant,
    pipeline::{create_descriptor_set_layout, create_pipeline, create_render_pass},
    std::{
        ffi::CStr,
        mem::size_of,
        os::raw::c_void,
        path::Path,
        ptr::{addr_of, copy_nonoverlapping},
        slice,
    },
    swapchain::{create_swapchain, create_swapchain_image_views},
    texture::{create_texture_image, create_texture_image_view, create_texture_sampler},
    vertex::Vertex,
    vulkanalia::{
        loader::{LibloadingLoader, LIBRARY},
        prelude::v1_3::{
            vk::{
                self, ExtDebugUtilsExtension, Handle, KhrSurfaceExtension, KhrSwapchainExtension,
            },
            Device, DeviceV1_0, Entry, EntryV1_0, HasBuilder, Instance, InstanceV1_0,
        },
        window as vk_window, Version,
    },
    winit::{
        dpi::LogicalSize,
        event::{Event, KeyEvent, StartCause, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        keyboard::{Key, NamedKey},
        window::{Window, WindowBuilder},
    },
};

/// Whether validation layers should be enabled.
const ENABLE_VALIDATION_LAYER: bool = cfg!(debug_assertions);

/// Names of the validation layer.
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// Required device extensions.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

/// Vulkan SDK version that started requiring the portability subset extension for macOS.
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

/// Maximum number of frames that can be processed concurrently.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub type Point3 = cgmath::Point3<f32>;
pub type Vec2 = Vector2<f32>;
pub type Vec3 = Vector3<f32>;
pub type Vec4 = Vector4<f32>;
pub type Mat4 = Matrix4<f32>;

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).message) }.to_string_lossy();
    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({type_:?}) {message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({type_:?}) {message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({type_:?}) {message}");
    } else {
        trace!("({type_:?}) {message}");
    }

    vk::FALSE
}

/// The Vulkan handles and associated properties used by our Vulkan engine.
#[derive(Default)]
struct EngineData {
    // Debug
    messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    surface: vk::SurfaceKHR,
    // Physical / Logical device
    physical_device: vk::PhysicalDevice,
    msaa_samples: vk::SampleCountFlags,
    // Queues
    graphics_queue: vk::Queue,
    graphics_queue_family_index: u32,
    present_queue: vk::Queue,
    present_queue_family_index: u32,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // Color image
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    // Depth
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    // Texture
    mip_levels: u32,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    // Buffers
    vertex_buffer: BufferAllocation,
    index_buffer: BufferAllocation,
    uniform_buffers: Vec<BufferAllocation>,
    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    // Command buffers
    command_pool: vk::CommandPool,
    framebuffers_command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    // Sync objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
    // Gui
    egui_integration: Option<Integration>,
    theme: Option<EguiTheme>,
}

unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut EngineData,
) -> Result<Instance> {
    let enginelication_info = vk::ApplicationInfo::builder()
        .application_name(b"vulkan-engine\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"vulkan-engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 3, 0));
    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();
    if ENABLE_VALIDATION_LAYER && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if ENABLE_VALIDATION_LAYER {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let layers = if ENABLE_VALIDATION_LAYER {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&enginelication_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);
    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));
    if ENABLE_VALIDATION_LAYER {
        info = info.push_next(&mut debug_info);
    }
    let instance = entry.create_instance(&info, None)?;
    if ENABLE_VALIDATION_LAYER {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

unsafe fn create_framebuffers(device: &Device, data: &mut EngineData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[data.color_image_view, data.depth_image_view, *i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

unsafe fn create_sync_objects(device: &Device, data: &mut EngineData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

/// Vulkan engine.
struct Engine {
    _entry: Entry,
    instance: Instance,
    data: EngineData,
    device: Device,
    frame: usize,
    resized: bool,
    // Number of seconds since the last loop
    last_frame_time: Instant,
    // Number of seconds since the last frame
    last_update_time: Instant,
    camera: Camera,
}

impl Engine {
    /// Create a Vulkan engine.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|err| anyhow!("{err}"))?;
        let mut data = EngineData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;

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

        create_color_objects(&instance, &device, &mut data)?;
        create_depth_objects(&instance, &device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        // TODO: Proper scene loading
        load_obj_meshes(&instance, &device, &mut data)?;
        // TODO: WIP
        load_gltf_meshes(
            &instance,
            &device,
            &mut data,
            Path::new("resources/basicmesh.glb"),
        )?;
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &mut data)?;

        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        let camera = Camera::new(
            point3(0.0, 0.5, 2.5),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 1.0, 0.0),
            data.swapchain_extent.width,
            data.swapchain_extent.height,
        );

        let egui_integration = gui::Integration::new(
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
        data.theme = Some(EguiTheme::Dark);
        data.egui_integration = Some(egui_integration);

        Ok(Self {
            _entry: entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            last_frame_time: Instant::now(),
            last_update_time: Instant::now(),
            camera,
        })
    }

    /// Recreate swapchain.
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_color_objects(&self.instance, &self.device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        self.camera.set_dimensions(
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
        );

        if let Some(egui) = self.data.egui_integration.as_mut() {
            egui.recreate_swapchain(
                self.data.swapchain_extent.width,
                self.data.swapchain_extent.height,
                self.data.swapchain,
                self.data.swapchain_format,
            )?;
        }

        Ok(())
    }

    /// Update uniform buffer.
    unsafe fn update_uniform_buffer(&mut self, image_index: usize) -> Result<()> {
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

    /// Update secondary command buffers.
    unsafe fn update_secondary_command_buffer(
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

        let model = Mat4::from_angle_y(Deg(-90.0)) * Mat4::from_angle_x(Deg(-90.0));
        let model_bytes = slice::from_raw_parts(addr_of!(model).cast::<u8>(), size_of::<Mat4>());

        let opacity_bytes = &100.0f32.to_ne_bytes()[..];

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
            self.data.pipeline,
        );
        self.device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.data.vertex_buffer.buffer],
            &[0],
        );
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.data.index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );
        self.device
            .cmd_draw_indexed(command_buffer, self.data.indices.len() as u32, 1, 0, 0, 0);

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    unsafe fn upload_mesh(&mut self, indices: &[u32], vertices: &[Vertex]) -> Result<MeshBuffers> {
        let (index_buffer_size, vertex_buffer_size) = (
            (indices.len() * size_of::<u32>()) as u64,
            (vertices.len() * size_of::<Vertex>()) as u64,
        );

        let mut new_surface = MeshBuffers::default();
        new_surface.vertex_buffer = create_vertex_buffer(
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.graphics_queue,
            self.data.command_pool,
            vertices,
            vertex_buffer_size,
        )?;

        // TODO: Get the address of the vertex buffer
        todo!();
    }

    /// Update egui integration.
    unsafe fn update_gui(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        window: &Window,
    ) -> Result<()> {
        if let (Some(egui_integration), Some(theme)) = (
            self.data.egui_integration.as_mut(),
            self.data.theme.as_mut(),
        ) {
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

    /// Update command buffer.
    unsafe fn update_command_buffer(&mut self, image_index: usize, window: &Window) -> Result<()> {
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
        // XXX - Not based on model number anymore
        let secondary_command_buffers = &[self.update_secondary_command_buffer(image_index, 0)?];
        self.device
            .cmd_execute_commands(command_buffer, secondary_command_buffers);
        self.device.cmd_end_render_pass(command_buffer);

        self.update_gui(command_buffer, image_index, window)?;

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    /// Render a frame.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
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

    /// Destroy swapchain.
    unsafe fn destroy_swapchain(&mut self) {
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
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        for image_view in self.data.swapchain_image_views.drain(..) {
            self.device.destroy_image_view(image_view, None);
        }
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    /// Destroy the engine.
    ///
    /// # Unsafe
    /// This method releases vk objects memory that is not managed by Rust.
    unsafe fn destroy(&mut self) -> Result<()> {
        self.device.device_wait_idle()?;
        if let Some(gui) = self.data.egui_integration.as_mut() {
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

fn render(engine: &mut Engine, window: &Window) -> Result<()> {
    /// Limit the number of frames to 60 per second.
    const FRAME_CAP: f32 = 1.0 / 60.0;

    let now = Instant::now();
    // Use delta time time if necessary (for physics, tweening, etc.)
    let _delta_time = (now - engine.last_update_time).as_secs_f32();
    if (now - engine.last_frame_time).as_secs_f32() >= FRAME_CAP {
        unsafe { engine.render(window) }?;
        engine.last_frame_time = now;
    }
    engine.last_update_time = now;

    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init();
    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("vulkan-engine")
        .with_inner_size(LogicalSize::new(1920, 1080))
        // TODO: Handle fullscreen
        // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        .build(&event_loop)?;
    window.set_cursor_visible(true);

    let mut engine = unsafe { Engine::create(&window)? };
    let (mut destroyed, mut minimized) = (false, false);

    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::NewEvents(StartCause::Poll) if !minimized && !destroyed && !target.exiting() => {
                window.request_redraw();
            }
            Event::WindowEvent { event, .. } => {
                if let Some(gui) = engine.data.egui_integration.as_mut() {
                    _ = gui.handle_event(&window, &event);
                }

                match event {
                    WindowEvent::RedrawRequested
                        if !minimized && !destroyed && !target.exiting() =>
                    {
                        if let Err(err) = render(&mut engine, &window) {
                            error!("Cannot render frame: {err}");
                            unsafe {
                                if let Err(err) = engine.destroy() {
                                    error!("Error when destroying engine: {err}");
                                }
                            }
                            destroyed = true;
                            target.exit();
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => match event {
                        KeyEvent {
                            logical_key: Key::Named(key),
                            ..
                        } => {
                            if key == NamedKey::Escape && !destroyed {
                                unsafe {
                                    if let Err(err) = engine.destroy() {
                                        error!("Error when destroying engine: {err}");
                                    }
                                }
                                destroyed = true;
                                target.exit();
                            }
                        }
                        KeyEvent {
                            logical_key: key, ..
                        } => {
                            engine.camera.on_keyboard(key);
                        }
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        engine.camera.on_mouse(position.cast());
                    }
                    WindowEvent::Resized(size) => {
                        if size.width == 0 || size.height == 0 {
                            minimized = true;
                        } else {
                            minimized = false;
                            engine.resized = true;
                        }
                    }
                    WindowEvent::CloseRequested => {
                        if !destroyed {
                            unsafe {
                                if let Err(err) = engine.destroy() {
                                    error!("Error when destroying engine: {err}");
                                }
                            }
                            destroyed = true;
                            target.exit();
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
