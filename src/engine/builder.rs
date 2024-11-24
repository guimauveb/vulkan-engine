use super::{
    camera::Camera,
    descriptors::{DescriptorAllocator, DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio},
    device::{create_logical_device, pick_physical_device, QueueFamilyIndices},
    frame::Frame,
    gui::{EguiTheme, Integration as EguiIntegration},
    image::RgbToRgbaPushConstants,
    material::{Material, MetallicRoughness, MetallicRoughnessBuilder},
    memory::{AllocatedImage, AllocatedImageInfo},
    meshes::{DrawContext, DrawPushConstants, Scene},
    pipelines::{ComputePipelineBuilder, GraphicsPipelineBuilder, Pipeline},
    swapchain::Swapchain,
    Engine, ImmediateSubmit, VulkanInterface, ALLOCATOR, ENABLE_VALIDATION_LAYER,
    MAX_FRAMES_IN_FLIGHT, PORTABILITY_MACOS_VERSION, VALIDATION_LAYER,
};
use anyhow::{anyhow, Result};
use cgmath::{point3, vec3};
use log::{debug, error, info, trace, warn};
use std::{
    collections::{HashMap, HashSet},
    ffi::{c_void, CStr},
};
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    prelude::v1_3::{
        vk::{self, ExtDebugUtilsExtension},
        DeviceV1_0, Entry, EntryV1_0, HasBuilder, Instance,
    },
    vk::Handle,
    window as vk_window,
};
use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

/// Logger callback used when validation layer is enabled
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

pub struct EngineBuilder {
    interface: VulkanInterface,
    event_loop: EventLoop<()>,
    window: Window,
    surface: vk::SurfaceKHR,
    indices: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: Swapchain,
    frames: Vec<Frame>,
    draw_image: AllocatedImage,
    depth_image: AllocatedImage,
    default_sampler_nearest: vk::Sampler,
    default_sampler_linear: vk::Sampler,
    single_image_descriptor_layout: vk::DescriptorSetLayout,
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
    immediate_fence: vk::Fence,
    immediate_command_buffer: vk::CommandBuffer,
    immediate_command_pool: vk::CommandPool,
    scene_descriptor_layout: vk::DescriptorSetLayout,
    metal_rough_material: MetallicRoughness,
    gui: Option<EguiIntegration>,
    camera: Option<Camera>,
}

impl EngineBuilder {
    /// Initialize the builder
    pub fn new() -> Result<Self> {
        let event_loop = EventLoop::new()?;
        let window = WindowBuilder::new()
            .with_title("vulkan-engine")
            .with_inner_size(LogicalSize::new(1920, 1080))
            .build(&event_loop)?;
        window.set_cursor_visible(true);

        let loader = unsafe { LibloadingLoader::new(LIBRARY)? };
        let entry = unsafe { Entry::new(loader).map_err(|err| anyhow!("{err}"))? };
        let mut messenger = vk::DebugUtilsMessengerEXT::default();
        let instance = create_vulkan_instance(&window, &entry, &mut messenger)?;
        let surface = unsafe { vk_window::create_surface(&instance, &window, &window)? };
        let physical_device = pick_physical_device(&instance, surface)?;

        let indices = QueueFamilyIndices::get(&instance, physical_device, surface)?;
        let mut unique_indices = HashSet::new();
        unique_indices.insert(indices.graphics);
        unique_indices.insert(indices.present);
        let device = create_logical_device(&entry, &instance, physical_device, &unique_indices)?;

        let graphics_queue = unsafe { device.get_device_queue(indices.graphics, 0) };
        let present_queue = unsafe { device.get_device_queue(indices.present, 0) };

        let interface = VulkanInterface::new(entry, instance, physical_device, device, messenger);

        Ok(Self {
            interface,
            window,
            event_loop,
            surface,
            indices,
            graphics_queue,
            present_queue,
            swapchain: Swapchain::default(),
            frames: (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| Frame::default())
                .collect(),
            draw_image: AllocatedImage::default(),
            depth_image: AllocatedImage::default(),
            default_sampler_nearest: vk::Sampler::default(),
            default_sampler_linear: vk::Sampler::default(),
            single_image_descriptor_layout: vk::DescriptorSetLayout::default(),
            draw_extent: vk::Extent2D::default(),
            descriptor_pool: DescriptorAllocator::default(),
            descriptor_writer: DescriptorWriter::default(),
            draw_image_descriptors: vk::DescriptorSet::default(),
            draw_image_descriptor_layout: vk::DescriptorSetLayout::default(),
            rgb_to_rgba_descriptors: vk::DescriptorSet::default(),
            rgb_to_rgba_descriptor_layout: vk::DescriptorSetLayout::default(),
            gradient_pipeline: Pipeline::default(),
            rgb_to_rgba_pipeline: Pipeline::default(),
            mesh_pipeline: Pipeline::default(),
            immediate_fence: vk::Fence::default(),
            immediate_command_buffer: vk::CommandBuffer::default(),
            immediate_command_pool: vk::CommandPool::default(),
            scene_descriptor_layout: vk::DescriptorSetLayout::default(),
            metal_rough_material: MetallicRoughness::default(),
            gui: None,
            camera: None,
        })
    }

    /// Build an [`Engine`]
    pub fn build(self) -> Engine {
        Engine {
            interface: self.interface,
            event_loop: Some(self.event_loop),
            window: self.window,
            surface: self.surface,
            indices: self.indices,
            graphics_queue: self.graphics_queue,
            _present_queue: self.present_queue,
            swapchain: self.swapchain,
            draw_image: self.draw_image,
            depth_image: self.depth_image,
            white_image: AllocatedImage::default(),
            checkerboard_image: AllocatedImage::default(),
            default_sampler_nearest: self.default_sampler_nearest,
            default_sampler_linear: self.default_sampler_linear,
            draw_extent: self.draw_extent,
            frames: self.frames,
            current_frame: 0,
            descriptor_pool: self.descriptor_pool,
            descriptor_writer: self.descriptor_writer,
            draw_image_descriptors: self.draw_image_descriptors,
            draw_image_descriptor_layout: self.draw_image_descriptor_layout,
            rgb_to_rgba_descriptors: self.rgb_to_rgba_descriptors,
            rgb_to_rgba_descriptor_layout: self.rgb_to_rgba_descriptor_layout,
            gradient_pipeline: self.gradient_pipeline,
            rgb_to_rgba_pipeline: self.rgb_to_rgba_pipeline,
            mesh_pipeline: self.mesh_pipeline,
            immediate_submit: ImmediateSubmit::new(
                self.immediate_command_pool,
                self.immediate_command_buffer,
                self.immediate_fence,
                self.graphics_queue,
            ),
            default_material: Material::default(),
            metal_rough_material: self.metal_rough_material,
            main_draw_context: DrawContext::default(),
            scene: Scene::default(),
            scene_descriptor_layout: self.scene_descriptor_layout,
            scenes: HashMap::default(),
            gui: self.gui,
            gui_theme: Some(EguiTheme::Dark),
            camera: self.camera,
        }
    }

    /// Initialize the swapchain
    pub fn set_swapchain(&mut self) -> Result<()> {
        self.swapchain = Swapchain::new(&self.interface, self.surface, self.indices, &self.window)?;

        Ok(())
    }

    /// Initialize the images
    pub fn set_images(&mut self) -> Result<()> {
        self.set_draw_image()?;
        self.set_depth_image()?;

        Ok(())
    }

    /// Initialize draw image
    fn set_draw_image(&mut self) -> Result<()> {
        let extent = vk::Extent3D::builder()
            .width(self.swapchain.extent.width)
            .height(self.swapchain.extent.height)
            .depth(1)
            .build();
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let info = AllocatedImageInfo::default()
            .extent(extent)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .usage(usage)
            .mipmapped(false);
        self.draw_image = ALLOCATOR.allocate_image(&self.interface, info)?;

        Ok(())
    }

    /// Initialize depth image
    fn set_depth_image(&mut self) -> Result<()> {
        let extent = vk::Extent3D::builder()
            .width(self.swapchain.extent.width)
            .height(self.swapchain.extent.height)
            .depth(1)
            .build();
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        let info = AllocatedImageInfo::default()
            .extent(extent)
            .format(vk::Format::D32_SFLOAT)
            .usage(usage)
            .mipmapped(false);
        self.depth_image = ALLOCATOR.allocate_image(&self.interface, info)?;

        Ok(())
    }

    /// Initialize commands
    pub fn set_commands(&mut self) -> Result<()> {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.indices.graphics)
            .build();
        for frame in &mut self.frames {
            unsafe {
                frame.command_pool = self
                    .interface
                    .device
                    .create_command_pool(&pool_info, None)?;
                let cmd_alloc_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(frame.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1)
                    .build();
                frame.command_buffer = self
                    .interface
                    .device
                    .allocate_command_buffers(&cmd_alloc_info)?[0];
            }
        }

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
    pub fn set_sync_structures(&mut self) -> Result<()> {
        // One fence to control when the GPU has finished rendering the frame and 2 semaphores to
        // synchronize rendering with the swapchain.
        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        for frame in &mut self.frames {
            unsafe {
                frame.render_fence = self.interface.device.create_fence(&fence_info, None)?;
                frame.swapchain_semaphore = self
                    .interface
                    .device
                    .create_semaphore(&semaphore_info, None)?;
                frame.render_semaphore = self
                    .interface
                    .device
                    .create_semaphore(&semaphore_info, None)?;
            }
        }

        // Immediate submit fence
        self.immediate_fence = unsafe { self.interface.device.create_fence(&fence_info, None)? };

        Ok(())
    }

    /// Initialize resource descriptors
    pub fn set_descriptors(&mut self) -> Result<()> {
        // Create a descriptor pool that will hold 10 sets with 1 image each
        let sizes = vec![PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 1.0)];
        self.descriptor_pool = DescriptorAllocator::new(&self.interface.device, 10, sizes)?;

        // Descriptor set layout for the gradient compute draw
        let mut layout_builder = DescriptorLayoutBuilder::default();
        layout_builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);
        self.draw_image_descriptor_layout = layout_builder.build(
            &self.interface.device,
            vk::ShaderStageFlags::COMPUTE,
            Option::<vk::DescriptorSetLayoutBindingFlagsCreateInfo>::None,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;

        // Allocate a descriptor set for the draw image
        self.draw_image_descriptors = self
            .descriptor_pool
            .allocate(&self.interface.device, self.draw_image_descriptor_layout)?;

        self.descriptor_writer.write_image(
            0,
            self.draw_image.image_view,
            vk::Sampler::null(),
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        );
        self.descriptor_writer
            .update_set(&self.interface.device, self.draw_image_descriptors);
        self.descriptor_writer.clear();

        // Descriptor set layout for the rgb->rgba compute shader
        let mut layout_builder = DescriptorLayoutBuilder::default();
        // Output rgba image
        layout_builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);
        self.rgb_to_rgba_descriptor_layout = layout_builder.build(
            &self.interface.device,
            vk::ShaderStageFlags::COMPUTE,
            Option::<vk::DescriptorSetLayoutBindingFlagsCreateInfo>::None,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;

        // Allocate a descriptor set for the rgb->rgba buffer
        self.rgb_to_rgba_descriptors = self
            .descriptor_pool
            .allocate(&self.interface.device, self.rgb_to_rgba_descriptor_layout)?;

        // Frame descriptors
        for frame in &mut self.frames {
            let frame_sizes = vec![
                PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
                PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
                PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
                PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
            ];
            frame.frame_descriptors =
                DescriptorAllocator::new(&self.interface.device, 1000, frame_sizes)?;
        }

        // Scene descriptor layout
        let mut builder = DescriptorLayoutBuilder::default();
        builder.add_binding(0, vk::DescriptorType::UNIFORM_BUFFER);
        self.scene_descriptor_layout = builder.build(
            &self.interface.device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            Option::<vk::DescriptorSetLayoutBindingFlagsCreateInfo>::None,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;

        // Texture descriptor layout
        let mut builder = DescriptorLayoutBuilder::default();
        builder.add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        self.single_image_descriptor_layout = builder.build(
            &self.interface.device,
            vk::ShaderStageFlags::FRAGMENT,
            Option::<vk::DescriptorSetLayoutBindingFlagsCreateInfo>::None,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;

        Ok(())
    }

    /// Create the compute pipelines
    pub fn set_compute_pipelines(&mut self) -> Result<()> {
        let mut builder = ComputePipelineBuilder::new(&self.interface.device);
        let set_layouts = &[self.draw_image_descriptor_layout];
        let layout = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .build();
        builder.set_layout(&layout)?;
        builder.set_shader(include_bytes!("../../shaders/spv/gradient.spv"))?;
        builder.set_shader_stage(vk::ShaderStageFlags::COMPUTE);

        self.gradient_pipeline = builder.build()?;

        // rgb to rgba conversion shader
        let mut builder = ComputePipelineBuilder::new(&self.interface.device);
        let push_constant_ranges = &[vk::PushConstantRange::builder()
            .offset(0)
            .size(size_of::<RgbToRgbaPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build()];
        let set_layouts = &[self.rgb_to_rgba_descriptor_layout];
        let layout = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(push_constant_ranges)
            .set_layouts(set_layouts)
            .build();
        builder.set_layout(&layout)?;
        builder.set_shader(include_bytes!("../../shaders/spv/rgb_to_rgba.spv"))?;
        builder.set_shader_stage(vk::ShaderStageFlags::COMPUTE);

        self.rgb_to_rgba_pipeline = builder.build()?;

        Ok(())
    }

    /// Create the mesh graphics pipeline
    pub fn set_mesh_pipeline(&mut self) -> Result<()> {
        let mut builder = GraphicsPipelineBuilder::new(&self.interface.device);
        let push_constant_ranges = &[vk::PushConstantRange::builder()
            .offset(0)
            .size(size_of::<DrawPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()];
        let set_layouts = &[self.single_image_descriptor_layout];
        let layout = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(push_constant_ranges)
            .set_layouts(set_layouts)
            .build();
        builder.set_layout(&layout)?;
        builder.set_shaders(
            // TODO: Check if these are the correct shaders (or rename)
            include_bytes!("../../shaders/spv/colored_triangle_mesh_vert.spv"),
            include_bytes!("../../shaders/spv/tex_image_frag.spv"),
        )?;
        builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        builder.set_polygon_mode(vk::PolygonMode::FILL);
        // No backface culling
        builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE);
        // No multisampling
        builder.disable_multisampling();
        // Color blending
        builder.disable_blending();
        // Depth test
        builder.enable_depthtest(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL);
        // Connect the image format we will draw into
        builder.set_color_attachment_format(self.draw_image.image_format);
        builder.set_depth_format(self.depth_image.image_format);
        builder.set_viewport_state();
        builder.set_dynamic_states(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        self.mesh_pipeline = builder.build()?;

        // Destroy texture layout
        unsafe {
            self.interface
                .device
                .destroy_descriptor_set_layout(self.single_image_descriptor_layout, None);
        }

        Ok(())
    }

    /// Initialize material pipeline
    pub fn set_metal_roughness_material_pipeline(&mut self) -> Result<()> {
        let mut builder = MetallicRoughnessBuilder::new(&self.interface.device);
        builder.set_pipelines(
            self.scene_descriptor_layout,
            self.draw_image.image_format,
            self.depth_image.image_format,
        )?;
        self.metal_rough_material = builder.build();

        Ok(())
    }

    /// Initialize samplers
    pub fn set_samplers(&mut self) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .build();
        self.default_sampler_nearest =
            unsafe { self.interface.device.create_sampler(&sampler_info, None)? };
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .build();
        self.default_sampler_linear =
            unsafe { self.interface.device.create_sampler(&sampler_info, None)? };

        Ok(())
    }

    /// Initialize GUI
    pub fn set_gui(&mut self) -> Result<()> {
        self.gui = Some(EguiIntegration::new(
            &self.interface,
            self.indices.graphics,
            self.graphics_queue,
            &self.swapchain,
            &self.window,
        )?);

        Ok(())
    }

    /// Set engine camera
    pub fn set_camera(&mut self) {
        self.camera = Some(Camera::new(
            point3(0.0, 0.5, 10.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 1.0, 0.0),
            self.swapchain.extent.width,
            self.swapchain.extent.height,
        ));
    }
}

/// Create a Vulkan instance
fn create_vulkan_instance(
    window: &Window,
    entry: &Entry,
    messenger: &mut vk::DebugUtilsMessengerEXT,
) -> Result<Instance> {
    let enginelication_info = vk::ApplicationInfo::builder()
        .application_name(b"vulkan-engine\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"vulkan-engine\0")
        .engine_version(vk::make_version(0, 0, 0))
        .api_version(vk::make_version(1, 3, 0))
        .build();
    let available_layers = unsafe {
        entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>()
    };
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
        .user_callback(Some(debug_callback))
        .build();
    if ENABLE_VALIDATION_LAYER {
        info = info.push_next(&mut debug_info);
    }
    let instance = unsafe { entry.create_instance(&info, None)? };

    if ENABLE_VALIDATION_LAYER {
        *messenger = unsafe { instance.create_debug_utils_messenger_ext(&debug_info, None)? };
    }

    Ok(instance)
}
