use super::{
    ENABLE_VALIDATION_LAYER, MAX_FRAMES_IN_FLIGHT, PORTABILITY_MACOS_VERSION, VALIDATION_LAYER,
};
use crate::{
    buffer::{BufferAllocation, UniformBufferObject},
    device::{check_physical_device, get_max_msaa_samples, QueueFamilyIndices, DEVICE_EXTENSIONS},
    image::{create_image, create_image_view, get_depth_format},
    mesh::{loader::MeshLoader, DrawPushConstants, MeshAsset},
    swapchain::{
        get_swapchain_extent, get_swapchain_present_mode, get_swapchain_surface_format,
        SwapchainSupport,
    },
};
use anyhow::{anyhow, Result};
use hashbrown::HashSet;
use log::{debug, error, info, trace, warn};
use std::{ffi::CStr, mem::size_of, os::raw::c_void};
use vulkanalia::{
    bytecode::Bytecode,
    prelude::v1_3::{
        vk::{self, ExtDebugUtilsExtension, Handle, KhrSwapchainExtension},
        Device, DeviceV1_0, Entry, EntryV1_0, HasBuilder, Instance, InstanceV1_0,
    },
    window as vk_window,
};
use winit::window::Window;

/// Logger callback in use when validation layer is enabled
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

/// Vulkan handles and associated properties used by the Vulkan [engine](super::Engine)
#[derive(Default)]
pub struct EngineData {
    // Debug
    pub(super) messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    pub surface: vk::SurfaceKHR,
    // Physical / Logical device
    pub physical_device: vk::PhysicalDevice,
    pub msaa_samples: vk::SampleCountFlags,
    // Queues
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family_index: u32,
    pub present_queue: vk::Queue,
    pub present_queue_family_index: u32,
    // Swapchain
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    // Graphics pipeline
    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub graphics_pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    // Framebuffers
    pub framebuffers: Vec<vk::Framebuffer>,
    // Color image
    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,
    // Depth
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    // Meshes
    pub mesh_loader: MeshLoader,
    pub meshes: Vec<MeshAsset>,
    // Texture
    pub mip_levels: u32,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    // Buffers
    pub vertex_buffer: BufferAllocation,
    pub index_buffer: BufferAllocation,
    pub uniform_buffers: Vec<BufferAllocation>,
    // Descriptors
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    // Command buffers
    pub command_pool: vk::CommandPool,
    pub framebuffers_command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    // Sync objects
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}

impl EngineData {
    /// Create a Vulkan instance
    pub unsafe fn create_instance(&mut self, window: &Window, entry: &Entry) -> Result<Instance> {
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
            self.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
        }

        Ok(instance)
    }
}

/// Surface methods
impl EngineData {
    /// Create a surface
    pub unsafe fn create_surface(
        &mut self,
        instance: &Instance,
        display: &Window,
        window: &Window,
    ) -> Result<()> {
        self.surface = vk_window::create_surface(instance, &display, &window)?;
        Ok(())
    }
}

/// Device methods
impl EngineData {
    /// Select a physical device matching the requirements
    pub unsafe fn pick_physical_device(&mut self, instance: &Instance) -> Result<()> {
        for physical_device in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical_device);
            if let Err(err) = check_physical_device(instance, physical_device, self.surface) {
                warn!(
                    "Skipping physical device (`{}`): {err}",
                    properties.device_name
                );
            } else {
                info!("Selected physical device (`{}`)", properties.device_name);
                self.physical_device = physical_device;
                self.msaa_samples = get_max_msaa_samples(instance, self.physical_device);
                return Ok(());
            }
        }

        Err(anyhow!("Failed to find suitable physical device"))
    }

    /// Create logical device
    pub unsafe fn create_logical_device(
        &mut self,
        entry: &Entry,
        instance: &Instance,
    ) -> Result<Device> {
        let indices = QueueFamilyIndices::get(instance, self.physical_device, self.surface)?;
        let mut unique_indices = HashSet::new();
        unique_indices.insert(indices.graphics);
        unique_indices.insert(indices.present);
        let queue_priorities = &[1.0];
        let queue_infos = unique_indices
            .into_iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();

        let layers = if ENABLE_VALIDATION_LAYER {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        let mut extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();
        if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
        }

        let features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .sample_rate_shading(true);
        let mut features_12 =
            vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);
        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut features_12);
        let device = instance.create_device(self.physical_device, &info, None)?;

        self.graphics_queue = device.get_device_queue(indices.graphics, 0);
        self.graphics_queue_family_index = indices.graphics;
        self.present_queue = device.get_device_queue(indices.present, 0);
        self.present_queue_family_index = indices.present;

        Ok(device)
    }
}

// Framebuffers methods
impl EngineData {
    /// Create framebuffers
    pub unsafe fn create_framebuffers(&mut self, device: &Device) -> Result<()> {
        self.framebuffers = self
            .swapchain_image_views
            .iter()
            .map(|i| {
                let attachments = &[self.color_image_view, self.depth_image_view, *i];
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(self.render_pass)
                    .attachments(attachments)
                    .width(self.swapchain_extent.width)
                    .height(self.swapchain_extent.height)
                    .layers(1);

                device.create_framebuffer(&create_info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
}

// Synchronization objects methods
impl EngineData {
    /// Create semaphores and fences
    pub unsafe fn create_sync_objects(&mut self, device: &Device) -> Result<()> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            self.image_available_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            self.render_finished_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            self.in_flight_fences
                .push(device.create_fence(&fence_info, None)?);
        }

        self.images_in_flight = self
            .swapchain_images
            .iter()
            .map(|_| vk::Fence::null())
            .collect();

        Ok(())
    }
}

// Swapchain methods
impl EngineData {
    /// Create swapchain
    pub unsafe fn create_swapchain(
        &mut self,
        window: &Window,
        instance: &Instance,
        device: &Device,
    ) -> Result<()> {
        let indices = QueueFamilyIndices::get(instance, self.physical_device, self.surface)?;
        let support = SwapchainSupport::get(instance, self.physical_device, self.surface)?;

        let surface_format = get_swapchain_surface_format(&support.formats);
        let present_mode = get_swapchain_present_mode(&support.present_modes);
        let extent = get_swapchain_extent(window, support.capabilities);

        self.swapchain_format = surface_format.format;
        self.swapchain_extent = extent;

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0
            && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }
        let mut queue_family_indices = Vec::new();
        let image_sharing_mode = if indices.graphics != indices.present {
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.present);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            // NOTE - https://kylemayes.github.io/vulkanalia/swapchain/recreation.html#recreating-the-swapchain
            // It is possible to create a new swapchain while drawing commands on an image from the old swapchain are still in-flight.
            // You need to pass the previous swapchain to the old_swapchain field in the vk::SwapchainCreateInfoKHR struct
            // and destroy the old swapchain as soon as you've finished using it.
            .old_swapchain(vk::SwapchainKHR::null());

        self.swapchain = device.create_swapchain_khr(&info, None)?;
        self.swapchain_images = device.get_swapchain_images_khr(self.swapchain)?;

        Ok(())
    }

    /// Create swapchain image views
    pub unsafe fn create_swapchain_image_views(&mut self, device: &Device) -> Result<()> {
        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|i| {
                create_image_view(
                    device,
                    *i,
                    self.swapchain_format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }
}

/// Create shader module
unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode)?;
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}

/// Graphics pipeline methods
impl EngineData {
    /// Create graphics pipeline
    pub unsafe fn create_graphics_pipeline(&mut self, device: &Device) -> Result<()> {
        // Stages
        let vert = include_bytes!("../../shaders/vert.spv");
        let frag = include_bytes!("../../shaders/frag.spv");

        let vert_shader_module = create_shader_module(device, &vert[..])?;
        let frag_shader_module = create_shader_module(device, &frag[..])?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");
        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");

        // Vertex input state
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

        // Input assembly state
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport state
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.swapchain_extent);
        let (viewports, scissors) = (&[viewport], &[scissor]);
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        // Rasterization state
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        // Multisample state
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(self.msaa_samples)
            // Minimum fraction for sample shading; closer to one is smoother.
            .min_sample_shading(0.2)
            .rasterization_samples(self.msaa_samples);

        // Depth stencil state
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0) // Optional
            .max_depth_bounds(1.0) // Optional
            .stencil_test_enable(false);

        // Color blend state
        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // Unused for now
        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];
        _ = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

        let vert_push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<DrawPushConstants>() as u32);

        // Layout
        let set_layouts = &[self.descriptor_set_layout];
        let push_constant_ranges = &[vert_push_constant_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);
        self.graphics_pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        // Create
        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .layout(self.graphics_pipeline_layout)
            .render_pass(self.render_pass)
            .base_pipeline_handle(vk::Pipeline::null()) // Optional
            .base_pipeline_index(-1) // Optional
            .subpass(0);

        self.graphics_pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
            .0[0];

        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);

        Ok(())
    }

    /// Create render pass
    pub unsafe fn create_render_pass(
        &mut self,
        instance: &Instance,
        device: &Device,
    ) -> Result<()> {
        // Attachments
        let color_attachment = vk::AttachmentDescription::builder()
            .format(self.swapchain_format)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_stencil_attachment = vk::AttachmentDescription::builder()
            .format(get_depth_format(instance, self.physical_device)?)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let color_resolve_attachment = vk::AttachmentDescription::builder()
            .format(self.swapchain_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        // Subpasses
        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_resolve_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachments = &[color_attachment_ref];
        let resolve_attachments = &[color_resolve_attachment_ref];

        let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments)
            .depth_stencil_attachment(&depth_stencil_attachment_ref)
            .resolve_attachments(resolve_attachments);

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let (attachments, subpasses, dependencies) = (
            &[
                color_attachment,
                depth_stencil_attachment,
                color_resolve_attachment,
            ],
            &[subpass],
            &[dependency],
        );
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        self.render_pass = device.create_render_pass(&info, None)?;

        Ok(())
    }
}

// Image methods
impl EngineData {
    /// Create depth objects
    pub unsafe fn create_depth_objects(
        &mut self,
        instance: &Instance,
        device: &Device,
    ) -> Result<()> {
        let format = get_depth_format(instance, self.physical_device)?;
        (self.depth_image, self.depth_image_memory) = create_image(
            instance,
            device,
            self.physical_device,
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            1,
            self.msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.depth_image_view = create_image_view(
            device,
            self.depth_image,
            format,
            vk::ImageAspectFlags::DEPTH,
            1,
        )?;

        Ok(())
    }

    /// Create color objects
    pub unsafe fn create_color_objects(
        &mut self,
        instance: &Instance,
        device: &Device,
    ) -> Result<()> {
        (self.color_image, self.color_image_memory) = create_image(
            instance,
            device,
            self.physical_device,
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            1,
            self.msaa_samples,
            self.swapchain_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.color_image_view = create_image_view(
            device,
            self.color_image,
            self.swapchain_format,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        Ok(())
    }
}

// Uniform buffers methods
impl EngineData {
    /// Create uniform buffers
    pub unsafe fn create_uniform_buffers(
        &mut self,
        instance: &Instance,
        device: &Device,
    ) -> Result<()> {
        self.uniform_buffers.clear();

        for _ in 0..self.swapchain_images.len() {
            let uniform_buffer = BufferAllocation::new(
                instance,
                device,
                self.physical_device,
                size_of::<UniformBufferObject>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;
            self.uniform_buffers.push(uniform_buffer);
        }

        Ok(())
    }
}

// Command buffers methods
impl EngineData {
    /// Create command buffers
    pub unsafe fn create_command_buffers(&mut self, device: &Device) -> Result<()> {
        for image_index in 0..self.swapchain_images.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.framebuffers_command_pools[image_index])
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
            self.command_buffers.push(command_buffer);
        }

        self.secondary_command_buffers = vec![vec![]; self.swapchain_images.len()];

        Ok(())
    }
}

// Resource descriptors methods
impl EngineData {
    /// Create descriptor set layout
    pub unsafe fn create_descriptor_set_layout(&mut self, device: &Device) -> Result<()> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo_binding, sampler_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
        self.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

        Ok(())
    }

    /// Create descriptor sets
    pub unsafe fn create_descriptor_sets(&mut self, device: &Device) -> Result<()> {
        let layouts = vec![self.descriptor_set_layout; self.swapchain_images.len()];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        self.descriptor_sets = device.allocate_descriptor_sets(&info)?;

        for i in 0..self.swapchain_images.len() {
            let info = vk::DescriptorBufferInfo::builder()
                .buffer(self.uniform_buffers[i].buffer)
                .offset(0)
                // Could use vk::WHOLE_SIZE here
                .range(size_of::<UniformBufferObject>() as vk::DeviceSize);
            let buffer_info = &[info];
            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);
            let info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.texture_image_view)
                .sampler(self.texture_sampler);
            let image_info = &[info];
            let sampler_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info);

            device.update_descriptor_sets(
                &[ubo_write, sampler_write],
                &[] as &[vk::CopyDescriptorSet],
            );
        }

        Ok(())
    }

    /// Create descriptor pool
    pub unsafe fn create_descriptor_pool(&mut self, device: &Device) -> Result<()> {
        let image_count = self.swapchain_images.len() as u32;
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

        self.descriptor_pool = device.create_descriptor_pool(&info, None)?;

        Ok(())
    }
}
