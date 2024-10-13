use anyhow::Result;
use std::ptr::from_ref;
use vulkanalia::{
    bytecode::Bytecode,
    prelude::v1_3::{vk, Device, HasBuilder},
    vk::{DeviceV1_0, Handle},
};

/// Wrapper around a [`vk::Pipeline`] and the associated [`vk::PipelineLayout`]
#[derive(Default, Debug, Clone, Copy)]
pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl Pipeline {
    /// Constructor
    #[inline]
    pub fn new(pipeline: vk::Pipeline, layout: vk::PipelineLayout) -> Self {
        Self { pipeline, layout }
    }

    /// Cleanup resources
    pub fn cleanup(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Graphics [`vk::Pipeline`] builder
#[derive(Clone)]
pub struct GraphicsPipelineBuilder<'d> {
    device: &'d Device,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    dynamic_states: Vec<vk::DynamicState>,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    multisampling: vk::PipelineMultisampleStateCreateInfo,
    layout: vk::PipelineLayout,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo,
    rendering_info: vk::PipelineRenderingCreateInfo,
    color_attachment_format: vk::Format,
    viewport_state: vk::PipelineViewportStateCreateInfo,
}

impl<'d> GraphicsPipelineBuilder<'d> {
    /// Initialize the builder
    #[inline]
    pub fn new(device: &'d Device) -> Self {
        Self {
            device,
            shader_stages: Vec::new(),
            dynamic_states: Vec::new(),
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::default(),
            rasterizer: vk::PipelineRasterizationStateCreateInfo::default(),
            color_blend_attachment: vk::PipelineColorBlendAttachmentState::default(),
            multisampling: vk::PipelineMultisampleStateCreateInfo::default(),
            layout: vk::PipelineLayout::default(),
            depth_stencil: vk::PipelineDepthStencilStateCreateInfo::default(),
            rendering_info: vk::PipelineRenderingCreateInfo::default(),
            color_attachment_format: vk::Format::default(),
            viewport_state: vk::PipelineViewportStateCreateInfo::default(),
        }
    }

    /// Build a graphics [`vk::Pipeline`]
    pub fn build(mut self) -> Result<Pipeline> {
        // Connect the format to the render info structure
        self.rendering_info.color_attachment_formats = from_ref(&self.color_attachment_format);
        // Setup dummy color blending. We aren't using transparent objects yet.
        // No blending, but we do write to the color attachment
        let color_blend_attachments = &[self.color_blend_attachment];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachment_count(1)
            .attachments(color_blend_attachments)
            .build();

        // Vertex input is not needed as we're doing vertex pulling (passing the buffer device
        // address to the shader so that it can access it directly)
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&self.dynamic_states)
            .build();

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stage_count(self.shader_stages.len().try_into()?)
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&self.viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.layout)
            .dynamic_state(&dynamic_state)
            .push_next(&mut self.rendering_info)
            .build();
        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
                .0[0]
        };

        Ok(Pipeline::new(pipeline, self.layout))
    }

    /// Set pipeline layout
    pub fn set_layout(&mut self, layout_info: &vk::PipelineLayoutCreateInfo) -> Result<()> {
        self.layout = unsafe { self.device.create_pipeline_layout(layout_info, None)? };

        Ok(())
    }

    /// Initialize shaders
    pub fn set_shaders(&mut self, vertex_shader: &[u8], fragment_shader: &[u8]) -> Result<()> {
        self.shader_stages.clear();
        let vertex_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(create_shader_module(self.device, vertex_shader)?)
            .name(b"main\0")
            .build();
        let frag_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(create_shader_module(self.device, fragment_shader)?)
            .name(b"main\0")
            .build();
        self.shader_stages.push(vertex_info);
        self.shader_stages.push(frag_info);

        Ok(())
    }

    pub fn set_input_topology(&mut self, topology: vk::PrimitiveTopology) {
        self.input_assembly.topology = topology;
        self.input_assembly.primitive_restart_enable = vk::FALSE;
    }

    pub fn set_polygon_mode(&mut self, mode: vk::PolygonMode) {
        self.rasterizer.polygon_mode = mode;
        self.rasterizer.line_width = 1.0;
    }

    pub fn set_cull_mode(&mut self, cull_mode: vk::CullModeFlags, front_face: vk::FrontFace) {
        self.rasterizer.cull_mode = cull_mode;
        self.rasterizer.front_face = front_face;
    }

    /// No multisampling (1 sample per pixel)
    pub fn disable_multisampling(&mut self) {
        self.multisampling.sample_shading_enable = vk::FALSE;
        self.multisampling.rasterization_samples = vk::SampleCountFlags::_1;
        self.multisampling.min_sample_shading = 1.0;
        self.multisampling.alpha_to_coverage_enable = vk::FALSE;
        self.multisampling.alpha_to_one_enable = vk::FALSE;
    }

    #[allow(dead_code)]
    /// Enable additive blending
    ///
    /// `outColor = srcColor.rgb * srcColor.a + dstColor.rgb * 1.0`
    pub fn enable_blending_additive(&mut self) {
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        self.color_blend_attachment.blend_enable = vk::TRUE;
        self.color_blend_attachment.src_color_blend_factor = vk::BlendFactor::SRC_ALPHA;
        self.color_blend_attachment.dst_color_blend_factor = vk::BlendFactor::ONE;
        self.color_blend_attachment.color_blend_op = vk::BlendOp::ADD;
        self.color_blend_attachment.src_alpha_blend_factor = vk::BlendFactor::ONE;
        self.color_blend_attachment.dst_alpha_blend_factor = vk::BlendFactor::ZERO;
        self.color_blend_attachment.alpha_blend_op = vk::BlendOp::ADD;
    }

    #[allow(dead_code)]
    /// Enable alphablend blending
    ///
    /// `outColor = srcColor.rgb * srcColor.a + dstColor.rgb * (1.0 - srcColor.a)`
    pub fn enable_blending_alphablend(&mut self) {
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        self.color_blend_attachment.blend_enable = vk::TRUE;
        self.color_blend_attachment.src_color_blend_factor = vk::BlendFactor::SRC_ALPHA;
        self.color_blend_attachment.dst_color_blend_factor = vk::BlendFactor::ONE_MINUS_SRC_ALPHA;
        self.color_blend_attachment.color_blend_op = vk::BlendOp::ADD;
        self.color_blend_attachment.src_alpha_blend_factor = vk::BlendFactor::ONE;
        self.color_blend_attachment.dst_alpha_blend_factor = vk::BlendFactor::ZERO;
        self.color_blend_attachment.alpha_blend_op = vk::BlendOp::ADD;
    }

    #[allow(dead_code)]
    pub fn disable_blending(&mut self) {
        // Default write mask
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        // No blending
        self.color_blend_attachment.blend_enable = vk::FALSE;
    }

    pub fn set_color_attachment_format(&mut self, format: vk::Format) {
        self.color_attachment_format = format;
        self.rendering_info.color_attachment_count = 1;
    }

    pub fn set_depth_format(&mut self, format: vk::Format) {
        self.rendering_info.depth_attachment_format = format;
    }

    #[allow(dead_code)]
    pub fn enable_depthtest(
        &mut self,
        depth_write_enable: vk::Bool32,
        depth_compare_op: vk::CompareOp,
    ) {
        self.depth_stencil.depth_test_enable = vk::TRUE;
        self.depth_stencil.depth_write_enable = depth_write_enable;
        self.depth_stencil.depth_compare_op = depth_compare_op;
        self.depth_stencil.depth_bounds_test_enable = vk::FALSE;
        self.depth_stencil.stencil_test_enable = vk::FALSE;
        self.depth_stencil.min_depth_bounds = 0.0;
        self.depth_stencil.max_depth_bounds = 1.0;
    }

    #[allow(dead_code)]
    pub fn disable_depthtest(&mut self) {
        self.depth_stencil.depth_test_enable = vk::FALSE;
        self.depth_stencil.depth_write_enable = vk::FALSE;
        self.depth_stencil.depth_compare_op = vk::CompareOp::NEVER;
        self.depth_stencil.stencil_test_enable = vk::FALSE;
        self.depth_stencil.min_depth_bounds = 0.0;
        self.depth_stencil.max_depth_bounds = 1.0;
    }

    pub fn set_viewport_state(&mut self) {
        self.viewport_state.viewport_count = 1;
        self.viewport_state.scissor_count = 1;
    }

    // Dynamic states
    pub fn set_dynamic_states(&mut self, states: Vec<vk::DynamicState>) {
        self.dynamic_states = states;
    }
}

impl Drop for GraphicsPipelineBuilder<'_> {
    fn drop(&mut self) {
        for shader_stage in self.shader_stages.drain(..) {
            unsafe {
                self.device.destroy_shader_module(shader_stage.module, None);
            }
        }
    }
}

/// Compute [`vk::Pipeline`] builder
pub struct ComputePipelineBuilder<'d> {
    device: &'d Device,
    layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    shader_create_info: vk::PipelineShaderStageCreateInfo,
}

impl<'d> ComputePipelineBuilder<'d> {
    /// Initialize the builder
    #[inline]
    pub fn new(device: &'d Device) -> Self {
        Self {
            device,
            layout: vk::PipelineLayout::default(),
            shader_module: vk::ShaderModule::default(),
            shader_create_info: vk::PipelineShaderStageCreateInfo::default(),
        }
    }

    /// Build a compute [`vk::Pipeline`]
    pub fn build(self) -> Result<Pipeline> {
        let compute_info = vk::ComputePipelineCreateInfo::builder()
            .layout(self.layout)
            .stage(self.shader_create_info)
            .build();
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_info], None)?
                .0[0]
        };

        Ok(Pipeline::new(pipeline, self.layout))
    }

    /// Set descriptor layouts
    pub fn set_descriptor_layouts(&mut self, layouts: &[vk::DescriptorSetLayout]) -> Result<()> {
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(layouts)
            .build();
        self.layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        Ok(())
    }

    /// Set shader stage data
    pub fn set_shader_stage(&mut self, stage_flags: vk::ShaderStageFlags) {
        self.shader_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(self.shader_module)
            .stage(stage_flags)
            .name(b"main\0")
            .build();
    }

    /// Set the [`vk::ShaderModule`] from raw bytes
    pub fn set_shader(&mut self, bytes: &[u8]) -> Result<()> {
        self.shader_module = create_shader_module(self.device, bytes)?;

        Ok(())
    }
}

impl Drop for ComputePipelineBuilder<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.shader_module, None);
        }
    }
}

fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode)?;
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code())
        .build();

    Ok(unsafe { device.create_shader_module(&info, None)? })
}
