use super::{
    descriptors::{DescriptorAllocator, DescriptorLayoutBuilder, DescriptorWriter},
    memory::{AllocatedImage, Allocation},
    meshes::DrawPushConstants,
    pipelines::{GraphicsPipelineBuilder, Pipeline},
    Vec4, ALLOCATOR,
};
use anyhow::Result;
use cgmath::vec4;
use vulkanalia::{
    prelude::v1_3::{vk, Device, HasBuilder},
    vk::DeviceV1_0,
};

pub struct GLTFMetallicRoughnessBuilder<'d> {
    device: &'d Device,
    opaque_pipeline: Pipeline,
    transparent_pipeline: Pipeline,
    material_layout: vk::DescriptorSetLayout,
}

impl<'d> GLTFMetallicRoughnessBuilder<'d> {
    /// Initialize the builder
    #[inline]
    pub fn new(device: &'d Device) -> Self {
        Self {
            device,
            opaque_pipeline: Pipeline::default(),
            transparent_pipeline: Pipeline::default(),
            material_layout: vk::DescriptorSetLayout::default(),
        }
    }

    /// Build a [`GLTFMetallicRoughness`]
    #[inline]
    pub fn build(self) -> GLTFMetallicRoughness {
        GLTFMetallicRoughness {
            opaque_pipeline: self.opaque_pipeline,
            transparent_pipeline: self.transparent_pipeline,
            material_layout: self.material_layout,
            material_constants: MaterialConstants::default(),
            material_resources: MaterialResources::default(),
            writer: DescriptorWriter::default(),
        }
    }

    /// Create the material pipelines
    pub fn set_pipelines(
        &mut self,
        scene_descriptor_layout: vk::DescriptorSetLayout,
        draw_image_format: vk::Format,
        depth_image_format: vk::Format,
    ) -> Result<()> {
        let mut layout_builder = DescriptorLayoutBuilder::default();
        layout_builder.add_binding(0, vk::DescriptorType::UNIFORM_BUFFER);
        layout_builder.add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        layout_builder.add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        self.material_layout = layout_builder.build(
            self.device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            Option::<vk::DescriptorSetLayoutBindingFlagsCreateInfo>::None,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;

        // Same builder is used to create both the opaque and transparent pipelines
        let mut pipeline_builder = GraphicsPipelineBuilder::new(self.device);
        let mattrix_range = &[vk::PushConstantRange::builder()
            .offset(0)
            .size(size_of::<DrawPushConstants>().try_into()?)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()];
        let layouts = &[scene_descriptor_layout, self.material_layout];
        let mesh_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(mattrix_range)
            .set_layouts(layouts)
            .build();
        pipeline_builder.set_layout(&mesh_layout_info)?;
        pipeline_builder.set_shaders(
            include_bytes!("../../shaders/spv/mesh_vert.spv"),
            include_bytes!("../../shaders/spv/mesh_frag.spv"),
        )?;
        pipeline_builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        pipeline_builder.set_polygon_mode(vk::PolygonMode::FILL);
        pipeline_builder.set_cull_mode(vk::CullModeFlags::FRONT, vk::FrontFace::CLOCKWISE);
        pipeline_builder.disable_multisampling();
        pipeline_builder.disable_blending();
        pipeline_builder.enable_depthtest(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL);

        // Render format
        pipeline_builder.set_color_attachment_format(draw_image_format);
        pipeline_builder.set_depth_format(depth_image_format);
        pipeline_builder.set_viewport_state();
        pipeline_builder
            .set_dynamic_states(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        // Clone the builder as it will be reused for the transparent pipeline
        self.opaque_pipeline = pipeline_builder.clone().build()?;

        // Create the transparent pipeline
        pipeline_builder.set_shaders(
            include_bytes!("../../shaders/spv/mesh_vert.spv"),
            include_bytes!("../../shaders/spv/mesh_frag.spv"),
        )?;
        pipeline_builder.enable_blending_additive();
        pipeline_builder.enable_depthtest(vk::FALSE, vk::CompareOp::GREATER_OR_EQUAL);
        self.transparent_pipeline = pipeline_builder.build()?;

        Ok(())
    }
}

#[derive(Default, Debug, Clone)]
pub struct GLTFMetallicRoughness {
    pub opaque_pipeline: Pipeline,
    pub transparent_pipeline: Pipeline,
    pub material_layout: vk::DescriptorSetLayout,
    pub material_constants: MaterialConstants,
    pub material_resources: MaterialResources,
    pub writer: DescriptorWriter,
}

impl GLTFMetallicRoughness {
    pub fn write_material(
        &mut self,
        device: &Device,
        pass: MaterialPass,
        resources: MaterialResources,
        descriptor_allocator: &mut DescriptorAllocator,
    ) -> Result<GLTFMaterial> {
        self.material_resources = resources;
        let pipeline = if pass == MaterialPass::Transparent {
            self.transparent_pipeline
        } else {
            self.opaque_pipeline
        };
        let material_set = descriptor_allocator.allocate(device, self.material_layout)?;

        self.writer.clear();
        self.writer.write_buffer(
            0,
            self.material_resources.data_alloc.buffer,
            size_of::<MaterialConstants>().try_into()?,
            self.material_resources.data_buffer_offset,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        self.writer.write_image(
            1,
            self.material_resources.color_image.image_view,
            self.material_resources.color_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
        // NOTE: The same image (engine.white_image) is used for both the color image and the
        // metal rough image, and writing to the same image twice is invalid
        // self.writer.write_image(
        //     2,
        //     self.material_resources.metal_rough_image.image_view,
        //     self.material_resources.metal_rough_sampler,
        //     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        //     vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        // );
        self.writer.update_set(device, material_set);

        Ok(GLTFMaterial::new(MaterialInstance::new(
            pipeline,
            material_set,
            pass,
        )))
    }

    /// Cleanup the resources
    pub fn cleanup(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.material_layout, None);
            // Both pipelines share the same layout so we only destroy it once
            device.destroy_pipeline_layout(self.opaque_pipeline.layout, None);
            device.destroy_pipeline(self.opaque_pipeline.pipeline, None);
            device.destroy_pipeline(self.transparent_pipeline.pipeline, None);
        }
        self.material_resources.cleanup(device);
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct GLTFMaterial {
    pub data: MaterialInstance,
}

impl GLTFMaterial {
    /// Constructor
    #[inline]
    pub fn new(data: MaterialInstance) -> Self {
        Self { data }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct MaterialInstance {
    pub pipeline: Pipeline,
    pub material_set: vk::DescriptorSet,
    pub pass: MaterialPass,
}

impl MaterialInstance {
    /// Constructor
    #[inline]
    fn new(pipeline: Pipeline, material_set: vk::DescriptorSet, pass: MaterialPass) -> Self {
        Self {
            pipeline,
            material_set,
            pass,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MaterialConstants {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    // Padding, needed for uniform buffers
    pub extra: [Vec4; 14],
}

impl Default for MaterialConstants {
    #[inline]
    fn default() -> Self {
        Self {
            color_factors: vec4(1.0, 1.0, 1.0, 1.0),
            metal_rough_factors: vec4(1.0, 1.0, 1.0, 1.0),
            extra: [vec4(1.0, 1.0, 1.0, 1.0); 14],
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct MaterialResources {
    pub color_image: AllocatedImage,
    pub color_sampler: vk::Sampler,
    pub metal_rough_image: AllocatedImage,
    pub metal_rough_sampler: vk::Sampler,
    pub data_alloc: Allocation,
    pub data_buffer_offset: vk::DeviceSize,
}

impl MaterialResources {
    /// Constructor
    #[inline]
    pub fn new(
        color_image: AllocatedImage,
        color_sampler: vk::Sampler,
        metal_rough_image: AllocatedImage,
        metal_rough_sampler: vk::Sampler,
        data_alloc: Allocation,
        data_buffer_offset: vk::DeviceSize,
    ) -> Self {
        Self {
            color_image,
            color_sampler,
            metal_rough_image,
            metal_rough_sampler,
            data_alloc,
            data_buffer_offset,
        }
    }

    /// Cleanup the resources
    pub fn cleanup(&mut self, device: &Device) {
        ALLOCATOR.deallocate(device, &self.data_alloc);
    }
}

#[repr(u8)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialPass {
    #[default]
    MainColor,
    Transparent,
    Other,
}
