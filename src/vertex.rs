use super::{Vec2, Vec3, Vec4};
use std::{
    hash::{Hash, Hasher},
    mem::size_of,
};
use vulkanalia::prelude::v1_3::{vk, HasBuilder};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub uv_x: f32,
    pub normal: Vec3,
    pub uv_y: f32,
    pub color: Vec4,
}

impl Vertex {
    pub fn new(position: Vec3, color: Vec4, tex_coord: Vec2, normal: Vec3) -> Self {
        Self {
            position,
            uv_x: tex_coord[0],
            color,
            uv_y: tex_coord[1],
            normal,
        }
    }

    pub fn binding_description() -> impl vk::Cast<Target = vk::VertexInputBindingDescription> {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions(
    ) -> [impl vk::Cast<Target = vk::VertexInputAttributeDescription>; 4] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0);
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32);
        let text_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32);
        let normal = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>() + size_of::<Vec2>()) as u32);

        [pos, color, text_coord, normal]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
            && self.uv_x == other.uv_x
            && self.normal == other.normal
            && self.uv_y == other.uv_y
            && self.color == other.color
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position[0].to_bits().hash(state);
        self.position[1].to_bits().hash(state);
        self.position[2].to_bits().hash(state);
        self.uv_x.to_bits().hash(state);
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
        self.uv_y.to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.color[3].to_bits().hash(state);
    }
}
