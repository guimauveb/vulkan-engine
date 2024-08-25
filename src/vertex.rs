use super::{Vec2, Vec3};
use std::{
    hash::{Hash, Hasher},
    mem::size_of,
};
use vulkanalia::prelude::v1_3::{vk, HasBuilder};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub color: Vec3,
    pub tex_coord: Vec2,
    pub normal: Vec3,
}

impl Vertex {
    pub const fn new(position: Vec3, color: Vec3, tex_coord: Vec2, normal: Vec3) -> Self {
        Self {
            position,
            color,
            tex_coord,
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
            && self.color == other.color
            && self.tex_coord == other.tex_coord
            && self.normal == other.normal
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position[0].to_bits().hash(state);
        self.position[1].to_bits().hash(state);
        self.position[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
    }
}
