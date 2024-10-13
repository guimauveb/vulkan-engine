use super::{material::MaterialInstance, Mat4, Vec4};
use anyhow::Result;
use cgmath::{vec4, SquareMatrix};
use std::rc::{Rc, Weak};
use vulkanalia::prelude::v1_3::vk;

#[derive(Debug, Clone, Copy)]
pub struct Scene {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_proj: Mat4,
    pub ambient_color: Vec4,
    pub sunlight_direction: Vec4,
    pub sunlight_color: Vec4,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            view: Mat4::identity(),
            projection: Mat4::identity(),
            view_proj: Mat4::identity(),
            ambient_color: vec4(0.0, 0.0, 0.0, 1.0),
            sunlight_direction: vec4(0.0, 0.0, 0.0, 1.0),
            sunlight_color: vec4(0.0, 0.0, 0.0, 1.0),
        }
    }
}

/// Dynamically renderable object
pub trait Renderable {
    fn draw(&self, top_matrix: Mat4, context: &mut DrawContext) -> Result<()>;
}

/// Drawable scene node
#[derive(Debug, Clone)]
pub struct Node {
    pub parent: Option<Weak<Node>>,
    pub children: Vec<Rc<Node>>,
    pub local_transform: Mat4,
    pub world_transform: Mat4,
}

impl Default for Node {
    #[inline]
    fn default() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            local_transform: Mat4::identity(),
            world_transform: Mat4::identity(),
        }
    }
}

impl Node {
    /// Constructor
    #[inline]
    pub fn new(
        parent: Option<Weak<Self>>,
        children: Vec<Rc<Self>>,
        local_transform: Mat4,
        world_transform: Mat4,
    ) -> Self {
        Self {
            parent,
            children,
            local_transform,
            world_transform,
        }
    }

    pub fn refresh_transform(&self, parent_matrix: &Mat4) {
        let world_transform = parent_matrix * self.local_transform;
        for child in &self.children {
            child.refresh_transform(&world_transform);
        }
    }
}

impl Renderable for Node {
    fn draw(&self, top_matrix: Mat4, context: &mut DrawContext) -> Result<()> {
        for child in &self.children {
            child.draw(top_matrix, context)?;
        }

        Ok(())
    }
}

/// Object to render
#[derive(Debug, Clone)]
pub struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub material: MaterialInstance,
    pub transform: Mat4,
    pub vertex_buffer_address: vk::DeviceAddress,
}

impl RenderObject {
    /// Constructor
    #[inline]
    pub fn new(
        index_count: u32,
        first_index: u32,
        index_buffer: vk::Buffer,
        material: MaterialInstance,
        transform: Mat4,
        vertex_buffer_address: vk::DeviceAddress,
    ) -> Self {
        Self {
            index_count,
            first_index,
            index_buffer,
            material,
            transform,
            vertex_buffer_address,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct DrawContext {
    pub opaque_surfaces: Vec<RenderObject>,
}
