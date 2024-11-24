mod loader;

use crate::engine::{
    descriptors::DescriptorAllocator,
    material::Material,
    meshes::{DrawContext, MeshAsset, RenderObject, Renderable},
    Mat4,
};
use anyhow::Result;
use cgmath::SquareMatrix;
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
};
use vulkanalia::{prelude::v1_3::vk, vk::DeviceV1_0, Device};

use super::{
    memory::{AllocatedImage, Allocation},
    ALLOCATOR,
};

/// gLTF scene graph
pub struct Scene {
    // Storage for all the data on a given gLTF file
    meshes: HashMap<String, Rc<RefCell<MeshAsset>>>,
    nodes: HashMap<String, Rc<RefCell<Node>>>,
    // Nodes that don't have a parent, for iterating through the file in tree order
    top_nodes: Vec<Rc<RefCell<Node>>>,
    materials: HashMap<String, Rc<Material>>,
    material_alloc: Allocation,
    images: Vec<AllocatedImage>,
    samplers: Vec<vk::Sampler>,
    descriptor_pool: DescriptorAllocator,
}

impl Scene {
    /// Constructor
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn new(
        meshes: HashMap<String, Rc<RefCell<MeshAsset>>>,
        nodes: HashMap<String, Rc<RefCell<Node>>>,
        top_nodes: Vec<Rc<RefCell<Node>>>,
        materials: HashMap<String, Rc<Material>>,
        material_alloc: Allocation,
        images: Vec<AllocatedImage>,
        samplers: Vec<vk::Sampler>,
        descriptor_pool: DescriptorAllocator,
    ) -> Self {
        Self {
            meshes,
            nodes,
            top_nodes,
            material_alloc,
            materials,
            images,
            samplers,
            descriptor_pool,
        }
    }

    // Destroy resources
    pub(crate) fn destroy(&mut self, device: &Device) {
        // Destroy nodes
        for (_, node) in self.nodes.drain() {
            node.borrow().destroy(device);
        }
        // Destroy pools
        self.descriptor_pool.destroy_pools(device);
        // Destroy shared material buffer
        ALLOCATOR.deallocate(device, &self.material_alloc);
        // Destroy texture images
        for image in self.images.drain(..) {
            ALLOCATOR.deallocate_image(device, image);
        }
        // Destroy samplers
        for sampler in self.samplers.drain(..) {
            unsafe {
                device.destroy_sampler(sampler, None);
            }
        }
    }
}

impl Renderable for Scene {
    /// Create renderables from the scene nodes
    fn draw(&self, top_matrix: Mat4, context: &mut DrawContext) -> Result<()> {
        for node in &self.top_nodes {
            node.borrow().draw(top_matrix, context)?;
        }

        Ok(())
    }
}

// TODO: Bench Rc<RefCell<...>>. Check (unsafe) doubly linked list implementation https://rust-unofficial.github.io/too-many-lists/
// Use vector/index based graph representations https://smallcultfollowing.com/babysteps/blog/2015/04/06/modeling-graphs-in-rust-using-vector-indices/
// (generational arena)
/// A gltf scene node
#[derive(Debug, Clone)]
pub struct Node {
    pub parent: Option<Weak<RefCell<Node>>>,
    pub children: Vec<Rc<RefCell<Node>>>,
    pub local_transform: Mat4,
    pub world_transform: Mat4,
    pub data: NodeData,
    destroyed: RefCell<bool>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            local_transform: Mat4::identity(),
            world_transform: Mat4::identity(),
            data: NodeData::Raw,
            destroyed: RefCell::new(false),
        }
    }
}

impl Node {
    /// Update transform matrix
    pub fn refresh_transform(&mut self, parent_matrix: &Mat4) {
        self.world_transform = parent_matrix * self.local_transform;
        for child in &self.children {
            child.borrow_mut().refresh_transform(&self.world_transform);
        }
    }

    /// Destroy resources
    pub fn destroy(&self, device: &Device) {
        if !*self.destroyed.borrow() {
            if let NodeData::Mesh(mesh) = &self.data {
                mesh.borrow_mut().destroy(device)
            }
        }
        *self.destroyed.borrow_mut() = true;
    }
}

impl Renderable for Node {
    fn draw(&self, top_matrix: Mat4, context: &mut DrawContext) -> Result<()> {
        match &self.data {
            NodeData::Mesh(mesh) => {
                let mesh = mesh.borrow();
                let node_matrix = top_matrix * self.world_transform;
                for surface in &mesh.surfaces {
                    let render_object = RenderObject::new(
                        surface.count,
                        surface.start_index,
                        mesh.index_alloc.buffer,
                        surface.material.data,
                        node_matrix,
                        mesh.vertex_alloc.device_address()?,
                    );
                    context.opaque_surfaces.push(render_object);
                }
            }
            NodeData::Raw => {}
        }

        for child in &self.children {
            child.borrow().draw(top_matrix, context)?;
        }

        Ok(())
    }
}

/// Specific data of a scene [`Node`]
#[derive(Debug, Clone, Default)]
pub enum NodeData {
    /// A mesh node
    Mesh(Rc<RefCell<MeshAsset>>),
    /// A node without any associated data
    #[default]
    Raw,
}
