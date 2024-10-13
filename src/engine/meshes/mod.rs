pub mod gltf;
pub mod obj;

use super::{
    material::GLTFMaterial,
    memory::Allocation,
    scene::{DrawContext, Node, RenderObject, Renderable},
    Engine, Mat4, Vec2, Vec3, Vec4, ALLOCATOR,
};
use anyhow::Result;
use log::debug;
use std::{
    hash::{Hash, Hasher},
    path::Path,
    ptr::copy_nonoverlapping as memcpy,
    rc::Rc,
};
use vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0, HasBuilder};

impl Engine {
    /// Load default meshes
    pub fn load_default_meshes(&mut self) -> Result<()> {
        debug!("Loading default meshes");
        self.load_gltf(Path::new("./assets/basicmesh.glb"))?;

        Ok(())
    }

    /// Upload a mesh to the engine
    pub fn upload_mesh(&mut self, mesh: RawMesh) -> Result<()> {
        let (index_buffer_size, vertex_buffer_size) = (
            size_of_val(&mesh.indices[..]) as vk::DeviceSize,
            size_of_val(&mesh.vertices[..]) as vk::DeviceSize,
        );
        let mesh_data_size = vertex_buffer_size + index_buffer_size;

        let vertex_info = vk::BufferCreateInfo::builder()
            .size(vertex_buffer_size)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .build();
        let vertex_alloc = ALLOCATOR.allocate_buffer(
            &self.interface,
            vertex_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let index_info = vk::BufferCreateInfo::builder()
            .size(index_buffer_size)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .build();
        let index_alloc = ALLOCATOR.allocate_buffer(
            &self.interface,
            index_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Use a staging buffer to copy the data into GPU only accessible memory for better
        // performance
        let staging_info = vk::BufferCreateInfo::builder()
            .size(mesh_data_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .build();
        let staging_alloc = ALLOCATOR.allocate_buffer(
            &self.interface,
            staging_info,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        debug_assert!(
            staging_alloc.size >= mesh_data_size,
            "Staging buffer too small for mesh data: {mesh_data_size} > {}",
            staging_alloc.size
        );
        let mut memory: *mut Vertex = staging_alloc
            .get_mapped_memory(&self.interface.device)?
            .cast();
        unsafe {
            // Copy vertex buffer
            memcpy(mesh.vertices.as_ptr(), memory, mesh.vertices.len());
            memory = memory.add(mesh.vertices.len());
            // Copy index buffer
            memcpy(mesh.indices.as_ptr(), memory.cast(), mesh.indices.len());
        }
        staging_alloc.unmap_memory(&self.interface.device);

        let copy = |cmd_buffer: vk::CommandBuffer| {
            let vertex_copy = vk::BufferCopy::builder().size(vertex_buffer_size).build();
            unsafe {
                self.interface.device.cmd_copy_buffer(
                    cmd_buffer,
                    staging_alloc.buffer,
                    vertex_alloc.buffer,
                    &[vertex_copy],
                );
            }
            let index_copy = vk::BufferCopy::builder()
                .src_offset(vertex_buffer_size)
                .size(index_buffer_size)
                .build();
            unsafe {
                self.interface.device.cmd_copy_buffer(
                    cmd_buffer,
                    staging_alloc.buffer,
                    index_alloc.buffer,
                    &[index_copy],
                );
            }

            Ok(())
        };
        self.immediate_submit
            .execute(&self.interface.device, copy)?;
        ALLOCATOR.deallocate(&self.interface.device, &staging_alloc);

        let mut mesh = MeshAsset::new(mesh.name, mesh.surfaces, index_alloc, vertex_alloc);
        for surface in &mut mesh.surfaces {
            surface.material = self.default_material.into();
        }
        let mesh_node = MeshNode::new(Node::default(), mesh.into());
        self.loaded_nodes
            .insert(mesh_node.mesh.name.clone(), mesh_node.into());

        Ok(())
    }
}

/// Push constants for mesh object draws
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DrawPushConstants {
    pub world_matrix: Mat4,
    pub vertex_buffer: vk::DeviceAddress,
}

impl DrawPushConstants {
    #[inline]
    pub fn new(world_matrix: Mat4, vertex_buffer: vk::DeviceAddress) -> Self {
        Self {
            world_matrix,
            vertex_buffer,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeshNode {
    pub node: Node,
    /// Using an [`Rc`] as we can have multiple nodes refering to the same mesh
    pub mesh: Rc<MeshAsset>,
}

impl MeshNode {
    /// Constructor
    #[inline]
    pub fn new(node: Node, mesh: Rc<MeshAsset>) -> Self {
        Self { node, mesh }
    }
}

impl Renderable for MeshNode {
    fn draw(&self, top_matrix: Mat4, context: &mut DrawContext) -> Result<()> {
        let node_matrix = top_matrix * self.node.world_transform;
        for surface in &self.mesh.surfaces {
            let render_object = RenderObject::new(
                surface.count,
                surface.start_index,
                self.mesh.index_alloc.buffer,
                surface.material.data,
                node_matrix,
                self.mesh.vertex_alloc.device_address()?,
            );
            context.opaque_surfaces.push(render_object);
        }
        self.node.draw(top_matrix, context)?;

        Ok(())
    }
}

/// Mesh data to upload to the engine
pub struct RawMesh {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
}

impl RawMesh {
    #[inline]
    pub fn new(
        name: String,
        surfaces: Vec<GeoSurface>,
        indices: Vec<u32>,
        vertices: Vec<Vertex>,
    ) -> Self {
        Self {
            name,
            surfaces,
            indices,
            vertices,
        }
    }
}

/// Mesh asset to render
#[derive(Default, Debug)]
pub struct MeshAsset {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub index_alloc: Allocation,
    pub vertex_alloc: Allocation,
}

impl MeshAsset {
    /// Constructor
    #[inline]
    pub fn new(
        name: String,
        surfaces: Vec<GeoSurface>,
        index_alloc: Allocation,
        vertex_alloc: Allocation,
    ) -> Self {
        Self {
            name,
            surfaces,
            index_alloc,
            vertex_alloc,
        }
    }

    /// Cleanup resources
    pub fn cleanup(&self, device: &Device) {
        ALLOCATOR.deallocate(device, &self.index_alloc);
        ALLOCATOR.deallocate(device, &self.vertex_alloc);
    }
}

/// Wehre a mesh starts and ends in a buffer
#[derive(Default, Debug)]
pub struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
    /// Using an [`Rc`] as we can have multiple meshes refering to the same material
    pub material: Rc<GLTFMaterial>,
}

impl GeoSurface {
    /// Constructor
    #[inline]
    pub fn new(start_index: u32, count: u32, material: Rc<GLTFMaterial>) -> Self {
        Self {
            start_index,
            count,
            material,
        }
    }
}

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
    #[inline]
    pub fn new(position: Vec3, color: Vec4, tex_coord: Vec2, normal: Vec3) -> Self {
        Self {
            position,
            uv_x: tex_coord[0],
            color,
            uv_y: tex_coord[1],
            normal,
        }
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
