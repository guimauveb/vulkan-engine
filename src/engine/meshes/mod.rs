pub mod obj;

use super::{
    material::{Material, MaterialInstance},
    memory::Allocation,
    Engine, Mat4, Vec2, Vec3, Vec4, VulkanInterface, ALLOCATOR,
};
use anyhow::Result;
use cgmath::{vec4, SquareMatrix};
use std::{
    ffi::c_void,
    hash::{Hash, Hasher},
    ptr::copy_nonoverlapping as memcpy,
    rc::Rc,
};
use vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0, HasBuilder};

impl Engine {
    /// Upload a mesh to the engine
    pub(crate) fn upload_mesh(&mut self, mesh: RawMesh) -> Result<MeshAsset> {
        let (index_size, vertex_size) = (
            size_of_val(&mesh.indices[..]) as vk::DeviceSize,
            size_of_val(&mesh.vertices[..]) as vk::DeviceSize,
        );
        let allocations = MeshAllocations::new(&self.interface, vertex_size, index_size)?;
        let mut staging_memory: MeshStagingMemory<Vertex, u32> =
            allocations.staging_memory(&self.interface.device)?;
        unsafe {
            memcpy(
                mesh.vertices.as_ptr(),
                staging_memory.vertices,
                mesh.vertices.len(),
            );
            staging_memory.vertices = staging_memory.vertices.add(mesh.vertices.len());
            memcpy(
                mesh.indices.as_ptr(),
                staging_memory.indices,
                mesh.indices.len(),
            );
        }

        let copy = |cmd_buffer: vk::CommandBuffer| {
            let vertex_copy = vk::BufferCopy::builder().size(vertex_size).build();
            unsafe {
                self.interface.device.cmd_copy_buffer(
                    cmd_buffer,
                    allocations.staging.buffer,
                    allocations.vertices.buffer,
                    &[vertex_copy],
                );
            }
            let index_copy = vk::BufferCopy::builder()
                .src_offset(vertex_size)
                .size(index_size)
                .build();
            unsafe {
                self.interface.device.cmd_copy_buffer(
                    cmd_buffer,
                    allocations.staging.buffer,
                    allocations.indices.buffer,
                    &[index_copy],
                );
            }

            Ok(())
        };
        self.immediate_submit
            .execute(&self.interface.device, copy)?;

        allocations.staging.unmap_memory(&self.interface.device);
        ALLOCATOR.deallocate(&self.interface.device, &allocations.staging);

        Ok(MeshAsset::new(
            mesh.name,
            mesh.surfaces,
            allocations.vertices,
            allocations.indices,
        ))
    }
}

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

/// Mesh data to upload to the engine
#[derive(Debug, Clone)]
pub struct RawMesh {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl RawMesh {
    #[inline]
    pub fn new(
        name: String,
        surfaces: Vec<GeoSurface>,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Self {
        Self {
            name,
            surfaces,
            vertices,
            indices,
        }
    }
}

/// Mesh asset to render
#[derive(Default, Debug, Clone)]
pub struct MeshAsset {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub vertex_alloc: Allocation,
    pub index_alloc: Allocation,
    destroyed: bool,
}

impl MeshAsset {
    /// Constructor
    #[inline]
    pub fn new(
        name: String,
        surfaces: Vec<GeoSurface>,
        vertex_alloc: Allocation,
        index_alloc: Allocation,
    ) -> Self {
        Self {
            name,
            surfaces,
            vertex_alloc,
            index_alloc,
            destroyed: false,
        }
    }

    /// Destroy resources
    pub fn destroy(&mut self, device: &Device) {
        if !self.destroyed {
            ALLOCATOR.deallocate(device, &self.vertex_alloc);
            ALLOCATOR.deallocate(device, &self.index_alloc);
        }
        self.destroyed = true;
    }
}

/// Meshes are written into GPU memory through a single staging buffer
#[derive(Copy, Clone, Debug)]
pub struct MeshAllocations {
    pub staging: Allocation,
    pub vertices: Allocation,
    pub indices: Allocation,
}

impl MeshAllocations {
    /// Constructor
    pub fn new(interface: &VulkanInterface, vertex_size: u64, index_size: u64) -> Result<Self> {
        // Staging buffers
        let staging_info = vk::BufferCreateInfo::builder()
            .size(vertex_size + index_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .build();
        let staging_alloc = ALLOCATOR.allocate_buffer(
            interface,
            staging_info,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Destination buffers
        let vertex_info = vk::BufferCreateInfo::builder()
            .size(vertex_size)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .build();
        let vertex_alloc = ALLOCATOR.allocate_buffer(
            interface,
            vertex_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let index_info = vk::BufferCreateInfo::builder()
            .size(index_size)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .build();
        let index_alloc = ALLOCATOR.allocate_buffer(
            interface,
            index_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Ok(Self {
            staging: staging_alloc,
            vertices: vertex_alloc,
            indices: index_alloc,
        })
    }

    /// Get mesh allocations staging memory pointers.
    ///
    /// Pointers are cast to the desired `Vertex` and `Index` types.
    pub fn staging_memory<Vertex, Index>(
        &self,
        device: &Device,
    ) -> Result<MeshStagingMemory<Vertex, Index>> {
        let staging = self.staging.mapped_memory::<c_void>(device)?;
        #[cfg(debug_assertions)]
        {
            let total_size = self.vertices.size + self.indices.size;
            assert!(
                self.staging.size >= total_size,
                "Staging buffer too small for mesh data: {total_size} > {}",
                self.staging.size,
            );
        }
        let vertices: *mut Vertex = staging.cast();
        let indices: *mut Index = unsafe { staging.add(self.vertices.size as usize).cast() };

        Ok(MeshStagingMemory { vertices, indices })
    }

    /// Destroy resources
    pub fn destroy(&self, device: &Device) {
        ALLOCATOR.deallocate(device, &self.staging);
        ALLOCATOR.deallocate(device, &self.vertices);
        ALLOCATOR.deallocate(device, &self.indices);
    }
}

/// Mesh staging memory pointers
#[derive(Copy, Clone, Debug)]
pub struct MeshStagingMemory<Vertex, Index> {
    pub vertices: *mut Vertex,
    pub indices: *mut Index,
}

/// Wehre a mesh starts and ends in a buffer
#[derive(Default, Debug, Clone)]
pub struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
    /// Using an [`Rc`] as we can have multiple meshes refering to the same material
    pub material: Rc<Material>,
}

impl GeoSurface {
    /// Constructor
    #[inline]
    pub fn new(start_index: u32, count: u32, material: Rc<Material>) -> Self {
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
