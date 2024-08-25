pub mod gltf;
pub mod obj;

use crate::{buffer::BufferAllocation, Mat4};
use vulkanalia::vk;

/// Holds the resources needed for a mesh
#[derive(Default)]
pub struct MeshBuffers {
    pub index_buffer: BufferAllocation,
    pub vertex_buffer: BufferAllocation,
}

impl MeshBuffers {
    #[inline]
    pub fn new(index_buffer: BufferAllocation, vertex_buffer: BufferAllocation) -> Self {
        Self {
            index_buffer,
            vertex_buffer,
        }
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

#[derive(Default)]
pub struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
}

impl GeoSurface {
    pub fn new(start_index: u32, count: u32) -> Self {
        Self { start_index, count }
    }
}

#[derive(Default)]
pub struct MeshAsset {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub mesh_buffers: MeshBuffers,
}

impl MeshAsset {
    #[inline]
    pub fn new(name: String, surfaces: Vec<GeoSurface>, mesh_buffers: MeshBuffers) -> Self {
        Self {
            name,
            surfaces,
            mesh_buffers,
        }
    }
}
