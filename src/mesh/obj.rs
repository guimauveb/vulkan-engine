// TODO:
#![allow(unused)]
use crate::{
    buffer::{create_index_buffer, create_vertex_buffer},
    engine::{engine_data::EngineData, Engine},
    vertex::Vertex,
};
use anyhow::Result;
use cgmath::{vec2, vec3, vec4};
use hashbrown::HashMap;
use std::{fs::File, io::BufReader, mem::size_of, path::Path};
use vulkanalia::{vk, Device, Instance};

use super::{loader::MeshLoader, MeshAsset};

/// Trait a mesh loader must implement to load obj meshes
pub trait ObjLoader {
    /// Load obj meshes from a file.
    fn load_obj(&self, engine: &Engine, path: &Path) -> Result<Vec<MeshAsset>>;
}

impl ObjLoader for MeshLoader {
    fn load_obj(&self, engine: &Engine, path: &Path) -> Result<Vec<MeshAsset>> {
        // let mut reader = BufReader::new(File::open("resources/viking_room.obj")?);
        // let (models, _) = tobj::load_obj_buf(
        //     &mut reader,
        //     &tobj::LoadOptions {
        //         triangulate: true,
        //         ..Default::default()
        //     },
        //     |_| Ok(Default::default()),
        // )?;

        // let mut unique_vertices = HashMap::new();
        // for model in models {
        //     for index in &model.mesh.indices {
        //         let pos_offset = (3 * index) as usize;
        //         let tex_coord_offset = (2 * index) as usize;
        //         let vertex = Vertex::new(
        //             vec3(
        //                 model.mesh.positions[pos_offset],
        //                 model.mesh.positions[pos_offset + 1],
        //                 model.mesh.positions[pos_offset + 2],
        //             ),
        //             vec4(1.0, 1.0, 1.0, 1.0),
        //             vec2(
        //                 model.mesh.texcoords[tex_coord_offset],
        //                 1.0 - model.mesh.texcoords[tex_coord_offset + 1],
        //             ),
        //             vec3(1.0, 0.0, 0.0),
        //         );
        //         if let Some(index) = unique_vertices.get(&vertex) {
        //             data.indices.push(*index as u32);
        //         } else {
        //             let index = data.vertices.len();
        //             unique_vertices.insert(vertex, index);
        //             data.vertices.push(vertex);
        //             data.indices.push(index as u32);
        //         }
        //     }
        // }

        // data.vertex_buffer = create_vertex_buffer(
        //     instance,
        //     device,
        //     data.physical_device,
        //     data.graphics_queue,
        //     data.command_pool,
        //     &data.vertices,
        //     (size_of::<Vertex>() * data.vertices.len()) as vk::DeviceSize,
        // )?;
        // data.index_buffer = create_index_buffer(
        //     instance,
        //     device,
        //     data.physical_device,
        //     data.graphics_queue,
        //     data.command_pool,
        //     &data.indices,
        //     (size_of::<u32>() * data.indices.len()) as vk::DeviceSize,
        // )?;

        Ok(Vec::new())
    }
}
