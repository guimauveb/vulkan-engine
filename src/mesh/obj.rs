use super::{loader::MeshLoader, GeoSurface, MeshAsset};
use crate::{engine::Engine, vertex::Vertex};
use anyhow::Result;
use cgmath::{vec2, vec3, vec4};
use hashbrown::HashMap;
use std::{fs::File, io::BufReader, path::Path};

/// Trait a mesh loader must implement to load obj meshes
pub trait ObjLoader {
    /// Load obj meshes from a file.
    fn _load_obj(&self, engine: &Engine, path: &Path) -> Result<Vec<MeshAsset>>;
}

impl ObjLoader for MeshLoader {
    fn _load_obj(&self, engine: &Engine, path: &Path) -> Result<Vec<MeshAsset>> {
        let mut reader = BufReader::new(File::open(path)?);
        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
            |_| Ok(Default::default()),
        )?;

        let (mut meshes, mut indices, mut vertices) = (Vec::new(), Vec::new(), Vec::new());
        let mut unique_vertices = HashMap::new();

        for model in models {
            indices.clear();
            vertices.clear();

            let new_surface =
                GeoSurface::new(indices.len() as u32, model.mesh.indices.len() as u32);

            for index in &model.mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;
                let vertex = Vertex::new(
                    vec3(
                        model.mesh.positions[pos_offset],
                        model.mesh.positions[pos_offset + 1],
                        model.mesh.positions[pos_offset + 2],
                    ),
                    vec4(1.0, 1.0, 1.0, 1.0),
                    vec2(
                        model.mesh.texcoords[tex_coord_offset],
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                    ),
                    vec3(1.0, 0.0, 0.0),
                );
                if let Some(index) = unique_vertices.get(&vertex) {
                    indices.push(*index as u32);
                } else {
                    let index = vertices.len();
                    unique_vertices.insert(vertex, index);
                    vertices.push(vertex);
                    indices.push(index as u32);
                }
            }
            unsafe {
                let mesh_buffers = engine.create_mesh(&indices, &vertices)?;
                let new_mesh = MeshAsset::new(model.name, vec![new_surface], mesh_buffers);
                meshes.push(new_mesh);
            }
        }

        Ok(meshes)
    }
}
