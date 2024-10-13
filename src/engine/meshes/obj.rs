use super::{GeoSurface, RawMesh, Vertex};
use crate::engine::{material::GLTFMaterial, Engine};
use anyhow::Result;
use cgmath::{vec2, vec3, vec4};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

impl Engine {
    /// Load obj meshes from a file.
    pub fn load_obj(&mut self, path: &Path) -> Result<()> {
        let mut reader = BufReader::new(File::open(path)?);
        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
            |_| Ok(Default::default()),
        )?;
        let mut unique_vertices = HashMap::new();
        for model in models {
            let mut indices = Vec::new();
            let mut vertices = Vec::new();
            let surface = GeoSurface::new(
                indices.len().try_into()?,
                model.mesh.indices.len().try_into()?,
                GLTFMaterial::default().into(),
            );

            for index in model.mesh.indices {
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
                    indices.push(u32::try_from(*index)?);
                } else {
                    let index = vertices.len();
                    unique_vertices.insert(vertex, index);
                    vertices.push(vertex);
                    indices.push(u32::try_from(index)?);
                }
            }
            self.upload_mesh(RawMesh::new(model.name, vec![surface], indices, vertices))?;
        }

        Ok(())
    }
}
