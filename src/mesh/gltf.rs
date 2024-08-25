use super::{GeoSurface, MeshAsset};
use crate::{engine::Engine, vertex::Vertex};
use anyhow::Result;
use cgmath::{vec2, vec3, vec4};
use log::error;
use std::path::Path;

pub unsafe fn load_gltf_meshes(engine: &mut Engine, path: &Path) -> Result<()> {
    let (gltf, buffers, _) = gltf::import(path)?;
    // Reuse the same vectors for all meshes so that we do not allocate memory for every mesh
    let (mut indices, mut vertices) = (Vec::new(), Vec::new());

    for mesh in gltf.meshes() {
        let mut surfaces = Vec::new();
        indices.clear();
        vertices.clear();

        for primitive in mesh.primitives() {
            let new_surface = if let Some(p_indices) = primitive.indices() {
                GeoSurface::new(indices.len() as u32, p_indices.count() as u32)
            } else {
                error!("Mesh indices accessor not provided");
                continue;
            };
            let initial_vtx = vertices.len();

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            // Load indices
            {
                if let Some(read_indices) = reader.read_indices() {
                    for index in read_indices.into_u32() {
                        indices.push(index + initial_vtx as u32);
                    }
                }
            }

            // Load vertex positions
            {
                if let Some(positions) = reader.read_positions() {
                    for (index, position) in positions.enumerate() {
                        let vertex = Vertex::new(
                            position.into(),
                            vec4(1., 1., 1., 1.),
                            vec2(0., 0.),
                            vec3(1., 0., 0.),
                        );
                        match vertices.get_mut(initial_vtx + index) {
                            Some(entry) => {
                                *entry = vertex;
                            }
                            None => {
                                vertices.push(vertex);
                            }
                        }
                    }
                }
            }

            // Load vertex normals
            {
                if let Some(normals) = reader.read_normals() {
                    for (index, normal) in normals.enumerate() {
                        vertices[initial_vtx + index].normal = normal.into();
                    }
                }
            }

            // Load UVs
            {
                if let Some(iter) = reader.read_tex_coords(0) {
                    for (index, tex_coord) in iter.into_f32().enumerate() {
                        vertices[initial_vtx + index].uv_x = tex_coord[0];
                        vertices[initial_vtx + index].uv_y = tex_coord[1];
                    }
                }
            }

            // Load vertex colors
            if let Some(iter) = reader.read_colors(0) {
                for (index, color) in iter.into_rgba_f32().enumerate() {
                    vertices[initial_vtx + index].color = color.into();
                }
            }
            surfaces.push(new_surface);
        }

        // From vkguide.dev: "With the OverrideColors as a compile time flag, we override the vertex colors with the vertex normals which is useful for debugging."
        // TODO: Manage this flag through the GUI
        let override_colors = false;
        if override_colors {
            for vertex in &mut vertices {
                vertex.color[0] = vertex.normal[0];
                vertex.color[1] = vertex.normal[1];
                vertex.color[2] = vertex.normal[2];
            }
        }

        let mesh_buffers = engine.upload_mesh(&indices, &vertices)?;
        let new_mesh = MeshAsset::new(
            mesh.name().unwrap_or_default().into(),
            surfaces,
            mesh_buffers,
        );
        engine.data.meshes.push(new_mesh);
    }

    Ok(())
}
