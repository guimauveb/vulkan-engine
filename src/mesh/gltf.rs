use super::{loader::MeshLoader, GeoSurface, MeshAsset};
use crate::{engine::Engine, vertex::Vertex};
use anyhow::Result;
use cgmath::{vec2, vec3, vec4};
use gltf::{mesh::Reader, Buffer};
use log::error;
use std::path::Path;

/// Trait a mesh loader must implement to load gLTF meshes
pub trait GltfLoader {
    /// Load gLTF meshes from a file.
    fn load_gltf(&self, engine: &Engine, path: &Path) -> Result<Vec<MeshAsset>>;
}

impl GltfLoader for MeshLoader {
    fn load_gltf(&self, engine: &Engine, path: &Path) -> Result<Vec<MeshAsset>> {
        let (gltf, buffers, _) = gltf::import(path)?;
        // Reuse the same vectors for all meshes so that we do not allocate memory for every mesh
        let (mut meshes, mut indices, mut vertices) = (Vec::new(), Vec::new(), Vec::new());

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

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let initial_vtx = vertices.len();

                load_indices(&reader, &mut indices, initial_vtx as u32);
                load_vertex_positions(&reader, &mut vertices, initial_vtx);
                load_vertex_normals(&reader, &mut vertices, initial_vtx);
                load_uvs(&reader, &mut vertices, initial_vtx);
                load_vertex_colors(&reader, &mut vertices, initial_vtx);

                // TODO: Manage this flag through the GUI
                let replace_colors = false;
                if replace_colors {
                    override_colors(&mut vertices);
                }

                surfaces.push(new_surface);
            }

            unsafe {
                let mesh_buffers = engine.create_mesh(&indices, &vertices)?;
                let new_mesh = MeshAsset::new(
                    mesh.name().unwrap_or_default().into(),
                    surfaces,
                    mesh_buffers,
                );
                meshes.push(new_mesh);
            }
        }

        Ok(meshes)
    }
}

fn load_indices<'a, 's>(
    reader: &Reader<'a, 's, impl Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>>,
    indices: &mut Vec<u32>,
    initial_vtx: u32,
) {
    if let Some(read_indices) = reader.read_indices() {
        for index in read_indices.into_u32() {
            indices.push(index + initial_vtx);
        }
    }
}

fn load_vertex_positions<'a, 's>(
    reader: &Reader<'a, 's, impl Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>>,
    vertices: &mut Vec<Vertex>,
    initial_vtx: usize,
) {
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

fn load_vertex_normals<'a, 's>(
    reader: &Reader<'a, 's, impl Fn(Buffer<'a>) -> Option<&'s [u8]> + Clone>,
    vertices: &mut [Vertex],
    initial_vtx: usize,
) {
    if let Some(normals) = reader.read_normals() {
        for (index, normal) in normals.enumerate() {
            vertices[initial_vtx + index].normal = normal.into();
        }
    }
}

fn load_uvs<'a, 's>(
    reader: &Reader<'a, 's, impl Fn(Buffer<'a>) -> Option<&'s [u8]> + Clone>,
    vertices: &mut [Vertex],
    initial_vtx: usize,
) {
    if let Some(iter) = reader.read_tex_coords(0) {
        for (index, tex_coord) in iter.into_f32().enumerate() {
            vertices[initial_vtx + index].uv_x = tex_coord[0];
            vertices[initial_vtx + index].uv_y = tex_coord[1];
        }
    }
}

fn load_vertex_colors<'a, 's>(
    reader: &Reader<'a, 's, impl Fn(Buffer<'a>) -> Option<&'s [u8]> + Clone>,
    vertices: &mut [Vertex],
    initial_vtx: usize,
) {
    if let Some(iter) = reader.read_colors(0) {
        for (index, color) in iter.into_rgba_f32().enumerate() {
            vertices[initial_vtx + index].color = color.into();
        }
    }
}

// From vkguide.dev: "With the OverrideColors as a compile time flag, we override the vertex colors with the vertex normals which is useful for debugging."
fn override_colors(vertices: &mut [Vertex]) {
    for vertex in vertices {
        vertex.color[0] = vertex.normal[0];
        vertex.color[1] = vertex.normal[1];
        vertex.color[2] = vertex.normal[2];
    }
}
