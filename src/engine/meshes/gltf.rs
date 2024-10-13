use super::{GeoSurface, RawMesh, Vertex};
use crate::engine::{material::GLTFMaterial, Engine};
use anyhow::Result;
use cgmath::{vec2, vec3, vec4};
use gltf::{mesh::Reader, Buffer};
use log::warn;
use std::path::Path;

impl Engine {
    /// Load gLTF meshes from a file.
    pub(crate) fn load_gltf(&mut self, path: &Path) -> Result<()> {
        let (gltf, buffers, _) = gltf::import(path)?;
        for mesh in gltf.meshes() {
            let (mut indices, mut vertices, mut surfaces) = (Vec::new(), Vec::new(), Vec::new());
            for primitive in mesh.primitives() {
                let new_surface = if let Some(p_indices) = primitive.indices() {
                    GeoSurface::new(
                        indices.len().try_into()?,
                        p_indices.count().try_into()?,
                        GLTFMaterial::default().into(),
                    )
                } else {
                    warn!("Mesh indices accessor not provided");
                    continue;
                };

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let initial_vtx = vertices.len();

                load_indices(&reader, &mut indices, initial_vtx.try_into()?);
                load_vertex_positions(&reader, &mut vertices, initial_vtx);
                load_vertex_normals(&reader, &mut vertices, initial_vtx);
                load_uvs(&reader, &mut vertices, initial_vtx);
                load_vertex_colors(&reader, &mut vertices, initial_vtx);

                // TODO: Manage this flag through the GUI
                let replace_colors = true;
                if replace_colors {
                    override_colors(&mut vertices);
                }

                surfaces.push(new_surface);
            }

            self.upload_mesh(RawMesh::new(
                mesh.name().unwrap_or_default().to_owned(),
                surfaces,
                indices,
                vertices,
            ))?;
        }

        Ok(())
    }
}

fn load_indices<'a, 's>(
    reader: &Reader<'a, 's, impl Fn(Buffer<'a>) -> Option<&'s [u8]> + Clone>,
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
    reader: &Reader<'a, 's, impl Fn(Buffer<'a>) -> Option<&'s [u8]> + Clone>,
    vertices: &mut Vec<Vertex>,
    initial_vtx: usize,
) {
    if let Some(positions) = reader.read_positions() {
        for (index, position) in positions.enumerate() {
            let vertex = Vertex::new(
                position.into(),
                vec4(1.0, 1.0, 1.0, 1.0),
                vec2(0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
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

// Override the vertex colors with the vertex normals, useful for debugging
fn override_colors(vertices: &mut [Vertex]) {
    for vertex in vertices {
        vertex.color[0] = vertex.normal[0];
        vertex.color[1] = vertex.normal[1];
        vertex.color[2] = vertex.normal[2];
    }
}
