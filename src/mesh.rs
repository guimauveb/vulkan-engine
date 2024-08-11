// TODO: Abstract and general mesh loading (implement most popular formats)
use {
    super::{
        buffer::{create_index_buffer, create_vertex_buffer, BufferAllocation},
        vertex::Vertex,
        EngineData, Mat4,
    },
    anyhow::Result,
    cgmath::{vec2, vec3},
    gltf::{
        accessor::Iter,
        mesh::util::{ReadColors, ReadIndices, ReadTexCoords},
    },
    hashbrown::HashMap,
    std::{fs::File, io::BufReader, mem::size_of, path::Path},
    vulkanalia::{vk, Device, Instance},
};

pub unsafe fn load_obj_meshes(
    instance: &Instance,
    device: &Device,
    data: &mut EngineData,
) -> Result<()> {
    let mut reader = BufReader::new(File::open("resources/viking_room.obj")?);
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
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;
            let vertex = Vertex::new(
                vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                vec3(1.0, 1.0, 1.0),
                vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
                vec3(1.0, 0.0, 0.0),
            );
            if let Some(index) = unique_vertices.get(&vertex) {
                data.indices.push(*index as u32);
            } else {
                let index = data.vertices.len();
                unique_vertices.insert(vertex, index);
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }
        }
    }

    data.vertex_buffer = create_vertex_buffer(
        instance,
        device,
        data.physical_device,
        data.graphics_queue,
        data.command_pool,
        &data.vertices,
        (size_of::<Vertex>() * data.vertices.len()) as vk::DeviceSize,
    )?;
    data.index_buffer = create_index_buffer(
        instance,
        device,
        data.physical_device,
        data.graphics_queue,
        data.command_pool,
        &data.indices,
        (size_of::<u32>() * data.indices.len()) as vk::DeviceSize,
    )?;

    Ok(())
}

/// Holds the resources needed for a mesh
#[derive(Default)]
pub struct MeshBuffers {
    pub index_buffer: BufferAllocation,
    pub vertex_buffer: BufferAllocation,
    // NOTE - Use the buffer address to efficiently access the buffer
    // We will be using this for our vertices because accessing a SSBO through device address is faster than accessing it through descriptor sets,
    // and we can send it through push constants for a really fast and really easy way of binding the vertex data to the shaders.
    pub vk_device_address: vk::DeviceAddress,
}

/// Push constants for our mesh object draws
struct GPUDrawPushConstants {
    world_matrix: Mat4,
    vertex_buffer: vk::DeviceAddress,
}

#[derive(Default)]
struct GeoSurface {
    start_index: u32,
    count: u32,
}

impl GeoSurface {
    pub fn new(start_index: u32, count: u32) -> Self {
        Self { start_index, count }
    }
}

#[derive(Default)]
struct MeshAsset {
    name: String,
    surfaces: Vec<GeoSurface>,
    mesh_buffers: MeshBuffers,
}

pub unsafe fn load_gltf_meshes(
    instance: &Instance,
    device: &Device,
    data: &mut EngineData,
    path: &Path,
) -> Result<()> {
    let (gltf, buffers, _) = gltf::import(path)?;
    // Reuse the same vectors for all meshes so that we do not allocate memory for every mesh
    let (mut indices, mut vertices) = (Vec::new(), Vec::<Vertex>::new());
    let mut meshes = Vec::new();

    for mesh in gltf.meshes() {
        let mut new_mesh = MeshAsset::default();
        new_mesh.name = mesh.name().unwrap_or_default().into();

        indices.clear();
        vertices.clear();

        for primitive in mesh.primitives() {
            let surface = GeoSurface::new(
                indices.len() as u32,
                primitive.indices().unwrap().count() as u32,
            );
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            // Original C++ code: newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;
            let initial_vtx = vertices.len();

            // Load indices
            {
                if let Some(ReadIndices::U32(Iter::Standard(iter))) = reader.read_indices() {
                    for index in iter {
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
                            vec3(1.0, 0.0, 0.0),
                            vec2(1., 1.),
                            vec3(1., 1., 1.),
                        );
                        match vertices.get_mut(initial_vtx + index) {
                            Some(entry) => {
                                *entry = vertex;
                            }
                            None => {
                                vertices.push(vertex);
                            }
                        }

                        vertices[initial_vtx + index] = vertex;
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
                if let Some(ReadTexCoords::F32(Iter::Standard(iter))) = reader.read_tex_coords(0) {
                    for (index, tex_coord) in iter.enumerate() {
                        vertices[initial_vtx + index].tex_coord = tex_coord.into();
                    }
                }
            }

            // Load vertex colors
            if let Some(ReadColors::RgbF32(Iter::Standard(iter))) = reader.read_colors(0) {
                for (index, color) in iter.enumerate() {
                    vertices[initial_vtx + index].color = color.into();
                }
            }
        }

        // From vkguide.dev: "With the OverrideColors as a compile time flag, we override the vertex colors with the vertex normals which is useful for debugging."
        // TODO: Manage this flag through the GUI
        let override_colors = true;
        if override_colors {
            for vertex in &mut vertices {
                vertex.color = vertex.normal;
            }
        }

        // TODO: Add meshes to the engine
        // new_mesh.mesh_buffers = upload_mesh(indices, vertices);
        meshes.push(new_mesh);
    }

    Ok(())
}
