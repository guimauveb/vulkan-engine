use super::{Node, NodeData, Scene};
use crate::engine::{
    descriptors::{DescriptorAllocator, PoolSizeRatio},
    image::write_pixels_to_image,
    material::{Material, MaterialConstants, MaterialPass, MaterialResources},
    memory::{AllocatedImage, AllocatedImageInfo, Allocation},
    meshes::{GeoSurface, MeshAsset, RawMesh, Vertex},
    Engine, Mat4, Vec4, ALLOCATOR,
};
use anyhow::Result;
use cgmath::{vec2, vec3, vec4, SquareMatrix};
use gltf::{
    image::{Data as ImageData, Format},
    material::AlphaMode,
    mesh::Reader,
    Buffer, Document,
};
use log::{debug, warn};
use std::{cell::RefCell, collections::HashMap, path::Path, rc::Rc, slice::from_raw_parts_mut};
use vulkanalia::{
    prelude::v1_3::vk,
    vk::{DeviceV1_0, Handle, HasBuilder},
    Device,
};

impl Engine {
    /// Load a gLTF scene from a file.
    pub(crate) fn load_gltf_scene(&mut self, path: &Path) -> Result<()> {
        debug!("Loading scene from '{}'", path.display());
        let (gltf, buffers, images) = gltf::import(path)?;

        let sizes = vec![
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 1.0),
        ];
        let mut descriptor_pool =
            DescriptorAllocator::new(&self.interface.device, gltf.materials().len() as u32, sizes)?;

        let samplers = load_samplers(&self.interface.device, &gltf)?;
        // FIXME: Some textures are not properly rendered. Might come from how images or materials
        // are loaded from gltf file.
        let images = load_images(self, images)?;
        let (materials, material_map, material_alloc) =
            load_materials(self, &gltf, &samplers, &images, &mut descriptor_pool)?;
        let (meshes, mesh_map) = load_meshes(self, &gltf, &buffers, &materials)?;
        let (node_map, top_nodes) = load_nodes(&gltf, &meshes)?;

        let scene = Scene::new(
            mesh_map,
            node_map,
            top_nodes,
            material_map,
            material_alloc,
            images,
            samplers,
            descriptor_pool,
        );
        self.scenes.insert("structure".into(), scene);

        Ok(())
    }
}

/// Extract samplers from gltf file
fn load_samplers(device: &Device, gltf: &Document) -> Result<Vec<vk::Sampler>> {
    gltf.samplers()
        .map(|sampler| {
            let info = vk::SamplerCreateInfo::builder()
                .max_lod(vk::LOD_CLAMP_NONE)
                .min_lod(0.0)
                .mag_filter(
                    sampler
                        .mag_filter()
                        .map(convert_mag_filter)
                        .unwrap_or(vk::Filter::NEAREST),
                )
                .min_filter(
                    sampler
                        .min_filter()
                        .map(convert_min_filter)
                        .unwrap_or(vk::Filter::NEAREST),
                )
                .mipmap_mode(
                    sampler
                        .min_filter()
                        .map(convert_mipmap_mode)
                        .unwrap_or(vk::SamplerMipmapMode::NEAREST),
                )
                .build();
            unsafe { device.create_sampler(&info, None) }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Convert [`gltf`] min filter to [`vk::Filter`]
fn convert_min_filter(filter: gltf::texture::MinFilter) -> vk::Filter {
    match filter {
        gltf::texture::MinFilter::Nearest
        | gltf::texture::MinFilter::NearestMipmapNearest
        | gltf::texture::MinFilter::NearestMipmapLinear => vk::Filter::NEAREST,
        gltf::texture::MinFilter::Linear
        | gltf::texture::MinFilter::LinearMipmapNearest
        | gltf::texture::MinFilter::LinearMipmapLinear => vk::Filter::LINEAR,
    }
}

/// Convert [`gltf`] mag filter to [`vk::Filter`]
fn convert_mag_filter(filter: gltf::texture::MagFilter) -> vk::Filter {
    match filter {
        gltf::texture::MagFilter::Nearest => vk::Filter::NEAREST,
        gltf::texture::MagFilter::Linear => vk::Filter::LINEAR,
    }
}

/// Convert [`gltf`] min filter to [`vk::SamplerMipMapMode`]
fn convert_mipmap_mode(mode: gltf::texture::MinFilter) -> vk::SamplerMipmapMode {
    match mode {
        gltf::texture::MinFilter::Nearest
        | gltf::texture::MinFilter::NearestMipmapNearest
        | gltf::texture::MinFilter::NearestMipmapLinear => vk::SamplerMipmapMode::NEAREST,
        gltf::texture::MinFilter::Linear
        | gltf::texture::MinFilter::LinearMipmapNearest
        | gltf::texture::MinFilter::LinearMipmapLinear => vk::SamplerMipmapMode::LINEAR,
    }
}

/// Load images from gltf file
fn load_images(engine: &mut Engine, images: Vec<ImageData>) -> Result<Vec<AllocatedImage>> {
    images
        .into_iter()
        .map(|image| {
            Ok(if image.format != Format::R8G8B8A8 {
                match image.format {
                    Format::R8G8B8 => {
                        debug!("Converting image `R8G8B8` to `R8G8B8A8`");
                        // FIXME: Fix gpu rgb->rgba conversion
                        // gpu_rgb_to_rgba(engine, image)?
                        rgb_to_rgba(engine, image)?
                    }
                    Format::R8 => {
                        let img_info = AllocatedImageInfo::default()
                                .extent(
                                    vk::Extent3D::builder()
                                        .width(image.width)
                                        .height(image.height)
                                        .depth(1)
                                        .build(),
                                )
                                .format(vk::Format::R8_UNORM)
                                .usage(
                                    vk::ImageUsageFlags::SAMPLED
                                        | vk::ImageUsageFlags::TRANSFER_SRC
                                        | vk::ImageUsageFlags::TRANSFER_DST,
                                )
                                .mipmapped(false);
                            let output_image = ALLOCATOR.allocate_image(&engine.interface, img_info)?;
                            write_pixels_to_image(
                                &engine.interface,
                                &engine.immediate_submit,
                                image.pixels.as_ptr(),
                                &output_image,
                                1,
                            )?;
                        output_image
                    }
                    format => {
                        warn!(
                            "Image format is `{format:?}` and must be converted to `R8B8G8A8`, using default image",
                        );
                        engine.checkerboard_image
                    }
                }
            } else {
                engine.checkerboard_image
            })
        })
        .collect::<Result<Vec<_>>>()
}

/// Convert an `RGB` image to `RGBA`
fn rgb_to_rgba(engine: &mut Engine, image: ImageData) -> Result<AllocatedImage> {
    let img_info = AllocatedImageInfo::default()
        .extent(
            vk::Extent3D::builder()
                .width(image.width)
                .height(image.height)
                .depth(1)
                .build(),
        )
        .format(vk::Format::R8G8B8A8_SRGB)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .mipmapped(false);
    let output_image = ALLOCATOR.allocate_image(&engine.interface, img_info)?;

    let alpha_size = image.pixels.len() / 3;
    let mut rgba_pixels = Vec::with_capacity(image.pixels.len() + alpha_size);
    for i in 0..image.pixels.len() {
        if i > 0 && i % 3 == 0 {
            rgba_pixels.push(1);
        }
        rgba_pixels.push(image.pixels[i]);
    }
    write_pixels_to_image(
        &engine.interface,
        &engine.immediate_submit,
        rgba_pixels.as_ptr(),
        &output_image,
        4,
    )?;

    Ok(output_image)
}

// FIXME: Does not really work, no validation errors but output colors are messed up.
// Might be an issue with gltf images loading.
// Check compute shader usage and glsl documentation.
/// Convert an `RGB` image to `RGBA` using the GPU
fn _gpu_rgb_to_rgba(engine: &mut Engine, mut image: ImageData) -> Result<AllocatedImage> {
    let img_info = AllocatedImageInfo::default()
        .extent(
            vk::Extent3D::builder()
                .width(image.width)
                .height(image.height)
                .depth(1)
                .build(),
        )
        .format(vk::Format::R8G8B8A8_UNORM)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
        .mipmapped(false);
    let output_image = ALLOCATOR.allocate_image(&engine.interface, img_info)?;

    let input_image_info = vk::BufferCreateInfo::builder()
        .size((size_of::<u32>() * image.pixels.len()) as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .build();
    let input_image_alloc = ALLOCATOR.allocate_buffer(
        &engine.interface,
        input_image_info,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let image_memory = unsafe {
        from_raw_parts_mut(
            input_image_alloc.mapped_memory::<u32>(&engine.interface.device)?,
            image.pixels.len(),
        )
    };
    let pixels_u32 = image
        .pixels
        .drain(..)
        .map(|pixel| pixel as u32)
        .collect::<Vec<_>>();
    image_memory.copy_from_slice(&pixels_u32);

    engine.descriptor_writer.write_image(
        0,
        output_image.image_view,
        vk::Sampler::null(),
        vk::ImageLayout::GENERAL,
        vk::DescriptorType::STORAGE_IMAGE,
    );
    engine
        .descriptor_writer
        .update_set(&engine.interface.device, engine.rgb_to_rgba_descriptors);
    engine.descriptor_writer.clear();

    let conversion = |cmd_buffer: vk::CommandBuffer| {
        engine._rgb_to_rgba(
            cmd_buffer,
            input_image_alloc,
            output_image,
            image.width,
            image.height,
        )?;
        Ok(())
    };
    engine
        .immediate_submit
        .execute(&engine.interface.device, conversion)?;

    input_image_alloc.unmap_memory(&engine.interface.device);
    ALLOCATOR.deallocate(&engine.interface.device, &input_image_alloc);

    Ok(output_image)
}

/// Extract materials from gltf file
fn load_materials(
    engine: &mut Engine,
    gltf: &Document,
    samplers: &[vk::Sampler],
    images: &[AllocatedImage],
    descriptor_pool: &mut DescriptorAllocator,
) -> Result<(Vec<Rc<Material>>, HashMap<String, Rc<Material>>, Allocation)> {
    let textures = gltf.textures().collect::<Vec<_>>();
    let mut materials = Vec::with_capacity(gltf.materials().len());
    let mut material_map = HashMap::with_capacity(gltf.materials().len());

    let material_size = (size_of::<MaterialConstants>() * gltf.materials().len()) as vk::DeviceSize;
    let material_info = vk::BufferCreateInfo::builder()
        .size(material_size)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .build();
    let material_alloc = ALLOCATOR.allocate_buffer(
        &engine.interface,
        material_info,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    debug_assert!(
        material_alloc.size >= material_size,
        "Buffer too small for material data: {material_size} > {}",
        material_alloc.size
    );
    let material_constants =
        material_alloc.mapped_memory::<MaterialConstants>(&engine.interface.device)?;
    let material_constants =
        unsafe { from_raw_parts_mut(material_constants, gltf.materials().len()) };

    for (idx, material) in gltf.materials().enumerate() {
        let color_factor: Vec4 = material.pbr_metallic_roughness().base_color_factor().into();
        let metallic_factor = material.pbr_metallic_roughness().metallic_factor();
        let roughness_factor = material.pbr_metallic_roughness().roughness_factor();

        // Write material parameters into buffer
        material_constants[idx] = MaterialConstants::new(
            color_factor,
            vec4(metallic_factor, roughness_factor, 0.0, 0.0),
        );
        let pass_type = if material.alpha_mode() == AlphaMode::Blend {
            MaterialPass::Transparent
        } else {
            MaterialPass::MainColor
        };

        // Use default material textures if there is no material
        let mut material_resources = MaterialResources::new(
            engine.white_image,
            engine.default_sampler_linear,
            engine.white_image,
            engine.default_sampler_linear,
            // Set uniform buffer for material data
            material_alloc,
            (idx * size_of::<MaterialConstants>()) as vk::DeviceSize,
        );

        // Grab textures from gltf file
        if let Some(info) = material.pbr_metallic_roughness().base_color_texture() {
            let texture_idx = info.texture().index();
            let image_idx = info.texture().source().index();
            material_resources.color_image = images[image_idx];
            if let Some(sampler_idx) = textures[texture_idx].sampler().index() {
                material_resources.color_sampler = samplers[sampler_idx];
            }
        }

        let material_name = material
            .name()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| idx.to_string());
        let material = engine.metal_rough_material.write_material(
            &engine.interface.device,
            pass_type,
            material_resources,
            descriptor_pool,
        )?;
        let material = Rc::new(material);
        materials.push(material.clone());
        material_map.insert(material_name, material.clone());
    }

    Ok((materials, material_map, material_alloc))
}

/// Load meshes from gltf file
fn load_meshes(
    engine: &mut Engine,
    gltf: &Document,
    buffers: &[gltf::buffer::Data],
    materials: &[Rc<Material>],
) -> Result<(
    Vec<Rc<RefCell<MeshAsset>>>,
    HashMap<String, Rc<RefCell<MeshAsset>>>,
)> {
    let mut meshes = Vec::with_capacity(gltf.meshes().len());
    let mut mesh_map = HashMap::with_capacity(gltf.meshes().len());
    for mesh in gltf.meshes() {
        let (mut indices, mut vertices, mut surfaces) = (Vec::new(), Vec::new(), Vec::new());
        for primitive in mesh.primitives() {
            let surface = if let Some(p_indices) = primitive.indices() {
                GeoSurface::new(
                    indices.len() as u32,
                    p_indices.count() as u32,
                    primitive
                        .material()
                        .index()
                        .map(|idx| &materials[idx])
                        .unwrap_or(&materials[0])
                        .clone(),
                )
            } else {
                warn!("Mesh indices accessor not provided");
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
            let replace_colors = true;
            if replace_colors {
                override_colors(&mut vertices);
            }

            surfaces.push(surface);
        }

        let mesh_name = mesh.name().unwrap_or_default().to_owned();
        let mesh =
            engine.upload_mesh(RawMesh::new(mesh_name.clone(), surfaces, vertices, indices))?;
        let mesh = Rc::new(RefCell::new(mesh));
        meshes.push(mesh.clone());
        mesh_map.insert(mesh_name, mesh);
    }

    Ok((meshes, mesh_map))
}

/// Load mesh indices
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

/// Load vertex positions
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

/// Load mesh vertex normals
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

/// Load mesh textures
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

/// Load mesh vertex colors
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

/// Load nodes from gltf file
fn load_nodes(
    gltf: &Document,
    meshes: &[Rc<RefCell<MeshAsset>>],
) -> Result<(HashMap<String, Rc<RefCell<Node>>>, Vec<Rc<RefCell<Node>>>)> {
    let mut nodes = Vec::with_capacity(gltf.nodes().len());
    let mut node_map = HashMap::with_capacity(gltf.nodes().len());
    let mut top_nodes = Vec::with_capacity(gltf.nodes().len());
    for (idx, node) in gltf.nodes().enumerate() {
        let mut scene_node = Node {
            local_transform: node.transform().matrix().into(),
            ..Default::default()
        };
        if let Some(mesh) = node.mesh() {
            scene_node.data = NodeData::Mesh(meshes[mesh.index()].clone());
        }
        let scene_node = Rc::new(RefCell::new(scene_node));
        nodes.push(scene_node.clone());
        node_map.insert(
            node.name()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| idx.to_string()),
            scene_node,
        );
    }

    // Loop again to setup transform hierarchy
    for node in gltf.nodes() {
        let scene_node = &nodes[node.index()];
        for child in node.children() {
            let child_idx = child.index();
            let child = &nodes[child_idx];
            scene_node.borrow_mut().children.push(child.clone());
            child.borrow_mut().parent = Some(Rc::downgrade(scene_node));
        }
    }

    // Find the top nodes (without parents)
    for node in &nodes {
        let mut node_ref = node.borrow_mut();
        if node_ref.parent.is_none() {
            top_nodes.push(node.clone());
            node_ref.refresh_transform(&Mat4::identity());
        }
    }

    Ok((node_map, top_nodes))
}
