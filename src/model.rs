// TODO - Abstract and general mesh loading (implement most popular formats)
use {
    super::{vertex::Vertex, EngineData},
    anyhow::Result,
    cgmath::{vec2, vec3},
    hashbrown::HashMap,
    std::{fs::File, io::BufReader},
};

pub fn load_model(data: &mut EngineData) -> Result<()> {
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
    for model in &models {
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

    Ok(())
}
