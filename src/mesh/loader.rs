use super::MeshAsset;
use crate::engine::Engine;
use anyhow::Result;
use std::path::Path;

/// Mesh loader
#[derive(Default)]
pub struct MeshLoader;

/// Trait a mesh loader must implement to load meshes of type `T`.
pub trait Loader<T> {
    /// Load meshes from a file.
    fn load(&self, engine: &Engine, path: &Path) -> Result<Vec<MeshAsset>>;
}
