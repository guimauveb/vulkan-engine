use crate::device;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Device error: {0}")]
    Device(#[from] device::Error),
}
