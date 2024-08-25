use super::swapchain::SwapchainSupport;
use anyhow::{anyhow, Result};
use hashbrown::HashSet;
use thiserror::Error as ThisError;
use vulkanalia::prelude::v1_3::{
    vk::{self, KhrSurfaceExtension},
    Instance, InstanceV1_0,
};

/// Required device extensions.
pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

/// Device error
#[derive(Debug, ThisError)]
pub enum Error {
    #[error("Suitability error: {0}")]
    Suitability(&'static str),
}

/// Holds the indices of the graphics queue and the present queue.
/// It is very likely that these two queues end up being the same.
#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    /// Constructor
    pub unsafe fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);
        let (mut graphics, mut present) = (None, None);
        for (index, properties) in properties.iter().enumerate() {
            if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics = Some(index as u32);
            }
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                surface,
            )? {
                present = Some(index as u32);
            }
        }
        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            // TODO: Better error
            Err(anyhow!(Error::Suitability("Required queue families")))
        }
    }
}

/// Get the maximum supported MSAA sample count for a device
pub unsafe fn get_max_msaa_samples(
    instance: &Instance,
    device: vk::PhysicalDevice,
) -> vk::SampleCountFlags {
    let properties = instance.get_physical_device_properties(device);
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;
    [
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .into_iter()
    .find(|c| counts.contains(*c))
    .unwrap_or(vk::SampleCountFlags::_1)
}

/// Check a physical device extensions
pub unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        let support = SwapchainSupport::get(instance, physical_device, surface)?;
        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(anyhow!(Error::Suitability("Swapchain support")));
        }
        Ok(())
    } else {
        // TODO: List missing extensions
        Err(anyhow!(Error::Suitability("Required device extensions")))
    }
}

/// Check if a physical device matches the requirements
pub unsafe fn check_physical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<()> {
    check_physical_device_extensions(instance, physical_device, surface)?;
    QueueFamilyIndices::get(instance, physical_device, surface)?;
    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(Error::Suitability("No sampler anisotropy")));
    }
    Ok(())
}
