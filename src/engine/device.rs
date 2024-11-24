use super::{
    swapchain::SwapchainSupport, ENABLE_VALIDATION_LAYER, PORTABILITY_MACOS_VERSION,
    VALIDATION_LAYER,
};
use anyhow::{anyhow, Result};
use log::{info, warn};
use std::collections::HashSet;
use thiserror::Error as ThisError;
use vulkanalia::prelude::v1_3::{
    vk::{self, KhrSurfaceExtension},
    Device, Entry, HasBuilder, Instance, InstanceV1_0,
};

/// Required device extensions.
pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

/// Device error
#[derive(Debug, ThisError)]
pub enum Error {
    #[error("Suitability error: {0}")]
    Suitability(&'static str),
}

/// Create logical device
pub fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    indices: &HashSet<u32>,
) -> Result<Device> {
    let queue_priorities = &[1.0];
    let queue_infos = indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
                .build()
        })
        .collect::<Vec<_>>();

    let layers = if ENABLE_VALIDATION_LAYER {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .sample_rate_shading(true)
        .build();
    // Extra features
    let mut vk12 = vk::PhysicalDeviceVulkan12Features::builder()
        .buffer_device_address(true)
        .build();
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::builder()
        .synchronization2(true)
        .build();
    let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::builder()
        .dynamic_rendering(true)
        .build();

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut vk12)
        .push_next(&mut sync2)
        .push_next(&mut dynamic_rendering)
        .build();
    let device = unsafe { instance.create_device(physical_device, &info, None)? };

    Ok(device)
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
    pub fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let (mut graphics, mut present) = (None, None);
        for (index, properties) in properties.iter().enumerate() {
            if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics = Some(index as u32);
            }
            if unsafe {
                instance.get_physical_device_surface_support_khr(
                    physical_device,
                    index as u32,
                    surface,
                )?
            } {
                present = Some(index as u32);
            }
        }
        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            // TODO: Improve error description
            Err(anyhow!(Error::Suitability("Required queue families")))
        }
    }
}

/// Get the maximum supported MSAA sample count for a device
pub fn max_msaa_samples(instance: &Instance, device: vk::PhysicalDevice) -> vk::SampleCountFlags {
    let properties = unsafe { instance.get_physical_device_properties(device) };
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

/// Pick the GPU that matches the requirements
pub fn pick_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
) -> Result<vk::PhysicalDevice> {
    unsafe {
        for physical_device in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical_device);
            if let Err(err) = check_physical_device(instance, physical_device, surface) {
                warn!(
                    "Skipping physical device (`{}`): {err}",
                    properties.device_name
                );
            } else {
                info!("Selected physical device (`{}`)", properties.device_name);
                return Ok(physical_device);
            }
        }
    }

    Err(anyhow!("Failed to find suitable physical device"))
}

/// Check if a physical device matches the requirements
fn check_physical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<()> {
    check_physical_device_extensions(instance, physical_device, surface)?;
    QueueFamilyIndices::get(instance, physical_device, surface)?;
    let features = unsafe { instance.get_physical_device_features(physical_device) };
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(Error::Suitability("No sampler anisotropy")));
    }
    Ok(())
}

/// Check a physical device extensions
fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<()> {
    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device, None)?
            .iter()
            .map(|e| e.extension_name)
            .collect::<HashSet<_>>()
    };
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
