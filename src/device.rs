use {
    super::{
        swapchain::SwapchainSupport, EngineData, DEVICE_EXTENSIONS, ENABLE_VALIDATION_LAYER,
        PORTABILITY_MACOS_VERSION, VALIDATION_LAYER,
    },
    anyhow::{anyhow, Result},
    hashbrown::HashSet,
    log::{info, warn},
    thiserror::Error as ThisError,
    vulkanalia::prelude::v1_0::{
        vk::{self, KhrSurfaceExtension},
        Device, DeviceV1_0, Entry, HasBuilder, Instance, InstanceV1_0,
    },
};

#[derive(Debug, ThisError)]
#[error("Missing {0}")]
pub struct SuitabilityError(pub &'static str);

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);
        let (mut graphics, mut present) = (None, None);
        for (index, properties) in properties.iter().enumerate() {
            // It is very likely that these two queues end up being the same.
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
            Err(anyhow!(SuitabilityError("Required queue families")))
        }
    }
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    data: &EngineData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        let support = SwapchainSupport::get(instance, data, physical_device)?;
        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(anyhow!(SuitabilityError("Swapchain support")));
        }
        Ok(())
    } else {
        // TODO - List missing extensions
        Err(anyhow!(SuitabilityError("Required device extensions")))
    }
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &EngineData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    check_physical_device_extensions(instance, data, physical_device)?;
    QueueFamilyIndices::get(instance, physical_device, data.surface)?;
    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy")));
    }
    Ok(())
}

unsafe fn get_max_msaa_samples(instance: &Instance, data: &EngineData) -> vk::SampleCountFlags {
    let properties = instance.get_physical_device_properties(data.physical_device);
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

pub unsafe fn pick_physical_device(instance: &Instance, data: &mut EngineData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);
        if let Err(err) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {err}",
                properties.device_name
            );
        } else {
            info!("Selected physical device (`{}`)", properties.device_name);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device"))
}

pub unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut EngineData,
) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, data.physical_device, data.surface)?;
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);
    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .into_iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(i)
                .queue_priorities(queue_priorities)
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
        .sample_rate_shading(true);
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);
    let device = instance.create_device(data.physical_device, &info, None)?;

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.graphics_queue_family_index = indices.graphics;
    data.present_queue = device.get_device_queue(indices.present, 0);
    data.present_queue_family_index = indices.present;

    Ok(device)
}
