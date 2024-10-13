use super::{device::QueueFamilyIndices, image::create_image_view, VulkanInterface};
use anyhow::Result;
use vulkanalia::{
    prelude::v1_3::{
        vk::{self, Handle, KhrSurfaceExtension, KhrSwapchainExtension},
        Device, HasBuilder,
    },
    vk::DeviceV1_0,
    Instance,
};
use winit::window::Window;

#[derive(Default)]
pub struct Swapchain {
    pub vk_swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    /// Create a [`Swapchain`]
    pub fn new(
        interface: &VulkanInterface,
        surface: vk::SurfaceKHR,
        indices: QueueFamilyIndices,
        window: &Window,
    ) -> Result<Self> {
        let mut builder = SwapchainBuilder::new(interface, surface, indices)?;
        builder.set_format(vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR);
        // Using FIFO to limit the framerate to the monitor refresh rate (hard vsync)
        builder.set_present_mode(vk::PresentModeKHR::FIFO);
        builder.set_extent(window);

        builder.build()
    }

    /// Cleanup the associated resources
    pub fn cleanup(&mut self, device: &Device) {
        unsafe {
            device.destroy_swapchain_khr(self.vk_swapchain, None);
            for view in self.views.drain(..) {
                device.destroy_image_view(view, None);
            }
        }
    }
}

struct SwapchainBuilder<'d> {
    device: &'d Device,
    support: SwapchainSupport,
    surface: vk::SurfaceKHR,
    indices: QueueFamilyIndices,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    extent: vk::Extent2D,
}

impl<'d> SwapchainBuilder<'d> {
    /// Initialize the builder
    fn new(
        interface: &'d VulkanInterface,
        surface: vk::SurfaceKHR,
        indices: QueueFamilyIndices,
    ) -> Result<Self> {
        let support =
            SwapchainSupport::get(&interface.instance, interface.physical_device, surface)?;

        Ok(Self {
            device: &interface.device,
            support,
            surface,
            indices,
            format: vk::SurfaceFormatKHR::default(),
            present_mode: vk::PresentModeKHR::default(),
            extent: vk::Extent2D::default(),
        })
    }

    /// Set the desired format.
    ///
    /// If the desired format is not available, it will fallback to a format
    /// compatible with the device.
    fn set_format(&mut self, format: vk::Format, color_space: vk::ColorSpaceKHR) {
        self.format = *self
            .support
            .formats
            .iter()
            .find(|f| f.format == format && f.color_space == color_space)
            .unwrap_or_else(|| &self.support.formats[0]);
    }

    /// Set the desired present mode.
    ///
    /// If the desired present mode is not available, it will fallback to a
    /// [`vk::PresentModeKHR::FIFO`] wich should always be available.
    fn set_present_mode(&mut self, present_mode: vk::PresentModeKHR) {
        self.present_mode = self
            .support
            .present_modes
            .iter()
            .find(|&&m| m == present_mode)
            .copied()
            .unwrap_or(vk::PresentModeKHR::FIFO);
    }

    /// Set the desired extent.
    ///
    /// For now the extent is set using swapchain support data.
    fn set_extent(&mut self, window: &Window) {
        let extent = if self.support.capabilities.current_extent.width == u32::MAX {
            let size = window.inner_size();
            let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
            vk::Extent2D::builder()
                .width(clamp(
                    self.support.capabilities.min_image_extent.width,
                    self.support.capabilities.max_image_extent.width,
                    size.width,
                ))
                .height(clamp(
                    self.support.capabilities.min_image_extent.height,
                    self.support.capabilities.max_image_extent.height,
                    size.height,
                ))
                .build()
        } else {
            self.support.capabilities.current_extent
        };
        self.extent = extent;
    }

    /// Build a [`Swapchain`]
    fn build(self) -> Result<Swapchain> {
        let mut image_count = self.support.capabilities.min_image_count + 1;
        if self.support.capabilities.max_image_count != 0
            && image_count > self.support.capabilities.max_image_count
        {
            image_count = self.support.capabilities.max_image_count;
        }
        let mut queue_family_indices = Vec::new();
        let image_sharing_mode = if self.indices.graphics == self.indices.present {
            vk::SharingMode::EXCLUSIVE
        } else {
            queue_family_indices.push(self.indices.graphics);
            queue_family_indices.push(self.indices.present);
            vk::SharingMode::CONCURRENT
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(self.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(self.support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            // NOTE - https://kylemayes.github.io/vulkanalia/swapchain/recreation.html#recreating-the-swapchain
            // It is possible to create a new swapchain while drawing commands on an image from the old swapchain are still in-flight.
            // You need to pass the previous swapchain to the old_swapchain field in the vk::SwapchainCreateInfoKHR struct
            // and destroy the old swapchain as soon as you've finished using it.
            .old_swapchain(vk::SwapchainKHR::null())
            .build();

        let vk_swapchain = unsafe { self.device.create_swapchain_khr(&info, None)? };
        let format = self.format.format;
        let images = unsafe { self.device.get_swapchain_images_khr(vk_swapchain)? };
        let views = images
            .iter()
            .map(|i| create_image_view(self.device, *i, format, vk::ImageAspectFlags::COLOR, 1))
            .collect::<Result<Vec<_>, _>>()?;
        let extent = self.extent;

        Ok(Swapchain {
            vk_swapchain,
            format,
            images,
            views,
            extent,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        unsafe {
            Ok(Self {
                capabilities: instance
                    .get_physical_device_surface_capabilities_khr(physical_device, surface)?,
                formats: instance
                    .get_physical_device_surface_formats_khr(physical_device, surface)?,
                present_modes: instance
                    .get_physical_device_surface_present_modes_khr(physical_device, surface)?,
            })
        }
    }
}
