use super::{image::create_image_view, VulkanInterface};
use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0, DeviceV1_2, HasBuilder, InstanceV1_0};

/// Vulkan memory allocator
pub struct Allocator;

impl Allocator {
    /// Create a new [`Allocation`].
    pub fn allocate_buffer(
        &self,
        interface: &VulkanInterface,
        info: vk::BufferCreateInfo,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Allocation> {
        let buffer = unsafe { interface.device.create_buffer(&info, None)? };
        let requirements = unsafe { interface.device.get_buffer_memory_requirements(buffer) };
        let mut memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index(interface, properties, requirements)?);

        let shader_device_address = info.usage & vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            == vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let memory = if shader_device_address {
            let mut flags = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            memory_info = memory_info.push_next(&mut flags);
            unsafe { interface.device.allocate_memory(&memory_info, None)? }
        } else {
            unsafe { interface.device.allocate_memory(&memory_info, None)? }
        };

        unsafe {
            interface.device.bind_buffer_memory(buffer, memory, 0)?;
        }

        let device_address = if shader_device_address {
            let info = vk::BufferDeviceAddressInfo::builder()
                .buffer(buffer)
                .build();
            Some(unsafe { interface.device.get_buffer_device_address(&info) })
        } else {
            None
        };

        Ok(Allocation {
            buffer,
            memory,
            size: requirements.size,
            device_address,
        })
    }

    /// Destroy the buffer and allocated resources.
    pub fn deallocate(&self, device: &Device, allocation: &Allocation) {
        allocation.destroy(device);
    }

    /// Create a new [`AllocatedImage`]
    pub fn allocate_image(
        &self,
        interface: &VulkanInterface,
        info: AllocatedImageInfo,
    ) -> Result<AllocatedImage> {
        let mip_levels = if info.mipmapped {
            (info.extent.width.max(info.extent.height) as f32)
                .log2()
                .floor() as u32
                + 1
        } else {
            1
        };
        let img_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(info.format)
            .extent(info.extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            // Not using MSAA for now
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(info.usage)
            .build();
        let image = unsafe { interface.device.create_image(&img_info, None)? };

        let requirements = unsafe { interface.device.get_image_memory_requirements(image) };
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index(
                interface,
                // Always allocate images on dedicated GPU memory
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                requirements,
            )?)
            .build();
        let memory = unsafe { interface.device.allocate_memory(&memory_info, None)? };
        unsafe { interface.device.bind_image_memory(image, memory, 0)? };
        let allocation = Allocation {
            buffer: vk::Buffer::default(),
            memory,
            size: requirements.size,
            device_address: None,
        };

        let aspect_flags = if info.format == vk::Format::D32_SFLOAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let image_view = create_image_view(
            &interface.device,
            image,
            info.format,
            aspect_flags,
            mip_levels,
        )?;

        Ok(AllocatedImage {
            image,
            image_view,
            allocation,
            image_extent: info.extent,
            image_format: info.format,
        })
    }

    /// Destroy the image and allocated resources.
    pub fn deallocate_image(&self, device: &Device, mut image: AllocatedImage) {
        image.destroy(device);
    }
}

/// A Vulkan memory allocation
#[derive(Default, Debug, Clone, Copy)]
pub struct Allocation {
    /// Buffer
    pub buffer: vk::Buffer,
    /// Memory
    pub memory: vk::DeviceMemory,
    /// Allocation size in bytes
    pub size: vk::DeviceSize,
    /// Device address if the buffer was allocated with the `SHADER_DEVICE_ADDRESS` flag
    device_address: Option<vk::DeviceAddress>,
}

impl Allocation {
    /// Map allocated memory
    pub fn mapped_memory<T>(&self, device: &Device) -> Result<*mut T> {
        unsafe {
            Ok(device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())?
                .cast())
        }
    }

    /// Unmap the memory
    pub fn unmap_memory(&self, device: &Device) {
        unsafe {
            device.unmap_memory(self.memory);
        }
    }

    /// Return the buffer device address
    ///
    /// # Error
    /// Errors if the buffer was not allocated with [`vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`]
    pub fn device_address(&self) -> Result<vk::DeviceAddress> {
        self.device_address
            .ok_or_else(|| anyhow!("Memory allocated without `SHADER_DEVICE_ADDRESS` flag"))
    }

    /// Destroy the buffer and release the associated memory.
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
}

impl AllocatedImage {
    /// Destroy the resources
    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
        }
        self.allocation.destroy(device);
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct AllocatedImageInfo {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub mipmapped: bool,
}

impl AllocatedImageInfo {
    pub fn extent(mut self, extent: vk::Extent3D) -> Self {
        self.extent = extent;
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = format;
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn mipmapped(mut self, mipmapped: bool) -> Self {
        self.mipmapped = mipmapped;
        self
    }
}

fn memory_type_index(
    interface: &VulkanInterface,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = unsafe {
        interface
            .instance
            .get_physical_device_memory_properties(interface.physical_device)
    };

    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
}
