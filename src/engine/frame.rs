use super::{descriptors::DescriptorAllocator, memory::Allocation, ALLOCATOR};
use vulkanalia::prelude::v1_3::{vk, Device, DeviceV1_0};

/// Structures and commands required to draw a given frame
#[derive(Default)]
pub struct Frame {
    pub swapchain_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
    pub frame_descriptors: DescriptorAllocator,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub allocations: Vec<Allocation>,
}

impl Frame {
    /// Destroy resources
    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_fence(self.render_fence, None);
            device.destroy_semaphore(self.render_semaphore, None);
            device.destroy_semaphore(self.swapchain_semaphore, None);
            self.frame_descriptors.destroy_pools(device);
            self.deallocate(device);
        }
    }

    /// Add an allocation linked to this [`Frame`].
    pub fn add_allocation(&mut self, allocation: Allocation) {
        self.allocations.push(allocation);
    }

    /// Deallocate all allocations of this [`Frame`]
    pub fn deallocate(&mut self, device: &Device) {
        for allocation in self.allocations.drain(..) {
            ALLOCATOR.deallocate(device, &allocation);
        }
    }
}
