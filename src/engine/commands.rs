use anyhow::Result;
use vulkanalia::{
    prelude::v1_3::{vk, Device, DeviceV1_0, HasBuilder},
    vk::DeviceV1_3,
};

/// Helper to submit immediate commands to the GPU. Holds immediate submit structures.
pub struct ImmediateSubmit {
    cmd_pool: vk::CommandPool,
    cmd_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    queue: vk::Queue,
}

impl ImmediateSubmit {
    /// Constructor
    #[inline]
    pub fn new(
        cmd_pool: vk::CommandPool,
        cmd_buffer: vk::CommandBuffer,
        fence: vk::Fence,
        queue: vk::Queue,
    ) -> Self {
        Self {
            cmd_pool,
            cmd_buffer,
            fence,
            queue,
        }
    }

    /// Submit commands to be executed immediately
    pub fn execute(
        &self,
        device: &Device,
        submission: impl Fn(vk::CommandBuffer) -> Result<()>,
    ) -> Result<()> {
        unsafe {
            device.reset_fences(&[self.fence])?;
            device.reset_command_buffer(self.cmd_buffer, vk::CommandBufferResetFlags::empty())?;
        }

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe {
            device.begin_command_buffer(self.cmd_buffer, &begin_info)?;
        }
        submission(self.cmd_buffer)?;
        unsafe {
            device.end_command_buffer(self.cmd_buffer)?;
        }

        let cmd_infos = &[vk::CommandBufferSubmitInfo::builder()
            .command_buffer(self.cmd_buffer)
            .device_mask(0)
            .build()];
        let submit_infos = &[vk::SubmitInfo2::builder()
            .command_buffer_infos(cmd_infos)
            .build()];
        unsafe {
            device.queue_submit2(self.queue, submit_infos, self.fence)?;
            device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        Ok(())
    }

    /// Cleanup resources
    pub fn cleanup(&self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.cmd_pool, None);
            device.destroy_fence(self.fence, None);
        }
    }
}
