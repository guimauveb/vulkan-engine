#![allow(unused)] // TODO: Remove

use anyhow::Result;
use std::{cmp::min, ffi::c_void};
use vulkanalia::{
    prelude::v1_3::vk,
    vk::{DeviceV1_0, HasBuilder},
    Device,
};

const MAX_SETS_PER_POOL: u32 = 4096;

/// Dynamically resizable descriptor pools.
pub struct DescriptorAllocator {
    device: Device,
    pool_ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl DescriptorAllocator {
    /// Constructor
    pub fn new(device: Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Result<Self> {
        let mut descriptor = Self {
            device,
            pool_ratios,
            full_pools: Vec::with_capacity(1),
            ready_pools: Vec::new(),
            sets_per_pool: (max_sets as f32 * 1.5) as u32,
        };

        descriptor.ready_pools.push(descriptor.create_pool()?);

        Ok(descriptor)
    }

    /// Reset all pools, pushing the now reset full pools to the ready pools list.
    pub fn clear_pools(&mut self) {
        for pool in &self.ready_pools {
            unsafe {
                self.device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty());
            }
        }
        for pool in self.full_pools.drain(..) {
            unsafe {
                self.device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty());
            }
            self.ready_pools.push(pool);
        }
    }

    /// Destroy all descriptor pools.
    pub fn destroy_pools(&mut self) {
        for pool in self.ready_pools.drain(..) {
            unsafe {
                self.device.destroy_descriptor_pool(pool, None);
            }
        }
        for pool in self.full_pools.drain(..) {
            unsafe {
                self.device.destroy_descriptor_pool(pool, None);
            }
        }
    }

    // TODO:
    /// Allocate a new [vk::DescriptorSet].
    pub fn allocate(
        &self,
        layout: vk::DescriptorSetLayout,
        p_next: *const c_void,
    ) -> Result<vk::DescriptorSet> {
        todo!();
    }

    /// Pick up a pool from the ready pools, or, if there is none, allocate a new one
    fn get_pool(&mut self) -> Result<vk::DescriptorPool> {
        let pool = if let Some(pool) = self.ready_pools.pop() {
            pool
        } else {
            let pool = self.create_pool()?;
            let sets_per_pool = (self.sets_per_pool as f32 * 1.5) as u32;
            self.sets_per_pool = if sets_per_pool > MAX_SETS_PER_POOL {
                MAX_SETS_PER_POOL
            } else {
                sets_per_pool
            };
            pool
        };

        Ok(pool)
    }

    /// Create a [vk::DescriptorPool].
    fn create_pool(&self) -> Result<vk::DescriptorPool> {
        let mut pool_sizes = Vec::new();
        for ratio in &self.pool_ratios {
            pool_sizes.push(
                vk::DescriptorPoolSize::builder()
                    .type_(ratio.descriptor_type)
                    .descriptor_count((ratio.ratio * self.sets_per_pool as f32) as u32)
                    .build(),
            );
        }
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(self.sets_per_pool)
            .pool_sizes(&pool_sizes)
            .build();
        let pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };

        Ok(pool)
    }
}

pub struct PoolSizeRatio {
    descriptor_type: vk::DescriptorType,
    ratio: f32,
}
