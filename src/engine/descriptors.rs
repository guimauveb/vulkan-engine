use anyhow::Result;
use vulkanalia::{
    prelude::v1_3::{vk, Device, HasBuilder},
    vk::{Cast, DeviceV1_0, ErrorCode, Handle},
};

const MAX_SETS_PER_POOL: u32 = 4096;

#[derive(Default)]
pub struct DescriptorLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorLayoutBuilder {
    /// Add a binding to the layout
    pub fn add_binding(&mut self, binding: u32, descriptor_type: vk::DescriptorType) {
        let bind = vk::DescriptorSetLayoutBinding::builder()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(1)
            .build();
        self.bindings.push(bind);
    }

    /// Clear all the bindings
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Build a [`vk::DescriptorSetLayout`]
    pub fn build<T: vk::ExtendsDescriptorSetLayoutCreateInfo>(
        mut self,
        device: &Device,
        shader_stages: vk::ShaderStageFlags,
        mut next: Option<impl Cast<Target = T>>,
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> Result<vk::DescriptorSetLayout> {
        for binding in &mut self.bindings {
            binding.stage_flags |= shader_stages;
        }
        let bindings = &self.bindings[..];
        let mut info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(bindings)
            .flags(flags);
        if let Some(next) = next.as_mut() {
            info = info.push_next(next);
        }

        Ok(unsafe { device.create_descriptor_set_layout(&info, None)? })
    }
}

/// Dynamically resizable descriptor pools.
#[derive(Default, Clone)]
pub struct DescriptorAllocator {
    pool_ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl DescriptorAllocator {
    /// Constructor. Allocates the first descriptor pool.
    pub fn new(device: &Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Result<Self> {
        let mut descriptor = Self {
            pool_ratios,
            full_pools: Vec::new(),
            ready_pools: Vec::with_capacity(1),
            sets_per_pool: (max_sets as f32 * 1.5) as u32,
        };

        descriptor.ready_pools.push(descriptor.create_pool(device)?);

        Ok(descriptor)
    }

    /// Allocate a new [`vk::DescriptorSet`].
    pub fn allocate(
        &mut self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let mut pool = self.get_or_create_pool(device)?;
        let layouts = &[layout];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(layouts);
        let set = match unsafe { device.allocate_descriptor_sets(&info) } {
            Err(err) => {
                if err == ErrorCode::OUT_OF_POOL_MEMORY || err == ErrorCode::FRAGMENTED_POOL {
                    self.full_pools.push(pool);
                    pool = self.get_or_create_pool(device)?;
                    let info = info.descriptor_pool(pool).build();
                    unsafe { device.allocate_descriptor_sets(&info)?[0] }
                } else {
                    return Err(err.into());
                }
            }
            Ok(sets) => sets[0],
        };
        self.ready_pools.push(pool);

        Ok(set)
    }

    /// Pick up a pool from the ready pools, or, if there is none, allocate a new one.
    fn get_or_create_pool(&mut self, device: &Device) -> Result<vk::DescriptorPool> {
        let pool = if let Some(pool) = self.ready_pools.pop() {
            pool
        } else {
            let pool = self.create_pool(device)?;
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

    /// Create a [`vk::DescriptorPool`].
    pub fn create_pool(&self, device: &Device) -> Result<vk::DescriptorPool> {
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
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(pool)
    }

    /// Reset all pools, pushing the now reset full pools to the ready pools list.
    pub fn clear_pools(&mut self, device: &Device) -> Result<()> {
        for pool in &self.ready_pools {
            unsafe {
                device.reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())?;
            }
        }
        for pool in self.full_pools.drain(..) {
            unsafe {
                device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())?;
            }
            self.ready_pools.push(pool);
        }

        Ok(())
    }

    /// Destroy all descriptor pools.
    pub fn destroy_pools(&mut self, device: &Device) {
        for pool in self.ready_pools.drain(..) {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }
        for pool in self.full_pools.drain(..) {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}

/// [`vk::DescriptorPool`] size ratio
#[derive(Default, Debug, Clone, Copy)]
pub struct PoolSizeRatio {
    descriptor_type: vk::DescriptorType,
    ratio: f32,
}

impl PoolSizeRatio {
    /// Constructor
    #[inline]
    pub fn new(descriptor_type: vk::DescriptorType, ratio: f32) -> Self {
        Self {
            descriptor_type,
            ratio,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct DescriptorWriter {
    image_infos: Vec<vk::DescriptorImageInfo>,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    writes: Vec<vk::WriteDescriptorSet>,
}

impl DescriptorWriter {
    pub fn write_image(
        &mut self,
        binding: u32,
        image: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
        descriptor_type: vk::DescriptorType,
    ) {
        self.image_infos.push(
            vk::DescriptorImageInfo::builder()
                .sampler(sampler)
                .image_view(image)
                .image_layout(layout)
                .build(),
        );
        let write = vk::WriteDescriptorSet::builder()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_type(descriptor_type)
            .image_info(&self.image_infos)
            .build();
        self.writes.push(write);
    }

    pub fn write_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        offset: vk::DeviceSize,
        descriptor_type: vk::DescriptorType,
    ) {
        self.buffer_infos.push(
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer)
                .offset(offset)
                .range(size)
                .build(),
        );
        let write = vk::WriteDescriptorSet::builder()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_type(descriptor_type)
            .buffer_info(&self.buffer_infos)
            .build();
        self.writes.push(write);
    }

    pub fn clear(&mut self) {
        self.image_infos.clear();
        self.writes.clear();
        self.buffer_infos.clear();
    }

    pub fn update_set(&mut self, device: &Device, set: vk::DescriptorSet) {
        for write in &mut self.writes {
            write.dst_set = set;
        }
        unsafe {
            device.update_descriptor_sets(&self.writes, &[] as &[vk::CopyDescriptorSet]);
        }
    }
}
