use minstant::Instant;

/// Frame rate limiter
pub struct FrameRateLimiter {
    /// Minimum time to wait before rendering a new frame (in seconds)
    frame_cap: Option<f32>,
    // Time of the last loop
    last_update_time: Instant,
    // Time of the last render
    last_frame_time: Instant,
}

impl FrameRateLimiter {
    /// Constructor
    pub fn new(frame_cap: Option<f32>) -> Self {
        Self {
            frame_cap,
            last_update_time: Instant::now(),
            last_frame_time: Instant::now(),
        }
    }

    // TODO: Set from GUI
    /// Update the frame cap
    pub fn _set_frame_cap(&mut self, frame_cap: Option<f32>) {
        self.frame_cap = frame_cap;
    }

    /// Returns `true` if enough time has passed to render a new frame
    pub fn render(&mut self) -> bool {
        if let Some(frame_cap) = self.frame_cap {
            let now = Instant::now();
            self.last_update_time = now;
            // TODO: Use delta time time if necessary (for physics, tweening, etc.)
            // let _delta_time = (now - last_update_time).as_secs_f32();
            if (now - self.last_frame_time).as_secs_f32() >= frame_cap {
                self.last_frame_time = now;
                return true;
            }
            false
        } else {
            true
        }
    }
}
