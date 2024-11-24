// TODO: - Border detection
// TODO: - Take framerate into account to adjust velocity
use super::{Mat4, Point3, Vec3};
use cgmath::{vec3, Angle, Deg, InnerSpace};
use log::warn;
use winit::{dpi::PhysicalPosition, event::MouseScrollDelta, keyboard::Key};

/// Camera speed
const SPEED: f32 = 0.2;

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    eye: Point3,
    target: Vec3,
    up: Vec3,
    yaw: f32,
    pitch: f32,
    mouse_position: PhysicalPosition<f32>,
    enabled: bool,
}

impl Camera {
    /// Constructor
    pub fn new(eye: Point3, target: Vec3, up: Vec3, width: u32, height: u32) -> Self {
        let target = target.normalize();
        let up = up.normalize();

        Self {
            eye,
            target,
            up,
            yaw: -90.0,
            pitch: 0.0,
            mouse_position: PhysicalPosition::new(width as f32 / 2.0, height as f32 / 2.0),
            enabled: true,
        }
    }

    /// Get current camera view matrix
    #[inline]
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.eye + self.target, self.up)
    }

    /// Update camera
    pub fn update(&mut self) {
        let (yaw_cos, yaw_sin, pitch_cos, pitch_sin) = (
            Deg(self.yaw).cos(),
            Deg(self.yaw).sin(),
            Deg(self.pitch).cos(),
            Deg(self.pitch).sin(),
        );
        let direction = vec3(yaw_cos * pitch_cos, pitch_sin, yaw_sin * pitch_cos);
        self.target = direction.normalize();
    }

    /// Handle keyboard key pressed
    pub fn handle_key_pressed(&mut self, key: Key) {
        if let Key::Character(key) = &key {
            match key.as_str() {
                "c" => {
                    self.enabled = false;
                }
                "C" => {
                    self.enabled = true;
                }
                _ => (),
            }
        }

        if !self.enabled {
            return;
        }

        if let Key::Character(key) = key {
            match key.as_str() {
                "w" => {
                    self.move_forward();
                }
                "s" => {
                    self.move_backward();
                }
                "a" => {
                    self.move_left();
                }
                "d" => {
                    self.move_right();
                }
                "q" => {
                    self.move_up();
                }
                "e" => {
                    self.move_down();
                }
                _ => {}
            }
        }
        self.update();
    }

    /// Process mouse cursor movement
    pub fn handle_cursor_motion(&mut self, position: PhysicalPosition<f32>) {
        if !self.enabled {
            return;
        }

        let (delta_x, delta_y) = (
            position.x - self.mouse_position.x,
            self.mouse_position.y - position.y,
        );
        self.mouse_position.x = position.x;
        self.mouse_position.y = position.y;

        self.yaw += delta_x * SPEED;
        self.pitch += delta_y * SPEED;
        self.pitch = self.pitch.clamp(-90.0, 90.0);
    }

    /// Process mouse wheel scroll
    pub fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        if !self.enabled {
            return;
        }

        match delta {
            MouseScrollDelta::LineDelta(horizontal, vertical) => {
                if vertical > 0.0 {
                    self.move_forward();
                } else if vertical < 0.0 {
                    self.move_backward();
                }
                if horizontal > 0.0 {
                    self.move_left();
                } else if horizontal < 0.0 {
                    self.move_right();
                }
            }
            MouseScrollDelta::PixelDelta(_) => {
                warn!("Mouse scroll in pixels unhandled");
            }
        }
        self.update();
    }

    #[inline]
    fn move_forward(&mut self) {
        self.eye += self.target * SPEED;
    }

    #[inline]
    fn move_backward(&mut self) {
        self.eye -= self.target * SPEED;
    }

    #[inline]
    fn move_left(&mut self) {
        let mut left = self.up.cross(self.target).normalize();
        left *= SPEED;
        self.eye += left;
    }

    #[inline]
    fn move_right(&mut self) {
        let mut right = self.target.cross(self.up).normalize();
        right *= SPEED;
        self.eye += right;
    }

    #[inline]
    fn move_up(&mut self) {
        self.eye += self.up * SPEED;
    }

    #[inline]
    fn move_down(&mut self) {
        self.eye -= self.up * SPEED;
    }
}
