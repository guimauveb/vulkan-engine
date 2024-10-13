// TODO: - Border detection
// TODO: - Take framerate into account to adjust velocity
use super::{Mat4, Point3, Vec3};
use cgmath::{vec3, Angle, Deg, InnerSpace};
use winit::{dpi::PhysicalPosition, keyboard::Key};

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
    pub fn get_view_matrix(&self) -> Mat4 {
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

    /// Process keyboard key pressed
    pub fn process_key_pressed(&mut self, key: Key) {
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
                    self.eye += self.target * SPEED;
                    self.update();
                }
                "s" => {
                    self.eye -= self.target * SPEED;
                    self.update();
                }
                "a" => {
                    let mut left = self.up.cross(self.target).normalize();
                    left *= SPEED;
                    self.eye += left;
                    self.update();
                }
                "d" => {
                    let mut right = self.target.cross(self.up).normalize();
                    right *= SPEED;
                    self.eye += right;
                    self.update();
                }
                "q" => {
                    self.eye += self.up * SPEED;
                    self.update();
                }
                "e" => {
                    self.eye -= self.up * SPEED;
                    self.update();
                }
                _ => {}
            }
        }
    }

    /// Process mouse movement
    pub fn process_mouse_motion(&mut self, position: PhysicalPosition<f32>) {
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
}
