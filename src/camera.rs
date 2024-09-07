// TODO: Expose camera controls in GUI
use super::{Mat4, Point3, Vec3};
use cgmath::{vec3, Angle, Deg, InnerSpace};
use winit::{
    dpi::PhysicalPosition,
    keyboard::{Key, NamedKey},
};

/// Mouse camera speed
const MOUSE_SPEED: f32 = 0.1;
/// Keyboard camera speed
const KEYBOARD_SPEED: f32 = 0.9;
/// Move the camera `EDGE_STEP`s towards the proper direction when the mouse is resting on one of the edges of the window.
const EDGE_STEP: f32 = 1.0;
/// Set the borders of the window to be `MARGIN` (pixels?) away from the actual borders.
const MARGIN: f32 = 1.0;

// TODO: Reposition cursor in the middle when the cursor enters back into the viewport
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    eye: Point3,
    target: Vec3,
    up: Vec3,
    view: Mat4,
    width: u32,
    height: u32,
    yaw: f32,
    pitch: f32,
    on_upper_edge: bool,
    on_lower_edge: bool,
    on_left_edge: bool,
    on_right_edge: bool,
    mouse_position: PhysicalPosition<f32>,
    enabled: bool,
}

impl Camera {
    pub fn new(eye: Point3, target: Vec3, up: Vec3, width: u32, height: u32) -> Self {
        Self::init(eye, target, up, width, height)
    }

    fn init(eye: Point3, target: Vec3, up: Vec3, width: u32, height: u32) -> Self {
        let target = target.normalize();
        let up = up.normalize();
        let view = Mat4::look_at_rh(eye, eye + target, up);

        Self {
            eye,
            target,
            up,
            view,
            width,
            height,
            yaw: -90.0,
            pitch: 0.0,
            on_upper_edge: false,
            on_lower_edge: false,
            on_left_edge: false,
            on_right_edge: false,
            mouse_position: PhysicalPosition::new(width as f32 / 2.0, height as f32 / 2.0),
            enabled: true,
        }
    }

    fn update(&mut self) {
        let (yaw_cos, yaw_sin, pitch_cos, pitch_sin) = (
            Deg(self.yaw).cos(),
            Deg(self.yaw).sin(),
            Deg(self.pitch).cos(),
            Deg(self.pitch).sin(),
        );
        let direction = vec3(yaw_cos * pitch_cos, pitch_sin, yaw_sin * pitch_cos);
        self.target = direction.normalize();
        self.view = Mat4::look_at_rh(self.eye, self.eye + self.target, self.up);
    }

    pub fn on_keyboard(&mut self, key: Key) {
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

        // FIXME - Arrow keys not detected
        match key {
            Key::Named(key) => match key {
                NamedKey::ArrowUp => {
                    self.eye += self.target * KEYBOARD_SPEED;
                    self.update();
                }
                NamedKey::ArrowDown => {
                    self.eye -= self.target * KEYBOARD_SPEED;
                    self.update();
                }
                NamedKey::ArrowLeft => {
                    let mut left = self.up.cross(self.target).normalize();
                    left *= KEYBOARD_SPEED;
                    self.eye += left;
                    self.update();
                }
                NamedKey::ArrowRight => {
                    let mut right = self.target.cross(self.up).normalize();
                    right *= KEYBOARD_SPEED;
                    self.eye += right;
                    self.update();
                }
                _ => {}
            },

            Key::Character(key) => match key.as_str() {
                "w" => {
                    self.eye += self.target * KEYBOARD_SPEED;
                    self.update();
                }
                "s" => {
                    self.eye -= self.target * KEYBOARD_SPEED;
                    self.update();
                }
                "a" => {
                    let mut left = self.up.cross(self.target).normalize();
                    left *= KEYBOARD_SPEED;
                    self.eye += left;
                    self.update();
                }
                "d" => {
                    let mut right = self.target.cross(self.up).normalize();
                    right *= KEYBOARD_SPEED;
                    self.eye += right;
                    self.update();
                }
                "q" => {
                    self.eye += self.up * KEYBOARD_SPEED;
                    self.update();
                }
                "e" => {
                    self.eye -= self.up * KEYBOARD_SPEED;
                    self.update();
                }

                _ => {}
            },
            // TODO: Handle more inputs to change camera behaviour (speed, etc)
            _ => {}
        }
    }

    pub fn on_mouse(&mut self, position: PhysicalPosition<f32>) {
        if !self.enabled {
            return;
        }

        let (delta_x, delta_y) = (
            position.x - self.mouse_position.x,
            self.mouse_position.y - position.y,
        );
        self.mouse_position.x = position.x;
        self.mouse_position.y = position.y;

        self.yaw += delta_x * MOUSE_SPEED;
        self.pitch += delta_y * MOUSE_SPEED;
        self.pitch = self.pitch.clamp(-90.0, 90.0);

        // self.on_border()

        self.update();
    }

    // FIXME - Border detection
    fn on_border(&mut self, _position: PhysicalPosition<f32>) {
        // if delta_x == 0.0 {
        //     if position.x <= MARGIN {
        //         self.on_left_edge = true;
        //     } else if position.x >= (self.width as f32 - MARGIN) {
        //         self.on_right_edge = true;
        //     }
        // } else {
        //     self.on_left_edge = false;
        //     self.on_right_edge = false;
        // }

        // if delta_y == 0.0 {
        //     if position.y == 0.0 {
        //         self.on_upper_edge = true;
        //     } else if position.y == self.height as f32 {
        //         self.on_lower_edge = true;
        //     }
        // } else {
        //     self.on_upper_edge = false;
        //     self.on_lower_edge = false;
        // }
    }

    pub fn on_render(&mut self) {
        let mut should_update = false;
        if self.on_left_edge {
            self.yaw -= EDGE_STEP;
            should_update = true;
        } else if self.on_right_edge {
            self.yaw += EDGE_STEP;
            should_update = true;
        }

        if self.on_upper_edge {
            if self.pitch > -90.0 {
                self.pitch -= EDGE_STEP;
                should_update = true;
            }
        } else if self.on_lower_edge && self.pitch < 90.0 {
            self.pitch += EDGE_STEP;
            should_update = true;
        }

        if should_update {
            self.update();
        }
    }

    pub fn view(&self) -> Mat4 {
        self.view
    }

    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
