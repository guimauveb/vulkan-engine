// TODO:- Custom build to automatically compile shaders when compiling the program
//      - Check pub/private code
//      - Docstring
//      - unsafe code handling

mod buffer;
mod camera;
mod command_buffers;
mod device;
mod engine;
mod frame_rate_limiter;
mod gui;
mod image;
mod mesh;
mod swapchain;
mod texture;
mod vertex;

use anyhow::Result;
use cgmath::{Matrix4, Vector2, Vector3, Vector4};
use device::QueueFamilyIndices;
use engine::Engine;
use log::error;
use winit::{
    dpi::LogicalSize,
    event::{Event, KeyEvent, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

pub type Point3 = cgmath::Point3<f32>;
pub type Vec2 = Vector2<f32>;
pub type Vec3 = Vector3<f32>;
pub type Vec4 = Vector4<f32>;
pub type Mat4 = Matrix4<f32>;

fn main() -> Result<()> {
    pretty_env_logger::init();
    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("vulkan-engine")
        .with_inner_size(LogicalSize::new(1920, 1080))
        // TODO: Add toggle in GUI
        // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        .build(&event_loop)?;
    window.set_cursor_visible(true);

    let mut engine = unsafe { Engine::new(&window)? };
    let (mut destroyed, mut minimized) = (false, false);

    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::NewEvents(StartCause::Poll) if !minimized && !destroyed && !target.exiting() => {
                window.request_redraw();
            }
            Event::WindowEvent { event, .. } => {
                if let Some(gui) = engine.data.egui_integration.as_mut() {
                    _ = gui.handle_event(&window, &event);
                }

                match event {
                    WindowEvent::RedrawRequested
                        if !minimized && !destroyed && !target.exiting() =>
                    {
                        if let Err(err) = unsafe { engine.render(&window) } {
                            error!("Cannot render frame: {err}");
                            unsafe {
                                if let Err(err) = engine.destroy() {
                                    error!("Error when destroying engine: {err}");
                                }
                            }
                            destroyed = true;
                            target.exit();
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => match event {
                        KeyEvent {
                            logical_key: Key::Named(key),
                            ..
                        } => {
                            if key == NamedKey::Escape && !destroyed {
                                unsafe {
                                    if let Err(err) = engine.destroy() {
                                        error!("Error when destroying engine: {err}");
                                    }
                                }
                                destroyed = true;
                                target.exit();
                            }
                        }
                        KeyEvent {
                            logical_key: key, ..
                        } => {
                            engine.camera.on_keyboard(key);
                        }
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        engine.camera.on_mouse(position.cast());
                    }
                    WindowEvent::Resized(size) => {
                        if size.width == 0 || size.height == 0 {
                            minimized = true;
                        } else {
                            minimized = false;
                            engine.resized = true;
                        }
                    }
                    WindowEvent::CloseRequested => {
                        if !destroyed {
                            unsafe {
                                if let Err(err) = engine.destroy() {
                                    error!("Error when destroying engine: {err}");
                                }
                            }
                            destroyed = true;
                            target.exit();
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
