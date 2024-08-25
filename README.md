# Vulkan engine (work in progress)

A basic Vulkan renderer written in Rust and built around [vulkanalia](https://github.com/KyleMayes/vulkanalia), [egui](https://github.com/emilk/egui), [egui-winit](https://github.com/emilk/egui/tree/master/crates/egui-winit) and [winit](https://github.com/rust-windowing/winit).

![vulkan-engine](https://github.com/guimauveb/vulkan-engine/blob/master/vulkan-engine.png?raw=true)

## Integration with egui and winit
With the help of [egui-winit-ash-integration](https://github.com/MatchaChoco010/egui-winit-ash-integration) I managed to successfully integrate `egui`, `winit`, `egui_winit` and `vulkanalia`.

## Disclaimer
**The code needs a major refactoring** (that I am currently working on while learning more about graphics programming and Vulkan.)

⚠️  While the program seems to work perfectly well and without any validation error, because I'm completely new to Vulkan and graphics programming in general I probably did some mistakes that neither the validation layers nor I can detect. ⚠️

## Example

Run the program in debug (enables the validation layers):

```bash
$ cargo r
```

Run the program in release:

```bash
$ cargo r --release
```

