# Vulkan based engine (work in progress)

A Vulkan 3D engine written in Rust and built using [vulkanalia](https://github.com/KyleMayes/vulkanalia) based on [vkguide.dev](https://vkguide.dev/).

![vulkan-engine](https://github.com/guimauveb/vulkan-engine/blob/master/screenshot.png?raw=true)

## Features 
TODO

## Disclaimer
⚠️  While the program runs without any validation error there might be errors, suboptimal logic or anti-patterns that neither the validation layers nor I can detect. ⚠️

## Example

Run the program in debug (enables the validation layers):

```bash
$ export RUST_LOG=debug
$ cargo r
```

Run the program in release:

```bash
$ cargo r --release
```

