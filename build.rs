use std::{
    io::{stderr, stdout, Write},
    process::Command,
};

fn main() {
    let output = Command::new("sh")
        .arg("./shaders/compile.sh")
        .output()
        .expect("Failed to compile shaders");
    // TODO: Panic if shader compile script failed
    println!("Status: {}", output.status);
    stdout().write_all(&output.stdout).unwrap();
    stderr().write_all(&output.stderr).unwrap();

    let watch_files = [
        "./shaders/compile.sh",
        "./shaders/compute.comp",
        "./shaders/colored_triangle.vert",
        "./shaders/colored_triangle.frag",
        "./shaders/colored_triangle_mesh.frag",
        "./shaders/tex_image.frag",
        "./shaders/mesh.vert",
        "./shaders/mesh.frag",
    ];
    for file in watch_files {
        println!("cargo::rerun-if-changed={file}");
    }
}
