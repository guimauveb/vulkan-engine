#!/bin/bash 

# Compute shaders
/usr/bin/glslc ./shaders/src/gradient.comp -o ./shaders/spv/gradient.spv --target-env=vulkan1.3 --target-spv=spv1.6
/usr/bin/glslc ./shaders/src/rgb_to_rgba.comp -o ./shaders/spv/rgb_to_rgba.spv --target-env=vulkan1.3 --target-spv=spv1.6

/usr/bin/glslc ./shaders/src/colored_triangle.vert -o ./shaders/spv/colored_triangle_vert.spv --target-env=vulkan1.3 --target-spv=spv1.6
/usr/bin/glslc ./shaders/src/colored_triangle_mesh.vert -o ./shaders/spv/colored_triangle_mesh_vert.spv --target-env=vulkan1.3 --target-spv=spv1.6
/usr/bin/glslc ./shaders/src/colored_triangle.frag -o ./shaders/spv/colored_triangle_frag.spv --target-env=vulkan1.3 --target-spv=spv1.6

/usr/bin/glslc ./shaders/src/tex_image.frag -o ./shaders/spv/tex_image_frag.spv --target-env=vulkan1.3 --target-spv=spv1.6

/usr/bin/glslc ./shaders/src/mesh.vert -o ./shaders/spv/mesh_vert.spv --target-env=vulkan1.3 --target-spv=spv1.6
/usr/bin/glslc ./shaders/src/mesh.frag -o ./shaders/spv/mesh_frag.spv --target-env=vulkan1.3 --target-spv=spv1.6

# GUI shaders
/usr/bin/glslc ./shaders/gui/src/vert.vert -o ./shaders/gui/spv/vert.spv --target-env=vulkan1.3 --target-spv=spv1.6
/usr/bin/glslc ./shaders/gui/src/frag.frag -o ./shaders/gui/spv/frag.spv --target-env=vulkan1.3 --target-spv=spv1.6
