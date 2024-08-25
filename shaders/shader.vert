#version 450

#extension GL_EXT_buffer_reference : require

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outTexCoord;

struct Vertex {
    vec3 position;
    vec3 color;
    vec2 texCoord;
    vec3 normal;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
    Vertex vertices[];
};

// Push constants block
layout(push_constant) uniform PushConstants
{
    mat4 render_matrix;
    VertexBuffer vertexBuffer;
} pc;

void main()
{
    // Load vertex data from device adress
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];

    // Output data
    gl_Position = ubo.proj * ubo.view * pc.render_matrix * vec4(v.position, 1.0f);
    outColor = v.color;
    outTexCoord = v.texCoord;
}
