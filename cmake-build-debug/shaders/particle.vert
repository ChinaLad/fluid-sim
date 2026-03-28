#version 450 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aInstancePos;
layout (location = 2) in vec4 aInstanceColor;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 LocalPos;
out vec4 ParticleColor;

void main() {
    // Map from [-0.05, 0.05] local space to [-1.0, 1.0] for the fragment shader
    LocalPos = aPos * 20.0;
    ParticleColor = aInstanceColor;

    // Extract camera axes from the view matrix for perfect billboarding
    vec3 cameraRight = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 cameraUp    = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);

    // Calculate world position so the quad faces the camera
    vec3 worldPos = aInstancePos
    + cameraRight * aPos.x
    + cameraUp * aPos.y;

    gl_Position = u_projection * u_view * vec4(worldPos, 1.0);
}