#version 450 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
layout (location = 2) in vec3 aInstancePos;
layout (location = 3) in vec4 aInstanceColor;
layout (location = 4) in vec3 aInstanceVel;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 uv;
out vec4 ParticleColor;
out vec3 ParticleVel;

void main() {
    uv = aUV;
    ParticleColor = aInstanceColor;
    ParticleVel = aInstanceVel;

    vec3 cameraRight = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 cameraUp = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);

    // Build the world position so the quad perfectly faces the camera
    vec3 worldPos = aInstancePos + (cameraRight * aPos.x) + (cameraUp * aPos.y);

    gl_Position = u_projection * u_view * vec4(worldPos, 1.0);
}