#version 450 core

in vec2 LocalPos;
in vec4 ParticleColor;
out vec4 FragColor;

void main() {
    // Calculate distance from the center of the quad
    float distSq = dot(LocalPos, LocalPos);

    // Discard pixels outside the radius to create a circle
    if(distSq > 1.0) {
        discard;
    }

    // A nice fluid blue, slightly darker toward the edges for fake depth
    vec3 color;
    if (distSq <= 1.0 && distSq >= 0.8) {
        color = vec3(1.0, 1.0, 1.0);
    }
    else if (distSq < 0.2) {
        color = ParticleColor.rgb * 5*(0.2 - distSq * 0.6);
    } else {
        discard;
    }

    FragColor = vec4(color, ParticleColor.a);
}