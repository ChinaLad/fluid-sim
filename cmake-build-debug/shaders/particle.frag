#version 450 core

in vec2 LocalPos;
in vec4 ParticleColor;
out vec4 FragColor;

void main() {
    // Calculate distance from the center of the quad
    float distSq = dot(LocalPos, LocalPos);

    // Discard pixels outside the radius to create a circle
    if(distSq > 0.8) {
        discard;
    }

    // A nice fluid blue, slightly darker toward the edges for fake depth
    vec3 color = ParticleColor.rgb * (1 - distSq * 0.5);


    FragColor = vec4(color, ParticleColor.a);
}