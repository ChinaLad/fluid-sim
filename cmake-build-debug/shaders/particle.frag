#version 450 core

in vec2 LocalPos;
out vec4 FragColor;

void main() {
    // Calculate distance from the center of the quad
    float distSq = dot(LocalPos, LocalPos);

    // Discard pixels outside the radius to create a circle
    if(distSq > 1.0) {
        discard;
    }

    // A nice fluid blue, slightly darker toward the edges for fake depth
    vec3 color = vec3(0.2, 0.6, 1.0) * (1.0 - distSq * 0.4);

    FragColor = vec4(color, 1.0);
}