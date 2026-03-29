#version 450 core

in vec2 uv;
in vec4 ParticleColor;
in vec3 ParticleVel;

out vec4 FragColor;

void main() {
    float distSq = dot(uv, uv);
    if (distSq > 1.0) {
        discard;
    }

    float z = sqrt(1.0 - distSq);
    vec3 normal = normalize(vec3(uv.x, uv.y, z)); // Normal vector pointing out from the sphere

    float speed = length(ParticleVel);

    vec3 heatColor = mix(ParticleColor.rgb, vec3(1.0, 1.0, 1.0), clamp(speed * 0.15, 0.0, 1.0));

    vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0)); // Light coming from top-right
    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));  // Camera looking straight down Z

    // Diffuse lighting (matte surface)
    float diff = max(dot(normal, lightDir), 0.0);

    // Specular lighting (shiny highlight)
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0); // 32 is the shininess

    float ambientStrength = clamp(0.3 + (speed * 0.05), 0.3, 0.9);
    vec3 ambient = ambientStrength * heatColor;

    vec3 diffuse = diff * heatColor;
    vec3 specular = vec3(0.6) * spec; // Bright white highlight

    float alpha = ParticleColor.a * smoothstep(1.0, 0.85, sqrt(distSq));

    FragColor = vec4(ambient + diffuse + specular, alpha);
}