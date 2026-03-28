#ifndef FLUID_SIM_VIS_H
#define FLUID_SIM_VIS_H

#pragma once
#include "../cmake-build-debug/include/glad/gl.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include "camera.h"
#include "shader.h"
#include "../simulation/particles.h"
#include "../simulation/params.h"

class Visualization {
public:
    GLFWwindow* window;
    Camera& camera;
    std::unique_ptr<Shader> shader;

    unsigned int vao, vbo, instanceVBO;
    std::vector<float> instanceBuffer;

    Visualization(Camera& cam, int width, int height);
    ~Visualization() = default;

    bool initWindow(int width, int height);
    void setupGeometry();

    static void scroll_callback(GLFWwindow* w, double xoffset, double yoffset);

    bool shouldClose() const;
    void processInput(float deltaTime);
    void updateParticles(const Particles& particles);
    void render(int n_particles);


    /**
     * Adjusts the viewport (area for rendering on the screen) when the window is
     * resized.
     *
     * @param w
     * @param width New window width
     * @param height New window height
     */
    void framebuffer_size_callback(GLFWwindow *w, const int width, const int height) {
        // defines affine transformation from normalized device coordinates to window coordinates
        glViewport(0, 0, width, height);
    }

    /**
     * Process mouse movements.
     *
     * @param w Window tracking mouse position
     * @param xpos Mouse cursor position on the X axis
     * @param ypos Mouse cursor position on the Y axis
     */
    void mouse_callback(GLFWwindow *w, double xpos, double ypos) {

    }
};

#endif //FLUID_SIM_VIS_H