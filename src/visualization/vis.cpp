#include "vis.h"

Visualization::Visualization(Camera& cam, int width, int height) : camera(cam) {
    initWindow(width, height);

    shader = std::make_unique<Shader>("shaders/particle.vert", "shaders/particle.frag");

    setupGeometry();
}

bool Visualization::initWindow(int width, int height) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, "Window", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    std::cout << "Window created." << std::endl;

    glfwMakeContextCurrent(window);
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    std::cout << "Window is made to be the current context." << std::endl;

    glfwSetWindowUserPointer(window, this);
    glfwSetScrollCallback(window, scroll_callback);

    glEnable(GL_DEPTH_TEST);
    return true;
}

void Visualization::setupGeometry() {
    float quad[] = { -0.05f, -0.05f,  0.05f, -0.05f,  -0.05f, 0.05f,
                      0.05f, -0.05f,  0.05f,  0.05f,  -0.05f, 0.05f };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &instanceVBO);

    glBindVertexArray(vao);

    // base geometry
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // instance positions
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1, 1);

    glBindVertexArray(0);
}

bool Visualization::shouldClose() const {
    return glfwWindowShouldClose(window);
}

void Visualization::processInput(float deltaTime) {
    glfwPollEvents();

    // ESCAPE pressed
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // move forwards on W press
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, TIME_DELTA);
    // move backwards on S press
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, TIME_DELTA);
    // move left on A press
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, TIME_DELTA);
    // move right on D press
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, TIME_DELTA);
}

void Visualization::updateParticles(const Particles& p) {
    instanceBuffer.resize(p.n_particles * 3);
    for (int i = 0; i < p.n_particles; ++i) {
        instanceBuffer[i*3 + 0] = p.x[i];
        instanceBuffer[i*3 + 1] = p.y[i];
        instanceBuffer[i*3 + 2] = p.z[i];
    }
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, instanceBuffer.size() * sizeof(float), instanceBuffer.data(), GL_DYNAMIC_DRAW);
}

void Visualization::render(int particleCount) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader->use();
    shader->setMat4("u_view", camera.GetViewMatrix());
    shader->setMat4("u_projection", camera.GetProjectionMatrix(1.0f)); // Assuming square window

    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, particleCount);

    glfwSwapBuffers(window);
}

void Visualization::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    auto* vis = static_cast<Visualization*>(glfwGetWindowUserPointer(window));
    vis->camera.ProcessMouseScroll((float)yoffset);
}