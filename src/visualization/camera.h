#ifndef FLUID_SIM_CAMERA_H
#define FLUID_SIM_CAMERA_H

#pragma once
#include "../cmake-build-debug/include/glad/gl.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

// Default camera values
constexpr float YAW = -90.0f;
constexpr float PITCH = 0.0f;
constexpr float SPEED = 5.0f;
constexpr float RADIUS = 10.0f;


// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera {
public:
    // camera Attributes
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    // euler Angles
    float yaw;
    float pitch;
    float radius;

    // camera options
    float movementSpeed;

    // constructor with vectors
    Camera(
        float radius = RADIUS,
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
        float yaw = YAW,
        float pitch = PITCH) : movementSpeed(SPEED), radius(radius) {
        this->worldUp = up;
        this->yaw = yaw;
        this->pitch = pitch;
        updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix() {
        return glm::lookAt(position, glm::vec3(0.0f, 0.0f, 0.0f), up);
    }

    glm::mat4 GetProjectionMatrix(float aspectRatio) const {
        return glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 100.0f);
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(Camera_Movement direction, float deltaTime) {
        float velocity = movementSpeed * deltaTime;
        if (direction == FORWARD)
            pitch += velocity;
        if (direction == BACKWARD)
            pitch -= velocity;
        if (direction == LEFT)
            yaw -= velocity;
        if (direction == RIGHT)
            yaw += velocity;

        if (pitch > 89.0f)  pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        updateCameraVectors();
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true) {

    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset) {
        radius -= yoffset;
        if (radius < 1.0f)
            radius = 1.0f;
        updateCameraVectors();
    }

    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(position, position + front, up);
    }

private:
    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors() {
        position.x = radius * cos(glm::radians(pitch)) * cos(glm::radians(yaw));
        position.y = radius * sin(glm::radians(pitch));
        position.x = radius * cos(glm::radians(pitch)) * sin(glm::radians(yaw));

        front = glm::normalize(glm::vec3(0.0f) - position);
        // also re-calculate the Right and Up vector
        right = glm::normalize(glm::cross(front, worldUp));
        // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        up = glm::normalize(glm::cross(right, front));
    }
};

#endif //FLUID_SIM_CAMERA_H