#ifndef FLUID_SIM_SIM_H
#define FLUID_SIM_SIM_H

#include "particles.h"
#include "params.h"

#include <iostream>

class Simulation {
public:
    Particles particles;

    Simulation(const Particles& p) {
        this->particles = p;
    }

    void applyForces();
    void applyGravity();
    void applyAttraction();
    void applyRepulsion();

    void updatePositions();
};
#endif //FLUID_SIM_SIM_H