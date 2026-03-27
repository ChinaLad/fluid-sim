#include "sim.h"

void Simulation::applyForces() {
    applyGravity();
}

void Simulation::applyGravity() {
    /**
     * TODO:
     * - reorder writes and read for more ILP
     * - unroll
     * - vectorize
     */

    for (int p = 0; p < particles.n_particles; p++) {
        particles.vy[p] -= GRAVITY * TIME_DELTA;
    }
}

void Simulation::updatePositions() {
    for (int p = 0; p < particles.n_particles; p++) {
        particles.x[p] += particles.vx[p] * TIME_DELTA;
    }
    for (int p = 0; p < particles.n_particles; p++) {
        particles.y[p] += particles.vy[p] * TIME_DELTA;
    }
    for (int p = 0; p < particles.n_particles; p++) {
        particles.z[p] += particles.vz[p] * TIME_DELTA;
    }

    for (int p = 0; p < particles.n_particles; p++) {
        std::cout << "x: " << particles.x[p]
        << " y: " << particles.y[p]
        <<" z: " << particles.z[p] << std::endl;
    }
}