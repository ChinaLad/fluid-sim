#include "sim.h"
#include <immintrin.h>

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
        float x_p_new = particles.x[p] + particles.vx[p] * TIME_DELTA;
        if (x_p_new >= x_pos_border) particles.x[p] = x_pos_border;
        else if (x_p_new <= x_neg_border) particles.x[p] = x_neg_border;
        else particles.x[p] = x_p_new;
    }
    for (int p = 0; p < particles.n_particles; p++) {
        float y_p_new = particles.y[p] + particles.vy[p] * TIME_DELTA;
        if (y_p_new >= y_pos_border) particles.y[p] = y_pos_border;
        else if (y_p_new <= y_neg_border) particles.y[p] = y_neg_border;
        else particles.y[p] = y_p_new;
    }
    for (int p = 0; p < particles.n_particles; p++) {
        float z_p_new = particles.z[p] + particles.vz[p] * TIME_DELTA;
        if (z_p_new >= z_pos_border) particles.z[p] = z_pos_border;
        else if (z_p_new <= z_neg_border) particles.z[p] = z_neg_border;
        else particles.z[p] = z_p_new;
    }

    for (int p = 0; p < particles.n_particles; p++) {
        std::cout << "x: " << particles.x[p]
        << " y: " << particles.y[p]
        <<" z: " << particles.z[p] << std::endl;
    }
}