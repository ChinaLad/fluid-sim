#ifndef FLUID_SIM_SIM_H
#define FLUID_SIM_SIM_H

#include "particles.h"
#include "params.h"

#include <iostream>

struct ForceProfile {
    float inv_sq_strength;

    float opt_dist;
    float bond_strength;
    float max_radius;
};

inline ForceProfile interaction_matrix[NUM_PARTICLE_TYPES][NUM_PARTICLE_TYPES];

class Simulation {
public:
    Particles particles;
    float x_pos_border, x_neg_border, y_pos_border, y_neg_border, z_pos_border, z_neg_border;

    Simulation(const Particles& p) {
        this->particles = p;
    }

    void setBoundaries(float x_pos, float x_neg, float y_pos, float y_neg, float z_pos, float z_neg) {
        x_pos_border = x_pos;
        x_neg_border = x_neg;
        y_pos_border = y_pos;
        y_neg_border = y_neg;
        z_pos_border = z_pos;
        z_neg_border = z_neg;
    }

    void applyForces();
    void applyGravity();
    void applyAttraction();
    void applyRepulsion();

    void updatePositions();

    void loadForceProfile(const std::string& filepath);
};
#endif //FLUID_SIM_SIM_H