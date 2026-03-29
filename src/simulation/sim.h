#ifndef FLUID_SIM_SIM_H
#define FLUID_SIM_SIM_H

#include "particles.h"
#include "params.h"

#include <iostream>

struct TypeProfile {
    float mass = 1.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f, a = 1.0f;
};

struct ForceProfile {
    float inv_sq_strength;

    float opt_dist;
    float bond_strength;
    float max_radius;
};

class Simulation {
public:
    Particles& particles;
    Particles sorted_buffer;

    ForceProfile interaction_matrix[NUM_PARTICLE_TYPES][NUM_PARTICLE_TYPES];
    TypeProfile type_profiles[NUM_PARTICLE_TYPES];

    // grid parameters
    float cell_size;
    int grid_width, grid_height, grid_depth, total_cells;

    // grid data structure
    std::vector<int> cell_start;
    std::vector<int> cell_end;

    struct ParticleHash {
        int cell_id;
        int particle_id;

        bool operator<(const ParticleHash& other) const {
            return cell_id < other.cell_id;
        }
    };
    std::vector<ParticleHash> hashes;

    float x_pos_border, x_neg_border, y_pos_border, y_neg_border, z_pos_border, z_neg_border;

    Simulation(Particles& p, float x_pos, float x_neg, float y_pos, float y_neg, float z_pos, float z_neg): particles(p), sorted_buffer(p.n_particles) {
        setBoundaries(x_pos, x_neg, y_pos, y_neg, z_pos, z_neg);

        cell_size = 0.5f;

        // Assuming your borders start at 0 or are shifted to be positive
        float sim_width = 4.0f;
        float sim_height = 4.0f;
        float sim_depth = 4.0f;

        grid_width = std::ceil(sim_width / cell_size);
        grid_height = std::ceil(sim_height / cell_size);
        grid_depth = std::ceil(sim_depth / cell_size);
        total_cells = grid_width * grid_height * grid_depth;

        cell_start.resize(total_cells, 0);
        cell_end.resize(total_cells, 0);
        hashes.resize(particles.n_particles);
    }

    void setBoundaries(float x_pos, float x_neg, float y_pos, float y_neg, float z_pos, float z_neg) {
        x_pos_border = x_pos;
        x_neg_border = x_neg;
        y_pos_border = y_pos;
        y_neg_border = y_neg;
        z_pos_border = z_pos;
        z_neg_border = z_neg;
    }

    void buildGrid();

    void applyForces();
    void applyGravity();
    void applyAttraction();
    void applyRepulsion();

    void updatePositions();

    void loadForceProfile(const std::string& filepath);
    void initTypes();
};
#endif //FLUID_SIM_SIM_H