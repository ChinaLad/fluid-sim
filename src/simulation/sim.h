#ifndef FLUID_SIM_SIM_H
#define FLUID_SIM_SIM_H

#include "particles.h"
#include "config.h"

class Simulation {
public:
    ScaleMode current_scale = ScaleMode::MESO;

    ForceProfile interaction_matrix[NUM_PARTICLE_TYPES][NUM_PARTICLE_TYPES];
    TypeProfile type_profiles[NUM_PARTICLE_TYPES];

    Particles& particles;
    Particles sorted_buffer;

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

        cell_size = 0.2f;

        // Assuming your borders start at 0 or are shifted to be positive
        float sim_width = x_pos - x_neg;
        float sim_height = y_pos - y_neg;
        float sim_depth = z_pos - z_neg;

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

    void applyForces() {
        (this->*computeForcesPtr)();
    }
    void applyGravity();
    void applyAttraction();
    void applyRepulsion();

    void updatePositions();

    void applyConfig(const SimulationConfig& config);
private:
    void (Simulation::*computeForcesPtr)() = &Simulation::computeMeso;
    void computeMacro(); // Gravity only
    void computeMeso();  // Lennard-Jones
    void computeMicro(); // Electrostatics + Repulsion
    void computeSubatomic(); // Strong Force
};
#endif //FLUID_SIM_SIM_H