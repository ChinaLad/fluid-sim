#ifndef FLUID_SIM_SIM_H
#define FLUID_SIM_SIM_H

#include "particles.h"
#include "config.h"

#include <immintrin.h>

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

        // integrate velocities
        constexpr float DRAG = 0.98f;
        __m256 drag_vec = _mm256_set1_ps(DRAG);
        __m256 time_delta_vec = _mm256_set1_ps(TIME_DELTA);

        int p = 0;
        for(; p+8 <= particles.n_particles; p+=8) {
            __m256 vx_p_vec = _mm256_load_ps(&particles.vx[p]);
            __m256 vy_p_vec = _mm256_load_ps(&particles.vy[p]);
            __m256 vz_p_vec = _mm256_load_ps(&particles.vz[p]);

            __m256 fx_p_vec = _mm256_load_ps(&particles.fx[p]);
            __m256 fy_p_vec = _mm256_load_ps(&particles.fy[p]);
            __m256 fz_p_vec = _mm256_load_ps(&particles.fz[p]);

            __m256 mas_inv_vec = _mm256_load_ps(&particles.masses_inv[p]);
            __m256 temp = _mm256_mul_ps(time_delta_vec, mas_inv_vec);

            vx_p_vec = _mm256_fmadd_ps(fx_p_vec, temp, vx_p_vec);
            vy_p_vec = _mm256_fmadd_ps(fy_p_vec, temp, vy_p_vec);
            vz_p_vec = _mm256_fmadd_ps(fz_p_vec, temp, vz_p_vec);

            vx_p_vec = _mm256_mul_ps(drag_vec, vx_p_vec);
            vy_p_vec = _mm256_mul_ps(drag_vec, vy_p_vec);
            vz_p_vec = _mm256_mul_ps(drag_vec, vz_p_vec);

            _mm256_store_ps(&particles.vx[p], vx_p_vec);
            _mm256_store_ps(&particles.vy[p], vy_p_vec);
            _mm256_store_ps(&particles.vz[p], vz_p_vec);
        }

        for(; p < particles.n_particles; p++) {
            float mass_inv = particles.masses_inv[p];
            particles.vx[p] = (particles.vx[p] + (particles.fx[p] * mass_inv) * TIME_DELTA) * DRAG;
            particles.vy[p] = (particles.vy[p] + (particles.fy[p] * mass_inv) * TIME_DELTA) * DRAG;
            particles.vz[p] = (particles.vz[p] + (particles.fz[p] * mass_inv) * TIME_DELTA) * DRAG;
        }
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