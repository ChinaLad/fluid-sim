#ifndef FLUID_SIM_CONFIG_H
#define FLUID_SIM_CONFIG_H

#include "params.h"

#include <iostream>
#include <fstream>
#include <sstream>

enum class ScaleMode {MACRO = 0, MESO = 1, MICRO = 2, SUBATOMIC = 3};


struct TypeProfile {
    float mass = 1.0f;
    float charge = 1.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f, a = 1.0f;
};

struct ForceProfile {
    float inv_sq_strength;

    float opt_dist;
    float bond_strength;
    float max_radius;
};

class SimulationConfig {
public:
    ScaleMode current_scale = ScaleMode::MESO;
    int n_particles = 0;
    float max_size = 0.2f;
    std::map<int, int> type_counts;

    ForceProfile interaction_matrix[NUM_PARTICLE_TYPES][NUM_PARTICLE_TYPES];
    TypeProfile type_profiles[NUM_PARTICLE_TYPES];

    void loadForceProfile(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error opening file " << filepath << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            char line_type;
            iss >> line_type;

            if (line_type == 'S') {
                int scale;
                if (iss >> scale) {
                    current_scale = static_cast<ScaleMode>(scale);
                }
            }
            else if (line_type == 'C') {
                // e.g., C 0 20000 (Spawn 20,000 of Type 0)
                int type_id, count;
                if (iss >> type_id >> count) {
                    type_counts[type_id] = count;
                    n_particles += count;
                }
            }

            if (line_type == 'T') {
                int id;
                if (iss >> id && id < NUM_PARTICLE_TYPES) {
                    iss >> type_profiles[id].mass
                        >> type_profiles[id].charge
                        >> type_profiles[id].r >> type_profiles[id].g
                        >> type_profiles[id].b >> type_profiles[id].a;
                }
            }
            else if (line_type == 'F') {
                int t1, t2;
                ForceProfile pf;
                if (iss >> t1 >> t2 >> pf.inv_sq_strength >> pf.opt_dist >> pf.bond_strength >> pf.max_radius) {
                    max_size = std::max(max_size, pf.max_radius);
                    interaction_matrix[t1][t2] = pf;
                    interaction_matrix[t2][t1] = pf; // Make symmetric
                }
            }
        }
    }
};

#endif //FLUID_SIM_CONFIG_H