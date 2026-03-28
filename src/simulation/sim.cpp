#include "sim.h"
#include <immintrin.h>
#include <fstream>
#include <sstream>
#include <iostream>

void Simulation::applyForces() {
    applyGravity();
}

void Simulation::applyGravity() {
    const __m256 time_vec = _mm256_set1_ps(TIME_DELTA);
    const __m256 gravity_vec = _mm256_set1_ps(-GRAVITY);

    int p = 0;
    for (; p+8 <= particles.n_particles; p+=8) {
        __m256 vy_p_vec = _mm256_load_ps(&particles.vy[p]);
        vy_p_vec = _mm256_fmadd_ps(time_vec, gravity_vec, vy_p_vec);
        _mm256_store_ps(&particles.vy[p], vy_p_vec);
    }

    for (; p < particles.n_particles; p++) {
        particles.vy[p] -= GRAVITY * TIME_DELTA;
    }
}

void Simulation::applyAttraction() {
    for (int p = 0; p < particles.n_particles; p++) {
        float x_p = particles.x[p];
        float y_p = particles.y[p];
        float z_p = particles.z[p];

        float vx_p = particles.vx[p];
        float vy_p = particles.vy[p];
        float vz_p = particles.vz[p];
        for (int q = p + 1; q < particles.n_particles; q++) {
            float x_q = particles.x[q];
            float y_q = particles.y[q];
            float z_q = particles.z[q];

            float vx_q = particles.vx[q];
            float vy_q = particles.vy[q];
            float vz_q = particles.vz[q];

            float dist_sq = (x_p - x_q) * (x_p - x_q) + (y_p - y_q) * (y_p - y_q) + (z_p - z_q) * (z_p - z_q);
        }
    }
}

void Simulation::applyRepulsion() {

}

void Simulation::updatePositions() {
    const __m256 time_vec = _mm256_set1_ps(TIME_DELTA);
    const __m256 bounce_vec = _mm256_set1_ps(-BOUNCE_COEF);

    const __m256 x_pos_vec = _mm256_set1_ps(x_pos_border);
    const __m256 x_neg_vec = _mm256_set1_ps(x_neg_border);

    int p = 0;

    for (; p+8 <= particles.n_particles; p+=8) {
        __m256 x_p_vec = _mm256_load_ps(&particles.x[p]);
        __m256 vx_p_vec = _mm256_load_ps(&particles.vx[p]);
        x_p_vec = _mm256_fmadd_ps(vx_p_vec, time_vec, x_p_vec);

        // x out of positive boundary
        __m256 out_pos_vec = _mm256_cmp_ps(x_p_vec, x_pos_vec, _CMP_GE_OQ);
        x_p_vec = _mm256_or_ps(_mm256_and_ps(out_pos_vec, x_pos_vec), _mm256_andnot_ps(out_pos_vec, x_p_vec));

        // x out of negative boundary
        __m256 out_neg_vec = _mm256_cmp_ps(x_p_vec, x_neg_vec, _CMP_LE_OQ);
        x_p_vec = _mm256_or_ps(_mm256_and_ps(out_neg_vec, x_neg_vec), _mm256_andnot_ps(out_neg_vec, x_p_vec));

        // bounce from borders
        __m256 invert_v_mask = _mm256_or_ps(out_pos_vec, out_neg_vec);
        vx_p_vec = _mm256_blendv_ps(vx_p_vec, _mm256_mul_ps(vx_p_vec, bounce_vec), invert_v_mask);

        _mm256_store_ps(&particles.x[p], x_p_vec);
        _mm256_store_ps(&particles.vx[p], vx_p_vec);
    }

    for (; p < particles.n_particles; p++) {
        float x_p = particles.x[p];
        float vx_p = particles.vx[p];

        float x_p_new = x_p + vx_p * TIME_DELTA;
        if (x_p_new >= x_pos_border) {
            x_p_new = x_pos_border;
            vx_p = -vx_p * BOUNCE_COEF;
        }
        else if (x_p_new <= x_neg_border) {
            x_p_new = x_neg_border;
            vx_p = -vx_p * BOUNCE_COEF;
        }

        particles.x[p] = x_p_new;
        particles.vx[p] = vx_p;
    }

    p = 0;

    __m256 y_pos_vec = _mm256_set1_ps(y_pos_border);
    __m256 y_neg_vec = _mm256_set1_ps(y_neg_border);
    for (; p+8 <= particles.n_particles; p+=8) {
        __m256 y_p_vec = _mm256_load_ps(&particles.y[p]);
        __m256 vy_p_vec = _mm256_load_ps(&particles.vy[p]);
        y_p_vec = _mm256_fmadd_ps(vy_p_vec, time_vec, y_p_vec);

        // y out of positive boundary
        __m256 out_pos_vec = _mm256_cmp_ps(y_p_vec, y_pos_vec, _CMP_GE_OQ);
        y_p_vec = _mm256_or_ps(_mm256_and_ps(out_pos_vec, y_pos_vec), _mm256_andnot_ps(out_pos_vec, y_p_vec));

        // y out of negative boundary
        __m256 out_neg_vec = _mm256_cmp_ps(y_p_vec, y_neg_vec, _CMP_LE_OQ);
        y_p_vec = _mm256_or_ps(_mm256_and_ps(out_neg_vec, y_neg_vec), _mm256_andnot_ps(out_neg_vec, y_p_vec));

        // bounce from borders
        __m256 invert_v_mask = _mm256_or_ps(out_pos_vec, out_neg_vec);
        vy_p_vec = _mm256_blendv_ps(vy_p_vec, _mm256_mul_ps(vy_p_vec, bounce_vec), invert_v_mask);

        _mm256_store_ps(&particles.y[p], y_p_vec);
        _mm256_store_ps(&particles.vy[p], vy_p_vec);
    }

    for (; p < particles.n_particles; p++) {
        float y_p = particles.y[p];
        float vy_p = particles.vy[p];

        float y_p_new = y_p + vy_p * TIME_DELTA;
        if (y_p_new >= y_pos_border) {
            y_p_new = y_pos_border;
            vy_p = -vy_p * BOUNCE_COEF;
        }
        else if (y_p_new <= y_neg_border) {
            y_p_new = y_neg_border;
            vy_p = -vy_p * BOUNCE_COEF;
        }

        particles.y[p] = y_p_new;
        particles.vy[p] = vy_p;
    }

    p = 0;

    __m256 z_pos_vec = _mm256_set1_ps(z_pos_border);
    __m256 z_neg_vec = _mm256_set1_ps(z_neg_border);
    for (; p+8 <= particles.n_particles; p+=8) {
        __m256 z_p_vec = _mm256_load_ps(&particles.z[p]);
        __m256 vz_p_vec = _mm256_load_ps(&particles.vz[p]);
        z_p_vec = _mm256_fmadd_ps(vz_p_vec, time_vec, z_p_vec);

        // z out of positive boundary
        __m256 out_pos_vec = _mm256_cmp_ps(z_p_vec, z_pos_vec, _CMP_GE_OQ);
        z_p_vec = _mm256_or_ps(_mm256_and_ps(out_pos_vec, z_pos_vec), _mm256_andnot_ps(out_pos_vec, z_p_vec));

        // z out of negative boundary
        __m256 out_neg_vec = _mm256_cmp_ps(z_p_vec, y_neg_vec, _CMP_LE_OQ);
        z_p_vec = _mm256_or_ps(_mm256_and_ps(out_neg_vec, z_neg_vec), _mm256_andnot_ps(out_neg_vec, z_p_vec));

        // bounce from borders
        __m256 invert_v_mask = _mm256_or_ps(out_pos_vec, out_neg_vec);
        vz_p_vec = _mm256_blendv_ps(vz_p_vec, _mm256_mul_ps(vz_p_vec, bounce_vec), invert_v_mask);

        _mm256_store_ps(&particles.z[p], z_p_vec);
        _mm256_store_ps(&particles.vz[p], vz_p_vec);
    }

    for (; p < particles.n_particles; p++) {
        float z_p = particles.z[p];
        float vz_p = particles.vy[p];

        float z_p_new = z_p + vz_p * TIME_DELTA;
        if (z_p_new >= z_pos_border) {
            z_p_new = z_pos_border;
            vz_p = -vz_p * BOUNCE_COEF;
        }
        else if (z_p_new <= z_neg_border) {
            z_p_new = z_neg_border;
            vz_p = -vz_p * BOUNCE_COEF;
        }

        particles.z[p] = z_p_new;
        particles.vz[p] = vz_p;
    }
}

void Simulation::loadForceProfile(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open preset: " << filepath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int type1, type2;
        ForceProfile profile;

        // Read the values from the line
        if (iss >> type1 >> type2 >> profile.inv_sq_strength >> profile.opt_dist >> profile.bond_strength >> profile.max_radius) {

            // Assign to the matrix
            interaction_matrix[type1][type2] = profile;

            // Optional: Make it symmetric so A->B is the same as B->A
            interaction_matrix[type2][type1] = profile;
                }
    }
    std::cout << "Loaded preset: " << filepath << std::endl;
}