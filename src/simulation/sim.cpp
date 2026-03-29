#include "sim.h"
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

void Simulation::buildGrid() {
    for (int i = 0; i < particles.n_particles; i++) {
        int cx = std::clamp((int)((particles.x[i] - x_neg_border)/cell_size), 0, grid_width - 1);
        int cy = std::clamp((int)((particles.y[i] - y_neg_border)/cell_size), 0, grid_height - 1);
        int cz = std::clamp((int)((particles.z[i] - z_neg_border)/cell_size), 0, grid_depth - 1);

        int cell_id = cx + (cy * grid_width) + (cz * grid_width * grid_height);

        hashes[i].cell_id = cell_id;
        hashes[i].particle_id = i;
    }

    std::sort(hashes.begin(), hashes.end());

    for (int i = 0; i < particles.n_particles; i++) {
        int original_idx = hashes[i].particle_id;

        sorted_buffer.p_type[i] = particles.p_type[original_idx];

        sorted_buffer.x[i] = particles.x[original_idx];
        sorted_buffer.y[i] = particles.y[original_idx];
        sorted_buffer.z[i] = particles.z[original_idx];

        sorted_buffer.vx[i] = particles.vx[original_idx];
        sorted_buffer.vy[i] = particles.vy[original_idx];
        sorted_buffer.vz[i] = particles.vz[original_idx];

        sorted_buffer.masses[i] = particles.masses[original_idx];
        sorted_buffer.masses_inv[i] = particles.masses_inv[original_idx];

        sorted_buffer.r[i] = particles.r[original_idx];
        sorted_buffer.g[i] = particles.g[original_idx];
        sorted_buffer.b[i] = particles.b[original_idx];
        sorted_buffer.a[i] = particles.a[original_idx];
    }

    std::swap(particles.x, sorted_buffer.x);
    std::swap(particles.y, sorted_buffer.y);
    std::swap(particles.z, sorted_buffer.z);
    std::swap(particles.vx, sorted_buffer.vx);
    std::swap(particles.vy, sorted_buffer.vy);
    std::swap(particles.vz, sorted_buffer.vz);
    std::swap(particles.p_type, sorted_buffer.p_type);
    std::swap(particles.masses, sorted_buffer.masses);
    std::swap(particles.masses_inv, sorted_buffer.masses_inv);
    std::swap(particles.r, sorted_buffer.r);
    std::swap(particles.g, sorted_buffer.g);
    std::swap(particles.b, sorted_buffer.b);
    std::swap(particles.a, sorted_buffer.a);

    std::fill(cell_start.begin(), cell_start.end(), 0);
    std::fill(cell_end.begin(), cell_end.end(), 0);

    for (int i = 0; i < particles.n_particles; i++) {
        int c = hashes[i].cell_id;

        if (i == 0 || c != hashes[i - 1].cell_id) {
            cell_start[c] = i;
        }

        if (i == particles.n_particles - 1 || c != hashes[i + 1].cell_id) {
            cell_end[c] = i + 1;
        }
    }
}

void Simulation::applyForces() {
    applyGravity();
    applyAttraction();
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
    __m256 small_dist = _mm256_set1_ps(0.00001f);
    __m256 zero = _mm256_setzero_ps();
    __m256 min_safe_dist = _mm256_set1_ps(0.05f);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int p = 0; p < particles.n_particles; p++) {
        int type_p = particles.p_type[p];
        float x_p = particles.x[p];
        float y_p = particles.y[p];
        float z_p = particles.z[p];
        float mass_p = particles.masses[p];

        __m256 x_p_vec = _mm256_set1_ps(x_p);
        __m256 y_p_vec = _mm256_set1_ps(y_p);
        __m256 z_p_vec = _mm256_set1_ps(z_p);
        __m256 masses_vec = _mm256_set1_ps(mass_p);

        int cx = std::clamp((int)((x_p - x_neg_border) / cell_size), 0, grid_width - 1);
        int cy = std::clamp((int)((y_p - y_neg_border) / cell_size), 0, grid_height - 1);
        int cz = std::clamp((int)((z_p - z_neg_border) / cell_size), 0, grid_depth - 1);

        __m256 fx_acc = _mm256_setzero_ps();
        __m256 fy_acc = _mm256_setzero_ps();
        __m256 fz_acc = _mm256_setzero_ps();

        float scalar_fx = 0.0f;
        float scalar_fy = 0.0f;
        float scalar_fz = 0.0f;

        // loop over the 3x3x3 neighboring cells (including its own cell)
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = cx + dx;
                    int ny = cy + dy;
                    int nz = cz + dz;

                    if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height || nz < 0 || nz >= grid_depth) continue;

                    int neighbor_cell = nx + (ny * grid_width) + (nz * grid_width * grid_height);

                    int start_idx = cell_start[neighbor_cell];
                    int end_idx = cell_end[neighbor_cell];

                    int q = start_idx;

                    for (; q+8 <= end_idx; q+=8) {
                        __m256 x_q_vec = _mm256_load_ps(&particles.x[q]);
                        __m256 y_q_vec = _mm256_load_ps(&particles.y[q]);
                        __m256 z_q_vec = _mm256_load_ps(&particles.z[q]);

                        __m256 mass_q_vec = _mm256_load_ps(&particles.masses[q]);

                        int t0 = particles.p_type[q + 0];
                        int t1 = particles.p_type[q + 1];
                        int t2 = particles.p_type[q + 2];
                        int t3 = particles.p_type[q + 3];
                        int t4 = particles.p_type[q + 4];
                        int t5 = particles.p_type[q + 5];
                        int t6 = particles.p_type[q + 6];
                        int t7 = particles.p_type[q + 7];

                        const ForceProfile& pr0 = interaction_matrix[type_p][t0];
                        const ForceProfile& pr1 = interaction_matrix[type_p][t1];
                        const ForceProfile& pr2 = interaction_matrix[type_p][t2];
                        const ForceProfile& pr3 = interaction_matrix[type_p][t3];
                        const ForceProfile& pr4 = interaction_matrix[type_p][t4];
                        const ForceProfile& pr5 = interaction_matrix[type_p][t5];
                        const ForceProfile& pr6 = interaction_matrix[type_p][t6];
                        const ForceProfile& pr7 = interaction_matrix[type_p][t7];

                        __m256 max_radius_vec = _mm256_setr_ps(
                            pr0.max_radius, pr1.max_radius, pr2.max_radius, pr3.max_radius,
                            pr4.max_radius, pr5.max_radius, pr6.max_radius, pr7.max_radius
                        );

                        __m256 max_radius_sq_vec = _mm256_mul_ps(max_radius_vec, max_radius_vec);

                        __m256 inv_sq_strength_vec = _mm256_setr_ps(
                            pr0.inv_sq_strength, pr1.inv_sq_strength, pr2.inv_sq_strength, pr3.inv_sq_strength,
                            pr4.inv_sq_strength, pr5.inv_sq_strength, pr6.inv_sq_strength, pr7.inv_sq_strength
                        );

                        __m256 opt_dist_vec = _mm256_setr_ps(
                            pr0.opt_dist, pr1.opt_dist, pr2.opt_dist, pr3.opt_dist,
                            pr4.opt_dist, pr5.opt_dist, pr6.opt_dist, pr7.opt_dist
                        );

                        __m256 bond_strength_vec = _mm256_setr_ps(
                            pr0.bond_strength, pr1.bond_strength, pr2.bond_strength, pr3.bond_strength,
                            pr4.bond_strength, pr5.bond_strength, pr6.bond_strength, pr7.bond_strength
                        );


                        __m256 dist_x_vec = _mm256_sub_ps(x_p_vec, x_q_vec);
                        __m256 dist_y_vec = _mm256_sub_ps(y_p_vec, y_q_vec);
                        __m256 dist_z_vec = _mm256_sub_ps(z_p_vec, z_q_vec);

                        __m256 dist_sq_vec = _mm256_mul_ps(dist_x_vec, dist_x_vec);
                        dist_sq_vec = _mm256_fmadd_ps(dist_y_vec, dist_y_vec, dist_sq_vec);
                        dist_sq_vec = _mm256_fmadd_ps(dist_z_vec, dist_z_vec, dist_sq_vec);

                        __m256 dist_inv_vec = _mm256_rsqrt_ps(dist_sq_vec);
                        __m256 dist_vec = _mm256_mul_ps(dist_sq_vec, dist_inv_vec);

                        __m256 skip_iter_mask = _mm256_or_ps(
                            _mm256_cmp_ps(dist_sq_vec, max_radius_sq_vec, _CMP_GT_OQ),
                            _mm256_cmp_ps(dist_sq_vec, small_dist, _CMP_LE_OQ)
                        );

                        __m256 force_vec = zero;

                        // macro: gravity
                        __m256 skip_macro_vec = _mm256_cmp_ps(inv_sq_strength_vec, zero, _CMP_EQ_OQ);

                        __m256 safe_dist_vec = _mm256_max_ps(dist_vec, min_safe_dist);
                        __m256 temp1 = _mm256_mul_ps(inv_sq_strength_vec, masses_vec);
                        temp1 = _mm256_mul_ps(temp1, mass_q_vec);
                        __m256 safe_dist_sq_vec = _mm256_mul_ps(safe_dist_vec, safe_dist_vec);

                        __m256 gravity = _mm256_sub_ps(zero, _mm256_div_ps(temp1, safe_dist_sq_vec));
                        gravity = _mm256_blendv_ps(gravity, zero, skip_macro_vec);

                        force_vec = _mm256_add_ps(force_vec, gravity);

                        // micro: Lennard-Jones
                        __m256 skip_micro_vec = _mm256_cmp_ps(bond_strength_vec, zero, _CMP_EQ_OQ);
                        __m256 ratio_vec = _mm256_mul_ps(opt_dist_vec, dist_inv_vec);
                        __m256 ratio2_vec = _mm256_mul_ps(ratio_vec, ratio_vec);
                        __m256 ratio6_vec = _mm256_mul_ps(ratio2_vec, _mm256_mul_ps(ratio2_vec, ratio2_vec));
                        __m256 ratio12_min_6_vec = _mm256_fmsub_ps(ratio6_vec, ratio6_vec, ratio6_vec);

                        __m256 micro = _mm256_mul_ps(bond_strength_vec, ratio12_min_6_vec);
                        micro = _mm256_blendv_ps(micro, zero, skip_micro_vec);

                        force_vec = _mm256_add_ps(force_vec, micro);


                        __m256 force_mult = _mm256_mul_ps(force_vec, dist_inv_vec);

                        force_mult = _mm256_blendv_ps(force_mult, zero, skip_iter_mask);

                        fx_acc = _mm256_fmadd_ps(dist_x_vec, force_mult, fx_acc);
                        fy_acc = _mm256_fmadd_ps(dist_y_vec, force_mult, fy_acc);
                        fz_acc = _mm256_fmadd_ps(dist_z_vec, force_mult, fz_acc);
                    }

                    for (; q < end_idx; q++) {
                        if (p == q) continue;

                        int type_q = particles.p_type[q];
                        const ForceProfile& profile = interaction_matrix[type_p][type_q];

                        float dist_x = x_p - particles.x[q];
                        float dist_y = y_p - particles.y[q];
                        float dist_z = z_p - particles.z[q];

                        float dist_sq = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;

                        if (dist_sq > profile.max_radius * profile.max_radius || dist_sq <= 0.00001f) continue;

                        float dist = std::sqrt(dist_sq);
                        float dist_inv = 1.0f / dist;
                        float force = 0.0f;

                        // macro: gravity
                        if (profile.inv_sq_strength != 0.0f) {
                            float safe_dist = std::max(dist, 0.05f);
                            force -= (profile.inv_sq_strength * mass_p * particles.masses[q]) / (safe_dist * safe_dist);
                        }

                        // micro: Lennard-Jones
                        if (profile.bond_strength != 0.0f) {
                            float ratio = profile.opt_dist * dist_inv;
                            float ratio2 = ratio * ratio;
                            float ratio6 = ratio2 * ratio2 * ratio2;
                            float ratio12 = ratio6 * ratio6;
                            force += profile.bond_strength * (ratio12 - ratio6);
                        }

                        // apply force only to P
                        scalar_fx += (dist_x * dist_inv) * force;
                        scalar_fy += (dist_y * dist_inv) * force;
                        scalar_fz += (dist_z * dist_inv) * force;
                    }
                }
            }
        }

        alignas(32) float f_x_arr[8];
        alignas(32) float f_y_arr[8];
        alignas(32) float f_z_arr[8];

        _mm256_store_ps(f_x_arr, fx_acc);
        _mm256_store_ps(f_y_arr, fy_acc);
        _mm256_store_ps(f_z_arr, fz_acc);

        for (int i = 0; i < 8; ++i) {
            scalar_fx += f_x_arr[i];
            scalar_fy += f_y_arr[i];
            scalar_fz += f_z_arr[i];
        }

        // Finally, write to memory exactly once per particle
        particles.fx[p] = scalar_fx;
        particles.fy[p] = scalar_fy;
        particles.fz[p] = scalar_fz;
    }

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

void Simulation::loadForceProfile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        char line_type;
        iss >> line_type;

        if (line_type == 'T') {
            int id;
            if (iss >> id && id < NUM_PARTICLE_TYPES) {
                iss >> type_profiles[id].mass
                    >> type_profiles[id].r >> type_profiles[id].g
                    >> type_profiles[id].b >> type_profiles[id].a;
            }
        }
        else if (line_type == 'F') {
            int t1, t2;
            ForceProfile pf;
            if (iss >> t1 >> t2 >> pf.inv_sq_strength >> pf.opt_dist >> pf.bond_strength >> pf.max_radius) {
                interaction_matrix[t1][t2] = pf;
                interaction_matrix[t2][t1] = pf; // Make symmetric
            }
        }
    }
}

void Simulation::initTypes() {
    for (int p = 0; p < particles.n_particles; p++) {
        int type = (p == 0) ? 0 : 1; // Example: 1 Sun, the rest Planets
        type = 1;
        particles.p_type[p] = type;

        // Fetch properties from the loaded preset
        particles.masses[p] = type_profiles[type].mass;
        particles.masses_inv[p] = 1.0f / particles.masses[p];

        particles.r[p] = type_profiles[type].r;
        particles.g[p] = type_profiles[type].g;
        particles.b[p] = type_profiles[type].b;
        particles.a[p] = type_profiles[type].a;
    }
}