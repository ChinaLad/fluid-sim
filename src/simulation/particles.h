#ifndef FLUID_SIM_PARTICLES_H
#define FLUID_SIM_PARTICLES_H
#include <cstdlib>
#include <random>
#include <new>

class Particles {
public:
    // array of particle types
    int* p_type;

    // arrays of coordinate components
    float* x;
    float* y;
    float* z;

    // arrays of velocity components
    float* vx;
    float* vy;
    float* vz;

    // arrays of force components
    float* fx;
    float* fy;
    float* fz;

    float* masses;
    float* masses_inv;

    // arrays of color values
    float* r;
    float* g;
    float* b;
    float* a;

    int n_particles{};

    Particles() {
        p_type = nullptr;
        x = nullptr;
        y = nullptr;
        z = nullptr;
        vx = nullptr;
        vy = nullptr;
        vz = nullptr;
        fx = nullptr;
        fy = nullptr;
        fz = nullptr;
        masses = nullptr;
        masses_inv = nullptr;
        r = nullptr;
        g = nullptr;
        b = nullptr;
        a = nullptr;
    }
    Particles(const int n) {
        this->n_particles = n;
        p_type = new int[n];
        x = new (std::align_val_t(32)) float[n];
        y = new (std::align_val_t(32)) float[n];
        z = new (std::align_val_t(32)) float[n];
        vx = new (std::align_val_t(32)) float[n]{};
        vy = new (std::align_val_t(32)) float[n]{};
        vz = new (std::align_val_t(32)) float[n]{};
        fx = new (std::align_val_t(32)) float[n]{};
        fy = new (std::align_val_t(32)) float[n]{};
        fz = new (std::align_val_t(32)) float[n]{};
        masses = new (std::align_val_t(32)) float[n];
        masses_inv = new (std::align_val_t(32)) float[n];
        r = new (std::align_val_t(32)) float[n];
        g = new (std::align_val_t(32)) float[n];
        b = new (std::align_val_t(32)) float[n];
        a = new (std::align_val_t(32)) float[n];
    }
    ~Particles()= default;

    void initParticlesRandomly(float x_pos_border, float x_neg_border,
                           float y_pos_border, float y_neg_border,
                           float z_pos_border, float z_neg_border) {

        // Setup random number generator and distributions for each axis
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distX(x_neg_border, x_pos_border);
        std::uniform_real_distribution<float> distY(y_neg_border, y_pos_border);
        std::uniform_real_distribution<float> distZ(z_neg_border, z_pos_border);

        for (int i = 0; i < n_particles; i++) {
            p_type[i] = 0;

            x[i] = distX(gen);
            y[i] = distY(gen);
            z[i] = distZ(gen);

            // Initialize velocities to 0
            vx[i] = 0.0f;
            vy[i] = 0.0f;
            vz[i] = 0.0f;

            fx[i] = 0.0f;
            fy[i] = 0.0f;
            fz[i] = 0.0f;

            float m = 1.0f;
            masses[i] = m;
            masses_inv[i] = 1/m;

            r[i] = 0.0f;
            g[i] = 0.0f;
            b[i] = 1.0f;
            a[i] = 1.0f;
        }
    }
};

#endif //FLUID_SIM_PARTICLES_H