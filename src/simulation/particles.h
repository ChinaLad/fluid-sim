#ifndef FLUID_SIM_PARTICLES_H
#define FLUID_SIM_PARTICLES_H
#include <cstdlib>
#include <random>

class Particles {
public:
    // arrays of coordinate components
    float* x;
    float* y;
    float* z;

    // arrays of velocity components
    float* vx;
    float* vy;
    float* vz;

    float* masses;

    int n_particles{};

    Particles() {
        x = nullptr;
        y = nullptr;
        z = nullptr;
        vx = nullptr;
        vy = nullptr;
        vz = nullptr;
        masses = nullptr;
    }
    Particles(const int n) {
        this->n_particles = n;
        x = new float[n];
        y = new float[n];
        z = new float[n];
        vx = new float[n]{};
        vy = new float[n]{};
        vz = new float[n]{};
        masses = new float[n];
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
            x[i] = distX(gen);
            y[i] = distY(gen);
            z[i] = distZ(gen);

            // Initialize velocities to 0
            vx[i] = 0.0f;
            vy[i] = 0.0f;
            vz[i] = 0.0f;
        }
    }
};

#endif //FLUID_SIM_PARTICLES_H