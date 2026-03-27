#ifndef FLUID_SIM_PARTICLES_H
#define FLUID_SIM_PARTICLES_H
#include <cstdlib>

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

    void initParticlesRandomly(int lowX, int highX, int lowY, int highY, int lowZ, int highZ) {
        for (int i = 0; i < n_particles; i++) {
            x[i] = static_cast<float>(lowX + rand() % (highX - lowX));
            y[i] = static_cast<float>(lowX + rand() % (highX - lowX));
            z[i] = static_cast<float>(lowX + rand() % (highX - lowX));
        }
    };
};

#endif //FLUID_SIM_PARTICLES_H