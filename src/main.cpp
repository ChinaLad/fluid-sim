#include "simulation/particles.h"
#include "simulation/sim.h"
#include "visualization/vis.h"

int main() {
    SimulationConfig config;
    config.loadForceProfile("presets/micro.txt");

    Particles p{config.n_particles};

    p.initParticlesRandomly(config ,3, -3, 3, -3, 3, -3);

    Simulation sim{p, 6, -6, 6, -6, 6, -6};
    sim.applyConfig(config);
    Camera camera{};
    Visualization v{camera, 1000, 1000};

    sim.updatePositions();
    v.updateParticles(p);
    v.render(p.n_particles);
    while (!v.shouldClose()) {
        v.processInput(TIME_DELTA);

        for (int s = 0; s < SUB_STEPS; s++) {
            sim.buildGrid();
            sim.applyForces();
            sim.updatePositions();
        }

        v.updateParticles(p);
        v.render(p.n_particles);
    }
}