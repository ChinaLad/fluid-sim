#include "simulation/particles.h"
#include "simulation/sim.h"
#include "visualization/vis.h"

int main() {
    Particles p{4000};

    p.initParticlesRandomly(2, -2, 2, -2, 2, -2);

    Simulation sim{p, 4, -4, 4, -4, 4, -4};
    sim.loadForceProfile("presets/atoms.txt");
    sim.initTypes();
    Camera camera{};
    Visualization v{camera, 1000, 1000};

    sim.updatePositions();
    v.updateParticles(p);
    v.render(p.n_particles);
    while (!v.shouldClose()) {
        v.processInput(TIME_DELTA);

        sim.buildGrid();

        sim.applyForces();

        sim.updatePositions();

        v.updateParticles(p);
        v.render(p.n_particles);
    }
}