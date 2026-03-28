#include "simulation/particles.h"
#include "simulation/sim.h"
#include "visualization/vis.h"

int main() {
    Particles p{5};

    p.initParticlesRandomly(2, -2, 2, -2, 2, -2);

    Simulation sim{p};
    sim.setBoundaries(2, -2, 2, -2, 2, -2);
    Camera camera{};
    Visualization v{camera, 800, 600};

    sim.updatePositions();
    v.updateParticles(p);
    while (!v.shouldClose()) {
        v.processInput(TIME_DELTA);
        sim.applyForces();
        sim.updatePositions();

        v.updateParticles(p);
        v.render(p.n_particles);
    }
}