#include "simulation/particles.h"
#include "simulation/sim.h"
#include "visualization/vis.h"

int main() {
    Particles p{1};

    p.x[0] = 0;
    p.y[0] = 0;
    p.z[0] = 0;

    Simulation sim{p};
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