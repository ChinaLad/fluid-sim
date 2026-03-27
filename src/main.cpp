#include "simulation/particles.h"
#include "simulation/sim.h"

int main() {
    Particles p{1};

    p.x[0] = 0;
    p.y[0] = 0;
    p.z[0] = 0;

    Simulation sim{p};

    sim.updatePositions();
    sim.applyForces();
    sim.updatePositions();
}