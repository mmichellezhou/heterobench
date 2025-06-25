#include "cpu_impl.h"

void move_particles_static_optimized(particle_t *particles, int n,
                                     linkedlist_static grid[gridsize2]) {
  move_particles_static(particles, n, grid);
}