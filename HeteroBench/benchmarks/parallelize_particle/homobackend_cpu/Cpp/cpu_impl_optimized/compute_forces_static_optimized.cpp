#include "cpu_impl.h"

void compute_forces_static_optimized(particle_t *particles, int n,
                                     linkedlist_static grid_static[gridsize2]) {
  compute_forces_static(particles, n, grid_static);
}