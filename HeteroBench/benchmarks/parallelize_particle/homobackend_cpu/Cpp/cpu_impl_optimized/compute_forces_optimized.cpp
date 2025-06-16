#include "cpu_impl.h"

void apply_force_static(particle_t& particle, particle_t& neighbor)
{
	apply_force_static_optimized(particle, neighbor);
}

void compute_forces_static_optimized(particle_t* particles, int n, linkedlist_static grid_static[gridsize2])
{
	compute_forces_static(particles, n, grid_static);
}