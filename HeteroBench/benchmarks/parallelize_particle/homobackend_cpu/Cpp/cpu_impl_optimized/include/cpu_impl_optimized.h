#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

//
//  saving parameters
//
// const int NSTEPS = 20;
// const int NPARTICLES = 1000000;
// const int SAVEFREQ = 10;

//
// grid routines
//
void compute_forces_static_optimized(particle_t *particles, int n,
                                     linkedlist_static grid_static[gridsize2]);
void move_particles_static_optimized(particle_t *particles, int n,
                                     linkedlist_static grid[gridsize2]);