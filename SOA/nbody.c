//
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//
typedef float f32;
typedef double f64;
typedef unsigned long long u64;

#define ALIGN 64

//
typedef struct particle_s {

  f32 x, y, z;
  f32 vx, vy, vz;

} particle_t;

typedef struct particle_a {

  f32 *x, *y, *z;
  f32 *vx, *vy, *vz;

} particle_array_t;

//
void init(particle_array_t *p, u64 n) {

  if (!p) {
    exit(1);
  }

  p->x = (f32 *) aligned_alloc(ALIGN, n * sizeof(f32));
  p->y = (f32 *) aligned_alloc(ALIGN, n * sizeof(f32));
  p->z = (f32 *) aligned_alloc(ALIGN, n * sizeof(f32));

  p->vx = (f32 *) aligned_alloc(ALIGN, n * sizeof(f32));
  p->vy = (f32 *) aligned_alloc(ALIGN, n * sizeof(f32));
  p->vz = (f32 *) aligned_alloc(ALIGN, n * sizeof(f32));

  for (u64 i = 0; i < n; i++) {
    //
    u64 r1 = (u64) rand();
    u64 r2 = (u64) rand();
    f32 sign = (r1 > r2) ? 1 : -1;

    //
    p->x[i] = sign * (f32) rand() / (f32) RAND_MAX;
    p->y[i] = (f32) rand() / (f32) RAND_MAX;
    p->z[i] = sign * (f32) rand() / (f32) RAND_MAX;

    //
    p->vx[i] = (f32) rand() / (f32) RAND_MAX;
    p->vy[i] = sign * (f32) rand() / (f32) RAND_MAX;
    p->vz[i] = (f32) rand() / (f32) RAND_MAX;
  }
}

//
void move_particles(particle_array_t p, const f32 dt, u64 n) {
  //
  const f32 softening = 1e-20;

  //
  for (u64 i = 0; i < n; i++) {
    //
    f32 fx = 0.0;
    f32 fy = 0.0;
    f32 fz = 0.0;

    //23 floating-point operations
    for (u64 j = 0; j < n; j++) {
      //Newton's law
      const f32 dx = p.x[j] - p.x[i]; //1
      const f32 dy = p.y[j] - p.y[i]; //2
      const f32 dz = p.z[j] - p.z[i]; //3
      const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening; //9
      const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0); //11

      //Net force
      fx += dx / d_3_over_2; //13
      fy += dy / d_3_over_2; //15
      fz += dz / d_3_over_2; //17
    }

    //
    p.vx[i] += dt * fx; //19
    p.vy[i] += dt * fy; //21
    p.vz[i] += dt * fz; //23
  }

  //3 floating-point operations
  for (u64 i = 0; i < n; i++) {
    p.x[i] += dt * p.vx[i];
    p.y[i] += dt * p.vy[i];
    p.z[i] += dt * p.vz[i];
  }
}


//
int main(int argc, char **argv) {
  //
  const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
  const u64 steps = 10;
  const f32 dt = 0.01;

  //
  f64 time = 0;

  //Steps to skip for warm up
  const u64 warmup = 3;

  //
  particle_array_t p = {0};
  init(&p, n);

  const u64 s = sizeof(particle_t) * n;

  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);

  //
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s");
  fflush(stdout);

  //
  for (u64 i = 0; i < steps; i++) {
    //Measure
    const f64 start = omp_get_wtime();

    move_particles(p, dt, n);

    const f64 end = omp_get_wtime();

    printf("%5llu %.8e %s\n",
           i,
           (end - start),
           (i < warmup) ? "*" : "");
    fflush(stdout);

    if (i >= warmup) {
      time += (end - start);
    }
  }

  time /= (steps - warmup);

  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%.8lf s\033[0m\n",
         "Average time:", "", time);
  printf("-----------------------------------------------------\n");

  //
  free(p.x);
  free(p.y);
  free(p.z);
  free(p.vx);
  free(p.vy);
  free(p.vz);

  //
  return 0;
}
