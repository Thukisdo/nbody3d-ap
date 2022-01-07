//
#include <math.h>
#include <mkl.h>
#include <omp.h>
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

  p->x = (f32 *)aligned_alloc(ALIGN, n * sizeof(f32));
  p->y = (f32 *)aligned_alloc(ALIGN, n * sizeof(f32));
  p->z = (f32 *)aligned_alloc(ALIGN, n * sizeof(f32));

  p->vx = (f32 *)aligned_alloc(ALIGN, n * sizeof(f32));
  p->vy = (f32 *)aligned_alloc(ALIGN, n * sizeof(f32));
  p->vz = (f32 *)aligned_alloc(ALIGN, n * sizeof(f32));

  for (u64 i = 0; i < n; i++) {
    //
    u64 r1 = (u64)rand();
    u64 r2 = (u64)rand();
    f32 sign = (r1 > r2) ? 1 : -1;

    //
    p->x[i] = sign * (f32)rand() / (f32)RAND_MAX;
    p->y[i] = (f32)rand() / (f32)RAND_MAX;
    p->z[i] = sign * (f32)rand() / (f32)RAND_MAX;

    //
    p->vx[i] = (f32)rand() / (f32)RAND_MAX;
    p->vy[i] = sign * (f32)rand() / (f32)RAND_MAX;
    p->vz[i] = (f32)rand() / (f32)RAND_MAX;
  }
}

//
void move_particles(particle_array_t p, const f32 dt, const u64 n) {

  //
  const u64 BLOCK_SIZE = 4096;
  for (u64 u = 0; u < n; u += BLOCK_SIZE) {
    //
    for (u64 i = 0; i < n; i++) {
      //
      f32 fx = 0;
      f32 fy = 0;
      f32 fz = 0;

      const f32 posx = p.x[i];
      const f32 posy = p.y[i];
      const f32 posz = p.z[i];

      // 23 floating-point operations
      for (u64 j = u; j < u + BLOCK_SIZE; j++) {
        // Newton's law
        const f32 dx = p.x[j] - posx;                      // 1
        const f32 dy = p.y[j] - posy;                      // 2
        const f32 dz = p.z[j] - posz;                      // 3
        const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz); // 8

        const f32 f = 1.0f / (d_2 * sqrtf(d_2)); // 11

        // Net force
        fx += dx * f; // 13
        fy += dy * f; // 15
        fz += dz * f; // 17
      }

      //
      p.vx[i] += dt * fx; // 19
      p.vy[i] += dt * fy; // 21
      p.vz[i] += dt * fz; // 23
    }
  }

  // 3 floating-point operations
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
  f64 rate = 0.0, drate = 0.0;

  // Steps to skip for warm up
  const u64 warmup = 3;

  //
  particle_array_t p;

  //
  init(&p, n);

  const u64 s = sizeof(f32) * (n * 6);

  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s,
         s >> 10, s >> 20);

  //
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s",
         "GFLOP/s");
  fflush(stdout);

  //
  for (u64 i = 0; i < steps; i++) {
    // Measure
    const f64 start = omp_get_wtime();

    move_particles(p, dt, n);

    const f64 end = omp_get_wtime();

    // Number of interactions/iterations
    const f32 h1 = (f32)(n) * (f32)(n - 1);

    // GFLOPS
    const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;

    if (i >= warmup) {
      rate += h2 / (end - start);
      drate += (h2 * h2) / ((end - start) * (end - start));
    }

    //
    printf("%5llu %10.3e %10.3e %8.1f %s\n", i, (end - start),
           h1 / (end - start), h2 / (end - start), (i < warmup) ? "*" : "");

    fflush(stdout);
  }

  //
  rate /= (f64)(steps - warmup);
  drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
         "Average performance:", "", rate, drate);
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
