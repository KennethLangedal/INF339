#pragma once

#define nonzero 4

typedef struct
{
    int N;
    double *A;
    int *I;
} mesh;

mesh init_mesh_4(int scale, double alpha, double beta);

void free_mesh(mesh *m);

void step_ref(mesh m, double *Vold, double *Vnew);

void step_par(mesh m, double *Vold, double *Vnew);