#include "p2.h"
#include <stdlib.h>
#include <stdalign.h>

mesh init_mesh_4(int scale, double alpha, double beta)
{
    int N = 1 << scale;

    mesh m;
    m.N = N * N;

    m.A = aligned_alloc(32, sizeof(double) * m.N * 4);
    m.I = aligned_alloc(16, sizeof(int) * m.N * 4);

    for (int i = 0; i < m.N; i++)
    {
        m.A[i * 4] = beta;
        m.A[i * 4 + 1] = beta;
        m.A[i * 4 + 2] = beta;
        m.A[i * 4 + 3] = alpha;

        m.I[i * 4] = i - 1;
        m.I[i * 4 + 1] = i + 1;
        m.I[i * 4 + 2] = (i & 1) ? i - N - 1 : i + N + 1;
        m.I[i * 4 + 3] = i;

        if ((i % N) == 0) // First element in row
        {
            m.I[i * 4] = i;
            m.A[i * 4] = 0.0;
        }

        if ((!(i & 1) && i >= N * N - N) || // Last row
            ((i & 1) && i < N))             // First row
        {
            m.I[i * 4 + 2] = i;
            m.A[i * 4 + 2] = 0.0;
        }

        if ((i % N) == N - 1) // Last element in row
        {
            m.I[i * 4 + 1] = i;
            m.A[i * 4 + 1] = 0.0;
        }
    }

    return m;
}

void free_mesh(mesh *m)
{
    m->N = 0;
    free(m->A);
    free(m->I);
}

void step_ref(mesh m, double *Vold, double *Vnew)
{
    for (int i = 0; i < m.N; i++)
    {
        Vnew[i] = 0.0;
        for (int j = 0; j < nonzero; j++)
            Vnew[i] += m.A[i * nonzero + j] * Vold[m.I[i * nonzero + j]];
    }
}

void step_par(mesh m, double *Vold, double *Vnew)
{
#pragma omp parallel for
    for (int i = 0.0; i < m.N; i++)
    {
        Vnew[i] = 0.0;
        for (int j = 0; j < nonzero; j++)
            Vnew[i] += m.A[i * nonzero + j] * Vold[m.I[i * nonzero + j]];

        // Vnew[i] = m.A[i * nonzero + 0] * Vold[m.I[i * nonzero + 0]] +
        //           m.A[i * nonzero + 1] * Vold[m.I[i * nonzero + 1]] +
        //           m.A[i * nonzero + 2] * Vold[m.I[i * nonzero + 2]] +
        //           m.A[i * nonzero + 3] * Vold[m.I[i * nonzero + 3]];
    }
}
