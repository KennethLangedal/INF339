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

    for (size_t i = 0; i < m.N; i++)
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

void reorder_separators(mesh m, int size, int rows, int *sep, int *old_id, int *new_id)
{
    double *tA = malloc(sizeof(double) * nonzero * rows);
    int *tI = malloc(sizeof(int) * nonzero * rows);

    for (size_t rank = 0; rank < size; rank++)
    {
        sep[rank] = 0;
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < nonzero; j++)
            {
                int u = m.I[(rank * rows + i) * nonzero + j];
                if (u < rank * rows || u >= (rank + 1) * rows)
                {
                    sep[rank]++;
                    break;
                }
            }
        }

        int ti = rank * rows, tj = rank * rows + sep[rank];
        for (size_t i = 0; i < rows; i++)
        {
            int any = 0;
            int u = i + rank * rows;
            for (size_t j = 0; j < nonzero; j++)
            {
                int v = m.I[(rank * rows + i) * nonzero + j];
                if (v < rank * rows || v >= (rank + 1) * rows)
                {
                    old_id[ti] = u;
                    new_id[u] = ti;
                    ti++;
                    any = 1;
                    break;
                }
            }
            if (!any)
            {
                old_id[tj] = u;
                new_id[u] = tj;
                tj++;
            }

            for (size_t k = 0; k < nonzero; k++)
                tA[(new_id[u] - rank * rows) * nonzero + k] = m.A[u * nonzero + k];

            for (size_t k = 0; k < nonzero; k++)
                tI[(new_id[u] - rank * rows) * nonzero + k] = m.I[u * nonzero + k];
        }

        for (size_t i = 0; i < nonzero * rows; i++)
            m.A[rank * rows * nonzero + i] = tA[i];

        for (size_t i = 0; i < nonzero * rows; i++)
            m.I[rank * rows * nonzero + i] = tI[i];
    }

    for (size_t i = 0; i < m.N * nonzero; i++)
        m.I[i] = new_id[m.I[i]];

    free(tA);
    free(tI);
}

void free_mesh(mesh *m)
{
    m->N = 0;
    free(m->A);
    free(m->I);
}

void step_ref(mesh m, double *Vold, double *Vnew)
{
    for (size_t i = 0; i < m.N; i++)
    {
        Vnew[i] = 0.0;
        for (size_t j = 0; j < nonzero; j++)
            Vnew[i] += m.A[i * nonzero + j] * Vold[m.I[i * nonzero + j]];
    }
}

void step_par(mesh m, double *Vold, double *Vnew)
{
#pragma omp parallel for
    for (size_t i = 0.0; i < m.N; i++)
    {
        Vnew[i] = 0.0;
        for (size_t j = 0; j < nonzero; j++)
            Vnew[i] += m.A[i * nonzero + j] * Vold[m.I[i * nonzero + j]];

        // Vnew[i] = m.A[i * nonzero + 0] * Vold[m.I[i * nonzero + 0]] +
        //           m.A[i * nonzero + 1] * Vold[m.I[i * nonzero + 1]] +
        //           m.A[i * nonzero + 2] * Vold[m.I[i * nonzero + 2]] +
        //           m.A[i * nonzero + 3] * Vold[m.I[i * nonzero + 3]];
    }
}
