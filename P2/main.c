#include "p2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define alpha 0.7
#define beta 0.1

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Give scale\n");
        return 1;
    }

    int scale = atoi(argv[1]);

    mesh m = init_mesh_4(scale, alpha, beta);
    double *Vold = malloc(sizeof(double) * m.N);
    double *Vnew = malloc(sizeof(double) * m.N);

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < m.N; j++)
            Vold[j] = 0.0;

        for (int j = 0; j < m.N; j++)
            Vnew[j] = 0.0;

        Vold[0] = 0xffffff;

        double t0 = omp_get_wtime();

        for (int t = 0; t < 100; t++)
        {
            step_par(m, Vold, Vnew);

            double *tmp = Vold;
            Vold = Vnew;
            Vnew = tmp;
        }

        double t1 = omp_get_wtime();

        double l2 = 0.0;
        for (int j = 0; j < m.N; j++)
            l2 += Vold[j] * Vold[j];

        l2 = sqrt(l2);

        double ops = (long long)m.N * 8ll * 100ll; // 4 multiplications and 4 additions
        double time = t1 - t0;

        printf("%lf GFLOPS, L2 = %lf\n", (ops / time) / 1e9, l2);
    }

    free(Vold);
    free(Vnew);
    free_mesh(&m);

    return 0;
}