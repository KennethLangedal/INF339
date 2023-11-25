#include "mtx.h"
#include "spmv.h"

#include <math.h>
#include <stdlib.h>
#include <omp.h>

void run(graph g, int it, double *input)
{
    double *V = malloc(sizeof(double) * g.N);
    double *Y = malloc(sizeof(double) * g.N);
    int *p;
#pragma omp parallel shared(p) firstprivate(g, it, input, V, Y)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        if (tid == 0)
        {
            printf("Time GFLOPs GBs CS (%d threads)\n", nt);
            p = malloc(sizeof(int) * (nt + 1));
            partition_graph_naive(g, 0, g.N, nt, p);
        }

        for (int t = 0; t < it; t++)
        {
#pragma omp for
            for (int i = 0; i < g.N; i++)
                V[i] = input[i];

#pragma omp barrier
            double t0 = omp_get_wtime();
#pragma omp barrier

            for (int i = 0; i < 100; i++)
            {
                spmv_part(g, p[tid], p[tid + 1], V, Y);
                double *tmp = V;
                V = Y;
                Y = tmp;
#pragma omp barrier
            }

            double t1 = omp_get_wtime();

            if (tid == 0)
            {

                double a = 1.0;
                for (int i = 0; i < g.N; i++)
                    a = a < fabs(V[i]) ? fabs(V[i]) : a;

                double L2 = 0.0;
                for (int i = 0; i < g.N; i++)
                    L2 += (V[i] / a) * (V[i] / a);

                double ops = (double)g.M * 2.0 * 100.0;
                double GFLOPS = (ops / (t1 - t0)) / 1e9;

                //                            g.E,  g.A                   g.V,   V,    Y
                double data = ((double)g.M * (4.0 + 8.0) + (double)g.N * (4.0 + 8.0 + 8.0)) * 100.0;
                double GBs = (data / (t1 - t0)) / 1e9;

                printf("%.2lf %.2lf %.2lf %.10lf\n", t1 - t0, GFLOPS, GBs, L2);
            }
#pragma omp barrier
        }

        if (tid == 0)
            free(p);
    }

    free(V);
    free(Y);
}

int main(int argc, char **argv)
{
    double t0 = omp_get_wtime();
    graph g = parse_and_validate_mtx(argv[1]);
    double t1 = omp_get_wtime();
    printf("%.2lfs parsing\n", t1 - t0);

    double *input = malloc(sizeof(double) * g.N);
    for (int i = 0; i < g.N; i++)
        input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;

    // Main program start - Full version
    for (int i = 2; i < argc; i++)
    {
        int nt = atoi(argv[i]);
        omp_set_num_threads(nt);
        run(g, 3, input);
    }

    free(input);
    free_graph(&g);

    return 0;
}