#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>
#include "gfx.h"
#include "p2.h"

#define W 40
#define alpha 0.7
#define beta 0.1

void draw_mesh(int scale, double *data)
{
    int N = 1 << scale;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int c = log2(data[i * N + j]) * 8;
            if (c > 255)
                c = 255;
            if (c < 0)
                c = 0;
            gfx_color(0, c, c);
            int y = i * W;
            int x = (j / 2) * W;

            if ((j & 1) == 0)
                gfx_fill_triangle(x, y, x + W, y + W, x, y + W);
            else
                gfx_fill_triangle(x, y, x + W, y, x + W, y + W);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Give scale\n");
        return 1;
    }

    int scale = atoi(argv[1]);
    int N = 1 << scale;

    int ysize = N * W;
    int xsize = (N / 2) * W;

    char c;

    gfx_open(xsize, ysize, "INF339");

    double *Vnew = malloc(sizeof(double) * N * N);
    double *Vold = malloc(sizeof(double) * N * N);

    for (int i = 0; i < N * N; i++)
    {
        Vnew[i] = 0.0;
        Vold[i] = 0.0;
    }

    mesh m = init_mesh_4(scale, alpha, beta);

    for (int i = 0; i < 1000000; i++)
    {
        if ((i % 1000) == 0)
        {
            Vnew[m.N / 2 + N / 2] = INT_MAX;
            // Vnew[m.N - 1] = INT_MAX;
        }

        draw_mesh(scale, Vnew);
        gfx_flush();
        // usleep(10);

        double *tmp = Vnew;
        Vnew = Vold;
        Vold = tmp;

        step_ref(m, Vold, Vnew);
    }

    free_mesh(&m);
    free(Vold);
    free(Vnew);

    return 0;
}
