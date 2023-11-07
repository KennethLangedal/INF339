#include "mtx.h"
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

#define GENERAL 0
#define SYMMETRIC 1

typedef struct
{
    int symmetry;
    int M, N, L;
    int *I, *J;
    double *A;
} mtx;

static inline void parse_int(char *data, size_t *p, int *v)
{
    while (data[*p] == ' ')
        (*p)++;

    *v = 0;

    int sign = 1;
    if (data[*p] == '-')
    {
        sign = -1;
        (*p)++;
    }

    while (data[*p] >= '0' && data[*p] <= '9')
    {
        *v = (*v) * 10 + data[*p] - '0';
        (*p)++;
    }

    *v *= sign;
}

static inline void parse_real(char *data, size_t *p, double *v)
{
    while (data[*p] == ' ')
        (*p)++;

    *v = 0.0;

    double sign = 1.0;
    if (data[*p] == '-')
    {
        sign = -1.0;
        (*p)++;
    }

    while (data[*p] >= '0' && data[*p] <= '9')
    {
        *v = (*v) * 10.0 + (double)(data[*p] - '0');
        (*p)++;
    }

    if (data[*p] == '.')
    {
        (*p)++;
        double s = 0.1;
        while (data[*p] >= '0' && data[*p] <= '9')
        {
            *v += (double)(data[*p] - '0') * s;
            (*p)++;
            s *= 0.1;
        }
    }
    *v *= sign;
}

static inline void skip_line(char *data, size_t *p)
{
    while (data[*p] != '\n')
        (*p)++;
    (*p)++;
}

mtx internal_parse_mtx_header(char *data, size_t *p)
{
    mtx m;

    char header[256];
    while (*p < 255 && data[*p] != '\n')
    {
        header[*p] = data[*p];
        (*p)++;
    }
    header[*p] = '\0';
    (*p)++;

    if (*p == 256)
    {
        fprintf(stderr, "Invalid header %s\n", header);
        exit(1);
    }

    char *token = strtok(header, " ");
    token = strtok(NULL, " ");
    token = strtok(NULL, " ");
    token = strtok(NULL, " ");
    token = strtok(NULL, " ");

    if (strcmp(token, "general") == 0)
        m.symmetry = GENERAL;
    else if (strcmp(token, "symmetric") == 0)
        m.symmetry = SYMMETRIC;
    else
    {
        fprintf(stderr, "Invalid symmetry %s\n", token);
        exit(1);
    }

    return m;
}

mtx internal_parse_mtx(FILE *f)
{
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = mmap(0, size, PROT_READ, MAP_SHARED, fileno_unlocked(f), 0);
    size_t p = 0;

    mtx m = internal_parse_mtx_header(data, &p);

    while (data[p] == '%')
        skip_line(data, &p);

    parse_int(data, &p, &m.M);
    parse_int(data, &p, &m.N);
    parse_int(data, &p, &m.L);

    m.I = malloc(sizeof(int) * m.L);
    m.J = malloc(sizeof(int) * m.L);
    m.A = malloc(sizeof(double) * m.L);

    for (int i = 0; i < m.L; i++)
    {
        skip_line(data, &p);

        while (data[p] == '%')
            skip_line(data, &p);

        parse_int(data, &p, m.I + i);
        parse_int(data, &p, m.J + i);
        parse_real(data, &p, m.A + i);
    }

    munmap(data, size);

    return m;
}

void internal_free_mtx(mtx *m)
{
    m->symmetry = GENERAL;
    m->M = 0;
    m->N = 0;
    m->L = 0;
    free(m->I);
    free(m->J);
    m->I = NULL;
    m->J = NULL;

    if (m->A != NULL)
        free(m->A);
    m->A = NULL;
}

graph parse_mtx(FILE *f)
{
    mtx m = internal_parse_mtx(f);

    graph g;
    g.N = m.N > m.M ? m.N : m.M;
    g.V = calloc(g.N + 1, sizeof(int));

    // Count degree
    for (int i = 0; i < m.L; i++)
    {
        g.V[m.I[i] - 1]++;
        if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
            g.V[m.J[i] - 1]++;
    }

    for (int i = 1; i <= g.N; i++)
    {
        g.V[i] += g.V[i - 1];
    }

    g.M = g.V[g.N];
    g.E = malloc(sizeof(int) * g.M);
    g.A = malloc(sizeof(double) * g.M);

    for (int i = 0; i < m.L; i++)
    {
        g.V[m.I[i] - 1]--;
        g.E[g.V[m.I[i] - 1]] = m.J[i] - 1;
        g.A[g.V[m.I[i] - 1]] = m.A[i];

        if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
        {
            g.V[m.J[i] - 1]--;
            g.E[g.V[m.J[i] - 1]] = m.I[i] - 1;
            g.A[g.V[m.J[i] - 1]] = m.A[i];
        }
    }

    internal_free_mtx(&m);

    return g;
}

void free_graph(graph *g)
{
    g->N = 0;
    g->M = 0;
    free(g->A);
    free(g->V);
    free(g->E);
    g->A = NULL;
    g->V = NULL;
    g->E = NULL;
}

int compare(const void *a, const void *b, void *c)
{
    int ia = *(int *)a, ib = *(int *)b;
    int *data = (int *)c;

    return data[ia] - data[ib];
}

void sort_edges(graph g)
{
    int *index = malloc(sizeof(int) * g.N);
    int *E_buffer = malloc(sizeof(int) * g.N);
    double *A_buffer = malloc(sizeof(double) * g.N);

    for (int u = 0; u < g.N; u++)
    {
        int degree = g.V[u + 1] - g.V[u];
        for (int i = 0; i < degree; i++)
            index[i] = i;

        qsort_r(index, degree, sizeof(int), compare, g.E + g.V[u]);

        for (int i = 0; i < degree; i++)
        {
            E_buffer[i] = g.E[g.V[u] + index[i]];
            A_buffer[i] = g.A[g.V[u] + index[i]];
        }

        memcpy(g.E + g.V[u], E_buffer, degree * sizeof(int));
        memcpy(g.A + g.V[u], A_buffer, degree * sizeof(double));
    }

    free(index);
    free(E_buffer);
    free(A_buffer);
}

int validate_graph(graph g)
{
    for (int u = 0; u < g.N; u++)
    {
        int degree = g.V[u + 1] - g.V[u];
        if (degree < 0 || degree > g.M)
            return 0;

        for (int i = g.V[u]; i < g.V[u + 1]; i++)
        {
            if (g.E[i] < 0 || g.E[i] > g.M)
                return 0;
            if (i > g.V[u] && g.E[i] <= g.E[i - 1])
                return 0;
        }
    }
    return 1;
}