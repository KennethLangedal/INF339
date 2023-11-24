#pragma once
#include <stdio.h>

typedef struct
{
    int N, M;
    int *V, *E;
    double *A;
} graph;

graph parse_and_validate_mtx(const char *path);

graph parse_mtx(FILE *f);

void free_graph(graph *g);

void sort_edges(graph g);

void normalize_graph(graph g);

int validate_graph(graph g);