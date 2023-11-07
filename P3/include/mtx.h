#pragma once
#include <stdio.h>

typedef struct
{
    int N, M;
    int *V, *E;
    double *A;
} graph;

graph parse_mtx(FILE *f);

void free_graph(graph *g);

void sort_edges(graph g);

int validate_graph(graph g);