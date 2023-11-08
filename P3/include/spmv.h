#pragma once
#include "mtx.h"

void spmv(graph g, double *x, double *y);

void partition_graph(graph g, int k, int *p, double *x);