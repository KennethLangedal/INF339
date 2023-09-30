#!/bin/bash

for ((i=1; i<=16; i++))
	sbatch -N 1 --ntasks-per-node $i jobscript