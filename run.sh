#!/bin/bash
#SBATCH -p rome16q # partition (queue)
#SBATCH -N 8 # number of nodes
#SBATCH -o slurm.out
#SBATCH -e slurm.err
#SBATCH --exclusive

ulimit -s 200240

for ((i=1; i<=16; i*=2))
do
	srun --exclusive -n $i --ntasts-per-node $i ./a.out 15 20
done

for ((i=1; i<=8; i*=2))
do
	srun --exclusive -n $i --ntasts-per-node 1 ./a.out 15 20
done

for ((i=1; i<=8; i*=2))
do
	srun --exclusive -n $((i*16)) --ntasts-per-node 16 ./a.out 15 20
done
