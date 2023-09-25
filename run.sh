#!/bin/bash
#SBATCH -p rome16q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1  # number of cores
#SBATCH --mem 1G # memory pool for all cores
#SBATCH -t 0-4:00 # time (D-HH:MM)
#SBATCH -o slurm.out # STDOUT
#SBATCH -e slurm.err # STDERR
 
ulimit -s 10240
 
module load slurm 
module load openmpi/gcc/64/4.0.1 
 
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_btl_openib_if_include=mlx5_0:1
export OMPI_MCA_btl=self,openib
export OMPI_MCA_btl_tcp_if_exclude=lo,dis0,enp113s0f0
 
# Alternative method
#mpirun -np  $SLURM_NTASKS numactl --cpunodebind=0 --localalloc  /home/torel/workspace/Benchmarks/MPI/OSU-Micro-Benchmarks/osu-micro-benchmarks-5.5/mpi/pt2pt/osu_mbw_mr 
 
# Prefered method using srun
#
srun --mpi=pmi2 -n $SLURM_NTASKS ./gemv_row 15
