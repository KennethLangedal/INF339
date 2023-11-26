#!/bin/bash

#SBATCH -p rome16q # partition (queue)
#SBATCH -N 8 # number of nodes
#SBATCH --ntasks-per-node 1  # number of cores
#SBATCH --cpus-per-task=16
#SBATCH -t 0-00:01 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --exclusive

ulimit -s 10240

module purge
module load slurm/21.08.8
module load libfabric/gcc/1.18.0
module load openmpi4/gcc/4.1.2
module load libevent/2.1.12-stable
module load cuda11.8/toolkit/11.8.0
module load metis

export OMPI_MCA_pml="^ucx"
export OMPI_MCA_btl_openib_if_include="mlx5_1:1"
export OMPI_MCA_btl_tcp_if_exclude=docker0,docker_gwbridge,eno1,eno2,lo,enp196s0f0np0,enp196s0f1np1,ib0,ib1,veth030713f,veth07ce296,veth50ead6f,veth73c0310,veth9e2a12b,veth9e2cc2e,vethecc4600,ibp65s0f1,enp129s0f0np0,enp129s0f1np1,ibp65s0f0
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_mpi_cuda_support=0

ldd /home/torel/bin/cpi-4.1.4.x86_64

export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY=0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#srun --mpi=pmi2 -n $SLURM_NTASKS ./main /global/D1/projects/mtx/datasets/suitesparse/JGD_GL7d/GL7d19/GL7d19.mtx

#mpirun -np $SLURM_NTASKS --bind-to none ./main /global/D1/projects/mtx/datasets/suitesparse/JGD_GL7d/GL7d19/GL7d19.mtx

cd P3

make

for file in $(find /global/D1/projects/UiB-INF339/matrices/ | grep .mtx)
do
	echo $(basename $file)
	mpirun -np $SLURM_NTASKS --bind-to none ./main $file 1 2 4 8
done