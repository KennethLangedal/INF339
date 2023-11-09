# INF339

Commands

```
srun -p rome16q -N 1 --pty bash --login
module load gcc/11.2.0
module load openmpi4/gcc/4.1.2
mpicc -O3 -march=native -ffast-math
ssh -AY2C kennethl@dnat.simula.no -p 60441
sudo cpupower frequency-set -u 2000mhz
```