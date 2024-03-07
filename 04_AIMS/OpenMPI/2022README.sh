!/bin/bash

module purge
module load foss/2022a 

mpicc mpi-helloworld.c -o mpi-helloworld
sleep 5
sbatch 2022mpi-helloworld.slurm

mpif90 mpi-helloworld.f90 -o mpi-helloworld
sleep 5
sbatch 2022mpi-helloworld.slurm

mpicc mpi-ping.c -o mpi-ping
sleep 5
sbatch 2022mpi-ping.slurm

mpicc mpi-sendrecv.c -o mpi-sendrecv
sleep 5
sbatch 2022mpi-sendrecv.slurm

mpif90 mpi-sendrecv.f90 -o mpi-sendrecv
sleep 5
sbatch 2022mpi-sendrecv.slurm

mpicc mpi-pingpong.c -o mpi-pingpong
sleep 5
sbatch 2022mpi-pingpong.slurm

mpicc mpi-gametheory.c -o mpi-gametheory
sleep 5
sbatch 2022mpi-gametheory.slurm

 You'll need compile with the math library for this one!

mpicc mpi-particle.c -lm -o mpi-particle
sleep 5
sbatch 2022mpi-gametheory.slurm

mpicc mpi-group.c -o mpi-group
sleep 5
sbatch 2022mpi-group.slurm

exit
