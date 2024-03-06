#!/bin/bash
#SBATCH --ntasks=16

module purge

module load GCC/11.3.0
module load OpenMPI/4.1.4 
srun -n 16 ./mpi-helloworld

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

