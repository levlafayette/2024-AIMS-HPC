#!/bin/bash
#SBATCH --job-name="MPI Group"
#SBATCH --ntasks=8
#SBATCH --time=0-00:10:00
module purge


module load gcc/8.3.0
module load openmpi/3.1.4
srun -n 8 ./mpi-group

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

