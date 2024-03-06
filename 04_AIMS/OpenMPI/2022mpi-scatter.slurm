#!/bin/bash
#SBATCH --job-name="Scatter"
#SBATCH --ntasks=4
#SBATCH --time=0-00:10:00

module purge
module load GCC/11.3.0
module load OpenMPI/4.1.4
srun -n 4 ./mpi-scatter

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

