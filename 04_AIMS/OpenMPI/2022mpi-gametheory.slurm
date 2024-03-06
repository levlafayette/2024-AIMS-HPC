#!/bin/bash
#SBATCH --job-name="GameTheory"
#SBATCH --ntasks=2
#SBATCH --time=0-00:10:00
module purge


module load GCC/11.3.0
module load OpenMPI/4.1.4 
srun -n 2 ./mpi-gametheory

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

