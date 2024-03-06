#!/bin/bash
#SBATCH --job-name="PingPong"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-00:10:00
#SBATCH --nodelist=spartan-bm044,spartan-bm045

module purge


module load GCC/11.3.0
module load OpenMPI/4.1.4
srun -n 2 ./mpi-pingpong

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s 

