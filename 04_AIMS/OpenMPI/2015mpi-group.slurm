#!/bin/bash
#SBATCH --job-name="MPI Group"
#SBATCH --ntasks=8
#SBATCH --time=0-00:10:00
module purge
/usr/local/module/spartan_old.sh
module load OpenMPI/1.10.2-GCC-4.9.2

srun ./mpi-group

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

