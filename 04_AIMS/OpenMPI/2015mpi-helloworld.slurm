#!/bin/bash
#SBATCH --ntasks=16

module purge
/usr/local/module/spartan_old.sh
module load OpenMPI/1.10.2-GCC-4.9.2

srun mpi-helloworld

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

