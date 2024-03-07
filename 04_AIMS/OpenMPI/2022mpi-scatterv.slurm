!/bin/bash
SBATCH --job-name="Scatterv"
SBATCH --ntasks=3
SBATCH --time=0-00:10:00

module purge
module load gcc/8.3.0
module load openmpi/3.1.4
srun -n 3 ./mpi-scatterv

DO NOT ADD/EDIT BEYOND THIS LINE##
Job monitor command to list the resource usage
my-job-stats -a -n -s

