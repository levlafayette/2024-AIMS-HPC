#!/bin/bash
#SBATCH --ntasks=4



module load python37

srun python3 multiproc1.py
