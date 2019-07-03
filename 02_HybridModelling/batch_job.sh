#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob_hybrid.out
#SBATCH -e ./tjob_hybrid.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J test_hybridpic
# Queue (Partition):
#SBATCH --partition=medium
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=2
# for OpenMP:
#SBATCH --cpus-per-task=32
#
#SBATCH --mail-type=all
#SBATCH --mail-user=floho@rzg.mpg.de
# Wall clock limit:
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=32
# For pinning threads correctly:
# export OMP_PLACES=threads

# Run the program
module load anaconda/3/5.1
module load gcc/9

python3 06_1d3vHybridCode_DipoleField_GEM_Rel.py
