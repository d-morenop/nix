#!/bin/bash
# Submission script for Nix.
#SBATCH --time=1-23:59:00
#SBATCH --job-name=Nix
#SBATCH -o nix.out
#SBATCH -e nix.err
#SBATCH --nodes=1            # 1
#SBATCH --cpus-per-task=1    # Allocate 12 CPUs for a single task. Eigen parallelizes within a single process. 
#SBATCH --ntasks=1            # Must be 1 for Eigen.
#SBATCH --mem-per-cpu=10000    # 2500, 2.5Gb. Strong scaling: 32000. Weak: 10000. Divide the memory per CPU so that the total amount remains constant.
#SBATCH --partition=batch

# For n = 1e4 and n_z = 1e4, the neccesary memory is 28903368K (29GB, so set 30GB). $ sstat [jobid] --format=JobID,MaxRSS,AveRSS

# During compilation, libraries are dinamically linked, so we need to load the modules at runtime.
module --force purge
module load releases/2023b
module load Eigen/3.4.0-GCCcore-13.2.0
module load yaml-cpp
module load netCDF/4.9.2-gompi-2023b


#path=/scratch/ulb/glaciol/dmoreno/nix/
#cd $path

# The libnetcdf.so.19 library exists in a non-standard directory.
# $ find / -name libnetcdf.so.19 2>/dev/null
# Temporarily add the library path to your environment
# Not necessary if it is included in ~/.bash
#export LD_LIBRARY_PATH=/home/users/a/b/abarth/.julia/artifacts/87831472e1d79c45830c3d71850680eb745345fb/lib:$LD_LIBRARY_PATH

# To propagate this path to your srun command.
#srun --export=ALL,LD_LIBRARY_PATH ./nix.o 2>&1 > /dev/null

# If already included in bash.
#srun ./nix.o 2>&1 > /dev/null
srun ./nix.o > out.txt 2>&1
