#!/bin/bash
# Bash file to run run_nix.py in Ceci (nic5) cluster.
# Execute it with: bash run_nix.sh
module --force purge
module load releases/2023a #  Nic5: releases/2023b
#module load releases/2023b #  Nic5: releases/2023b
#module load SciPy-bundle/2023.11-gfbf-2023b
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyYAML/6.0-GCCcore-12.3.0 # Nic5: PyYAML/6.0.1-GCCcore-13.2.0

path=/home/ulb/glaciol/dmoreno/nix/

cd $path

# Call python to execute run_nix.py. python or python3?
python3 run_nix.py
