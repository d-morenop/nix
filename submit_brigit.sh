#!/bin/bash
#SBATCH -p short 
#SBATCH -J flowline
#SBATCH -o flowline.out
#SBATCH -e flowline.err
#SBATCH -n 1
#SBATCH --mem=400
#SBATCH -t 0-23:00:00 

# Run the job. -p long, -t 0-168:00:00, -o flowline.out, -e flowline.err

# Try this if changin working directory from run_flowline.py.
./flow_line_mismip_4.o

# This works with old version of run_flowline.py
#/home/dmoren07/c++/flowline/output/mismip/exp3/test.sbatch/flow_line_mismip_4.o
