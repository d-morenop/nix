#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:23:54 2022

@author: dmoreno

# PYTHON SCRIPT TO COMPILE AND RUN THE FLOWLINE MODEL.
# This script copies flowline.cpp and its dependencies to a desired folder
# where it is compiled and run. The simulation output is then saved in a .nc file
# where the corresponding directory must be specified in write_nc.cpp and it may differ 
# from the compiling/running directory.
# 
# Daniel Moreno Parada (danielm@ucm.es).

"""

import os
import subprocess
import sys
import shutil
from subprocess import Popen, PIPE, STDOUT



# User defined directories.
path_flowline = "/home/dmoreno/scr/flowline/"
path_output   = "/home/dmoreno/flowline/mismip_therm/T_oce_f_q/steps/gamma_T_quad_long/gamma_sensitivity/n.250_T_air.188_T_oce_max.283_gamma_T_90.0/"
path_input    = "/home/dmoreno/c++/flowline/output/glacier_ews/"


# Create new directory if not existing. Make clean otherwise.
isdir = os.path.isdir(path_output)

if isdir == False:
    print('')
    print('-> Creating new directory.')
    print('')
    os.makedirs(path_output)
else:    
    print('')
    print('-> Existing directory. Make clean.')
    print('')

    # Boolean to check if path exists.
    isfile = os.path.isdir(path_output+'*.o')   
    
    # Make clean if path does exist.
    if isfile == True:
        #subprocess.run("rm "+path_output+"*.o", shell=True, check=True, \
        #               stdout=subprocess.PIPE, universal_newlines=True)
    
        subprocess.run("rm "+path_output+"*", shell=True, check=True, \
                       stdout=subprocess.PIPE, universal_newlines=True)

# Copy main script and write_nc to the output folder for compilation therein.
shutil.copyfile(path_flowline+'flow_line.cpp', path_output+'flow_line.cpp')
shutil.copyfile(path_flowline+'read-write_nc.cpp', path_output+'read-write_nc.cpp')
#shutil.copyfile(path_input+'noise.nc', path_output+'noise.nc')


# Compilation configuration. ['local', 'iceberg', 'brigit']
config = 'iceshelf'

if config == 'local':
    
    # Compiling command.
    cmd  = "g++ -I /path/to/eigen3/ -o "+path_output+"flow_line.o "+path_output+"flow_line.cpp -lnetcdf"

elif config == 'foehn':
    
    # Compiling command. -std=c++17
    cmd = "g++ -std=c++11 -I /usr/include/eigen3/ -o "+path_output+"flow_line.o "+path_output+"flow_line.cpp -lnetcdf"

elif config == 'iceshelf':
    
    # Compiling command. -std=c++17
    cmd = "g++ -std=c++11 -I /usr/include/eigen3/ -o "+path_output+"flow_line.o "+path_output+"flow_line.cpp -lnetcdf"

elif config == 'brigit':
    
    # Compiling command.
    cmd = "g++ -std=c++11 -I/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/include/ -I/usr/include/eigen3/ -L/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/lib/ -lnetcdf -o "+path_output+"flow_line.o "+path_output+"flow_line.cpp"
    
    # In Brigit, we need a submit.sh file to send job to the queue.
    shutil.copyfile(path_flowline+'submit.sh', path_output+'submit.sh')

# Create text file for terminal output. 
f = open(path_output+"out.txt", "w")

# Print compiling config.
print('')
print('-> Compiling configuration: ', config)
print('')

# Compile flowline with subprocess.
subprocess.run(cmd, shell=True, check=True, \
               stdout=f, universal_newlines=True)

print('')
print('-> Flowline compiled.')
print('')

print('')
print('-> Running flowline.')
print('')
    
# Run flowline in background. Note that the solution is stored in nc file.
# In Brigit, we need submit.sh to send it to the queue.
if config == 'brigit':

    # Old version
    #cmd_run = "sbatch "+path_output+"submit.sh"

    # Try changing working directory and then running sbatch there.
    os.chdir(path_output)
    cmd_run = "sbatch submit.sh"
    #print('cmd_run = ', cmd_run)
else:
    cmd_run = path_output+"flow_line.o &"


# Run flowline model.
p = subprocess.Popen(cmd_run, shell=True, \
                     stdout=f, universal_newlines=True)


