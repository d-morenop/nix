#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:23:54 2022

@author: dmoreno

# PYTHON SCRIPT TO COMPILE AND RUN NIX MODEL.
# This script copies nix.cpp and its dependencies to a desired folder
# where it is compiled and run. The simulation output is then saved in a .nc file
# where the corresponding directory must be specified in write_nc.cpp and it may differ 
# from the compiling/running directory.
# 
# Daniel Moreno Parada (danielm@ucm.es).

"""

import os
import subprocess
import shutil
import yaml



# Specify the path to your YAML file
yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params.yaml"

# Read the YAML file
with open(yaml_file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

# User defined directories in nix_params.yaml.
path_nix    = yaml_data['path']['nix']
path_output = yaml_data['path']['out']

# Subdirectories.
path_nix_scr = path_nix+"scr/"
path_nix_par = path_nix+"par/"

# Create new directory if not existing. Make clean otherwise.
isdir = os.path.isdir(path_output)

if isdir == False:
    print('')
    print('-> Creating new directory.')
    print('')
    os.makedirs(path_output)
else:    
    print('')
    print('-> Existing directory. Overwriting.')
    print('')

    # Remove the existing directory and its contents
    shutil.rmtree(path_output, ignore_errors=True)

    # Create directory.
    os.makedirs(path_output)


# Copy main script and write_nc to the output folder for compilation therein.
#shutil.copyfile(path_input+'noise.nc', path_output+'noise.nc')

path_output_scr = path_output+"scr/"
path_output_par = path_output+"par/"

shutil.copytree(path_nix_scr, path_output_scr)
shutil.copytree(path_nix_par, path_output_par)


# Compilation configuration. ['local', 'iceberg', 'brigit']
config = 'iceshelf'

if config == 'local':
    
    # Compiling command.
    cmd  = "g++ -I /path/to/eigen3/ -o "+path_output+"nix.o "+path_output_scr+"nix.cpp -lnetcdf"

elif config == 'foehn':
    
    # Compiling command. -std=c++17
    cmd = "g++ -std=c++17 -I /usr/include/eigen3/ -o "+path_output+"nix.o "+path_output_scr+"nix.cpp -lnetcdf"

elif config == 'iceshelf':
    
    # Compiling command. -std=c++17
    #cmd = "g++ -std=c++11 -I /usr/include/eigen3/ -o "+path_output+"nix.o "+path_output_scr+"nix.cpp -lnetcdf"

    # Compiling command. -std=c++17
    #cmd = "g++ -std=c++11 -I /usr/include/eigen3/ -o "+path_output+"nix.o "+path_output_scr+"nix.cpp -lnetcdf -lyaml-cpp"
    cmd = "g++ -std=c++17 -I /usr/include/eigen3/ -o "+path_output+"nix.o "+path_output_scr+"nix.cpp -lnetcdf -lyaml-cpp"

elif config == 'brigit':
    
    # Compiling command.
    cmd = "g++ -std=c++17 -I/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/include/ -I/usr/include/eigen3/ -L/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/lib/ -lnetcdf -o "+path_output+"nix.o "+path_output_scr+"nix.cpp"
    
    # In Brigit, we need a submit.sh file to send job to the queue.
    shutil.copyfile(path_nix+'submit.sh', path_output+'src/submit.sh')

# Create text file for terminal output. "wb" for unbuffered output.
f = open(path_output+"out.txt", "wb")

# Print compiling config.
print('')
print('-> Compiling configuration: ', config)
print('')

# Compile nix with subprocess.
subprocess.run(cmd, shell=True, check=True, \
               stdout=f, universal_newlines=True)

print('')
print('-> Nix compiled.')
print('')

print('')
print('-> Running Nix.')
print('')
    
# Run Nix in background. Note that the solution is stored in nc file.
# In Brigit, we need submit.sh to send it to the queue.
if config == 'brigit':

    # Old version
    #cmd_run = "sbatch "+path_output+"submit.sh"

    # Try changing working directory and then running sbatch there.
    os.chdir(path_output)
    cmd_run = "sbatch submit.sh"
    #print('cmd_run = ', cmd_run)
else:
    cmd_run = path_output+"nix.o &"


# Run Nix model.
p = subprocess.Popen(cmd_run, shell=True, \
                     stdout=f, universal_newlines=True)


