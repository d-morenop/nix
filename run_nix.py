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
import numpy as np
import itertools
from itertools import product


# Total number of items in a dictionary.
def permut_dict(d):
    dim = np.empty(len(d))
    c = 0
    for x in d:
        dim[c] = len(d[x])
        c += 1
    
    # np.pro() multiplies all elements in an array.
    n = int(np.prod(dim))
    return n

# Create lists with all permutation from input arrays.
def all_permutations(*arrays):
    # Use itertools.product to generate all permutations
    permutations = list(product(*arrays))

    # Preserve the original data type of the input arrays
    original_types = [type(array[0]) for array in arrays]

    # Convert permutations to the original data types
    result = [tuple(original_types[i](elem) for i, elem in enumerate(perm)) for perm in permutations]

    return result

# Function to find a key recursively in a dictionary and update its value.
def update_value_recursive(data, key, new_value):

    # Iterate through key-value pairs in the 'data' dictionary.
    for k, v in data.items():

        # Check if the current key (k) is equal to the target key.
        if k == key:
            # If the keys match, update the value associated with the key to the new value.
            data[k] = new_value
            #print('new_value = ', data[k])
            return True
        
        # If the value associated with the current key is itself a dictionary, recursively call the function.
        elif isinstance(v, dict):

            # If the recursive call returns True, propagate the True value up the call stack.
            if update_value_recursive(v, key, new_value):
                return True
    
    # If no match is found in the current dictionary or its nested dictionaries, return False.
    return False

# Modify the dictionary and then save as the yaml file.
def modify_yaml(file_path, path_modified, yaml_file_name, var_names, data_types, values):

    # Full path with name.
    full_path = file_path + yaml_file_name

    # Open and read yaml file.
    with open(full_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update output path.
    update_value_recursive(data, 'out', path_modified)

    # Update all desired parameters in yaml file.
    # Retrieve each entry in the dictionary and update its value.
    for i in range(len(var_names)):
        
        data_type = data_types[i]
        converted_value = data_type(values[i])
    
        print('converted_value = ', converted_value)

        # We have included float() to make sure it is given as a number. float(values[i])
        if update_value_recursive(data, var_names[i], converted_value):
            print(f" '{var_names[i]}' = '{converted_value}'.")
        else:
            print(f"Could not find '{var_names[i]}' in the dictionary.")

    # Save the updated dictionary.
    with open(full_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)





# Specify the path to your YAML file
yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_resolution.yaml"
yaml_file_name = "nix_params_resolution.yaml"


# Modify yaml file to run large ensembles of simulations.

#######################################################################
#######################################################################
# Define variable names and their corresponding values.

# Nix oscillations.
"""var_names = ['S_0', 'C_thw']

values_0 = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
values_1 = np.array([0.01, 0.02, 0.03, 0.04])"""

# Resolution study.
var_names = ['n', 'dt_min']

values_0 = np.array([2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12])
#values_0 = np.array([2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12])

values_1 = np.array([0.05])

# Data type of each array.
data_types = [int, float]

"""values_0 = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
values_1 = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
"""

#######################################################################
#######################################################################

# Preserve the trailing zero in cases where there are trailing zeros after the decimal point. 
# two decimals: {:.2f}.
# FIX THIS!!
values_0_str = len(values_0) * [None]
values_1_str = len(values_1) * [None]
values_0_str = ['{:.2f}'.format(value, len(str(value).split('.')[0])) for value in values_0]
values_1_str = ['{:.2f}'.format(value, len(str(value).split('.')[0])) for value in values_1]

# Create a string with all input values.
str_all = [values_0_str, values_1_str]

l_names = len(var_names)
values_all = [values_0, values_1]

# Include variable names in each value for folder naming.
for i in range(l_names):
    #values_all[i] = [var_names[i]+'.'+str(s) for s in values_all[i]]
    values_all[i] = [var_names[i]+'.'+s for s in str_all[i]]


# Create a dictionary to store variable names and their corresponding values.
variables = {var_names[0]: values_0, var_names[1]: values_1}



# Read the YAML file
with open(yaml_file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)


# User defined directories in nix_params.yaml.
path_nix    = yaml_data['path']['nix']
path_output = yaml_data['path']['out']


# Create empty list with lemgth equal to the number of total permutations.
N    = permut_dict(variables)    
name = N * [None]


# Folder names with all permutations.
# itertools.product() give the Cartesian product of input lists.
c = 0
for r in itertools.product(values_all[0], values_all[1]): 
    name[c] = r[0]+'_'+r[1]
    c += 1



# Future update: compact all values arrays?
perm = all_permutations(values_0, values_1)

# Loop over all permutations.
for i in range(N):

    print('perm_now = ', perm[i])
    
    # Update output path with names and values.
    path_modified = path_output+name[i]+'/'
    
    print('path_modified = ', path_modified)

    # Create new directory if not existing. Make clean otherwise.
    isdir = os.path.isdir(path_modified)

    if isdir == False:
        print('')
        print('-> Creating new directory.')
        print('')
        os.makedirs(path_modified)
    else:    
        print('')
        print('-> Existing directory. Overwriting.')
        print('')

        # Remove the existing directory and its content.
        shutil.rmtree(path_modified, ignore_errors=True)

        # Create directory.
        os.makedirs(path_modified)

    # Subdirectories.
    path_nix_scr = path_nix+"scr/"
    path_nix_par = path_nix+"par/"

    # Copy main script and write_nc to the output folder for compilation therein.
    #shutil.copyfile(path_input+'noise.nc', path_output+'noise.nc')
    path_output_scr = path_modified+"scr/"
    path_output_par = path_modified+"par/"

    shutil.copytree(path_nix_scr, path_output_scr)
    shutil.copytree(path_nix_par, path_output_par)

    # The copied yaml version in the modifed directory is then modified
    # for compilation therein.
    modify_yaml(path_output_par, path_modified, yaml_file_name, var_names, data_types, perm[i])


    # Compilation configuration. ['local', 'iceberg', 'brigit']
    config = 'iceshelf'

    if config == 'local':
        
        # Compiling command.
        cmd  = "g++ -I /path/to/eigen3/ -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lnetcdf"

    elif config == 'foehn':
        
        # Compiling command. -std=c++17
        cmd = "g++ -std=c++17 -I /usr/include/eigen3/ -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lnetcdf"

    elif config == 'iceshelf':
        
        # Compiling command. -std=c++17
        #cmd = "g++ -std=c++11 -I /usr/include/eigen3/ -o "+path_output+"nix.o "+path_output_scr+"nix.cpp -lnetcdf"

        # Compiling command. -std=c++17
        cmd = "g++ -std=c++17 -I /usr/include/eigen3/ -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lnetcdf -lyaml-cpp"

    elif config == 'brigit':
        
        # Compiling command.
        cmd = "g++ -std=c++17 -I/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/include/ -I/usr/include/eigen3/ -L/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/lib/ -lnetcdf -o "+path_output+"nix.o "+path_output_scr+"nix.cpp"
        
        # In Brigit, we need a submit.sh file to send job to the queue.
        shutil.copyfile(path_nix+'submit.sh', path_modified+'src/submit.sh')

    # Create text file for terminal output. "wb" for unbuffered output.
    f = open(path_modified+"out.txt", "wb")

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
        os.chdir(path_modified)
        cmd_run = "sbatch submit.sh"
        #print('cmd_run = ', cmd_run)
    else:
        cmd_run = path_modified+"nix.o &"


    # Run Nix model.
    p = subprocess.Popen(cmd_run, shell=True, \
                            stdout=f, universal_newlines=True)


