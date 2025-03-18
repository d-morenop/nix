#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:23:54 2022

@author: dmoreno

# PYTHON SCRIPT TO COMPILE AND RUN NIX MODEL v1.0.
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
        
        # Ensure that the value is taken with the correcto data type.
        data_type       = data_types[i]
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




#######################################################################
#######################################################################
# Define variable names and their corresponding values.

# Select desired experiment.
exp = 'parallel'

# OSCILLATIONS STUDY.
if exp == 'oscillations':

    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_oscillations.yaml"
    yaml_file_name = "nix_params_oscillations.yaml"

    var_names = ['S_0', 'C_thw']

    values_0 = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    values_1 = np.array([0.01, 0.02, 0.03, 0.04])

    # Data type of each array.
    data_types = [float, float]

    values = [values_0, values_1]


# RESOLUTION STUDY.
elif exp == 'parallel':

    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_parallel.yaml"
    yaml_file_name = "nix_params_parallel.yaml"

    var_names = ['vel_meth', 'n', 'n_z', 'dt_min', 'eps']

    values_0 = np.array(['Blatter-Pattyn'])  # 'SSA', 'DIVA', 'Blatter-Pattyn'
    values_1 = np.array([2**11])                             # , 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15
    values_2 = np.array([30, 40, 50])                               # 20
    values_3 = np.array([0.1]) # 0.1, 0.05
    values_4 = np.array([1.0e-7]) # 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9
    

    # Data type of each array.
    data_types = [str, int, int, float, float]

    values = [values_0, values_1, values_2, values_3, values_4]


# RESOLUTION STUDY.
elif exp == 'nic5':

    yaml_file_path = "/scratch/ulb/glaciol/dmoreno/nix/par/nix_params_parallel_nic5.yaml"
    yaml_file_name = "nix_params_parallel_nic5.yaml"

    var_names = ['n', 'n_z', 'dt_min', 'eps']

    values_0 = np.array([15000]) # 5000, 300, 500, 300, 4000
    values_1 = np.array([10000])  # 5000, 35, 200, 3000
    values_2 = np.array([2.0]) # 0.1, 0.05
    values_3 = np.array([1.0e-7]) # 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9

    # Data type of each array.
    data_types = [int, int, float, float]

    values = [values_0, values_1, values_2, values_3]



# RESOLUTION STUDY.
elif exp == 'lemaitre4':

    yaml_file_path = "/home/ulb/glaciol/dmoreno/nix/par/nix_params_parallel_lemaitre4.yaml"
    yaml_file_name = "nix_params_parallel_lemaitre4.yaml"

    var_names = ['n', 'n_z', 'dt_min', 'eps']

    values_0 = np.array([10000]) # 300, 500, 300, 4000
    values_1 = np.array([10000])  # 35, 200, 3000
    values_2 = np.array([2.0]) # 0.1, 0.05
    values_3 = np.array([1.0e-7]) # 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9
    

    # Data type of each array.
    data_types = [int, int, float, float]

    values = [values_0, values_1, values_2, values_3]


# RESOLUTION STUDY.
elif exp == 'resolution':

    #yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_resolution.yaml"
    #yaml_file_name = "nix_params_resolution.yaml"

    yaml_file_path = "/scratch/ulb/glaciol/dmoreno/nix/par/nix_params_parallel_nic5.yaml"
    yaml_file_name = "nix_params_parallel_nic5.yaml"

    var_names = ['n', 'n_z', 'dt_min']
    #values_0 = np.array([2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14])

    values_0 = np.array([3000, 2**12, 5000, 6000, 7000, 2**13, 9000, 10000, 2**14]) # [25, 50, 100, 150, 300, 600]
    values_1 = np.array([10]) # 0.01
    values_2 = np.array([0.05, 0.01]) # 0.1


    """values_0 = np.array([2**4]) # [25, 50, 100, 150, 300, 600]
    values_1 = np.array([10]) # 0.01
    values_2 = np.array([0.1]) # 0.01"""

    # Data type of each array.
    data_types = [int, int, float]

    values = [values_0, values_1, values_2]


# MISMIP_3 WITH ICE RATE FACTOR CONSTANT AND T_OCE FORCING.
elif exp == 'T_oce_A':
    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_mismip_A.yaml"
    yaml_file_name = "nix_params_mismip_A.yaml"

    var_names = ['n', 'n_z']

    values_0 = np.array([250])
    values_1 = np.array([25])

    # Data type of each array.
    data_types = [int, int]

    values = [values_0, values_1]


# MISMIP_3 WITH T_OCE FORCING (CONSTANT T_AIR).
elif exp == 'T_oce':
    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_mismip_therm_T_oce.yaml"
    yaml_file_name = "nix_params_mismip_therm_T_oce.yaml"

    var_names = ['n', 'n_z', 'T_air', 'gamma_T']

    values_0 = np.array([250])
    values_1 = np.array([35]) # 35
    values_2 = np.array([233.15]) # [173.15, 183.15, 193.15]
    values_3 = np.array([40.0e-5, 70.0e-5, 100.0e-5]) # [40.0e-5, 60.0e-5, 80.0e-5, 100.0e-5]

    # Data type of each array.
    data_types = [int, int, float, float]

    values = [values_0, values_1, values_2, values_3]


# MISMIP_3 WITH T_AIR FORCING (NO SHELF MELT).
elif exp == 'T_air':
    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_mismip_therm_T_air.yaml"
    yaml_file_name = "nix_params_mismip_therm_T_air.yaml"

    var_names = ['n', 'n_z']

    values_0 = np.array([250])
    values_1 = np.array([25, 30, 35])  # 25

    # Data type of each array.
    data_types = [int, int]

    values = [values_0, values_1]


# SENSITIVITY TESTS.
elif exp == 'T_oce_A_sensitivity':
    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_mismip_A.yaml"
    yaml_file_name = "nix_params_mismip_A.yaml"

    var_names = ['n', 'n_z', 'gamma_T']

    values_0 = np.array([250])
    values_1 = np.array([25])
    values_2 = np.array([0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3])

    # Data type of each array.
    data_types = [int, int, float]

    values = [values_0, values_1, values_2]

elif exp == 'T_oce_therm_sensitivity':
    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_mismip_therm_T_oce.yaml"
    yaml_file_name = "nix_params_mismip_therm_T_oce.yaml"

    var_names = ['n', 'n_z', 'gamma_T']

    values_0 = np.array([250])
    values_1 = np.array([25])
    values_2 = np.array([0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3])

    # Data type of each array.
    data_types = [int, int, float]

    values = [values_0, values_1, values_2]


# TRANSITION INDICATORS T_OCE FORCING (CONSTANT T_AIR).
elif exp == 'ews_schoof_T_oce':
    yaml_file_path = "/home/dmoreno/scr/nix/par/nix_params_ews_therm_T_oce.yaml"
    yaml_file_name = "nix_params_ews_therm_T_oce.yaml"

    var_names = ['n', 'n_z', 'T_air', 'gamma_T', 'tf_bc']

    values_0 = np.array([200, 400]) # 200, 400, For some reason in now crashing on 300??
    values_1 = np.array([35])
    values_2 = np.array([233.15]) # [173.15, 183.15, 193.15]
    values_3 = np.array([80.0e-5]) # [40.0e-5, 60.0e-5, 80.0e-5, 100.0e-5]
    values_4 = np.array([45.1e3, 45.25e3, 45.5e3, 46.0e3]) # [35.1e3, 35.25e3, 35.5e3, 36.0e3]

    # Data type of each array.
    data_types = [int, int, float, float, float]

    values = [values_0, values_1, values_2, values_3, values_4]

else:
    print('Experiment not recognised.')


#######################################################################
#######################################################################

# Initialize lists to store string representations of values
values_str = []

# Convert each array to a list of strings
for array in values:
    values_str.append([str(value) for value in array])

# Create a string with all input values.
#str_all = list(map(list, zip(*values_str)))  # Transpose the list of lists
str_all = values_str

# Include variable names in each value for folder naming.
values_all = []

for i, var_name in enumerate(var_names):
    values_with_names = [f"{var_name}.{s}" for s in str_all[i]]
    values_all.append(values_with_names)


########################################################################
# Read the YAML file
with open(yaml_file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)


# User defined directories in nix_params.yaml.
path_nix    = yaml_data['path']['nix']
path_output = yaml_data['path']['out']


# Initialize name list
name = []

# Folder name given from each permutations.
# itertools.product() gives the Cartesian product of input lists.
for r in itertools.product(*values_all): 
    name.append('_'.join(r))


# Future update: compact all values arrays?
#perm = all_permutations(values_0, values_1, values_2, values_3)

# Combine all values_i arrays into a single tuple.
# The *array syntax in Python is called the "unpacking operator" or "splat operator." 
# When used in a function call, it unpacks the elements of an iterable 
# (such as a tuple, list, or any iterable object) into individual arguments.
#all_values = (values_0, values_1, values_2, values_3)
perm = all_permutations(*values)


# Loop over all permutations.
for i in range(len(name)):

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
    config = 'parallel'

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

    elif config == 'parallel':
        
        # Compiling command. -std=c++17. -O2
        # Use nix_solver_omp.cpp to test parallelization.
        cmd = "g++ -std=c++17 -fopenmp -O3 -I /usr/include/eigen3/ -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lnetcdf -lyaml-cpp"

    elif config == 'brigit':
        
        # Compiling command.
        cmd = "g++ -std=c++17 -I/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/include/ -I/usr/include/eigen3/ -L/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/lib/ -lnetcdf -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp"
        
        # In Brigit, we need a submit.sh file to send job to the queue.
        shutil.copyfile(path_nix+'submit.sh', path_modified+'src/submit.sh')

    elif config == 'nic5':

        # In clusters, we need a submit.sh file to send job to the queue.
        shutil.copyfile(path_nix+'submit_nic5.sh', path_modified+'submit_nic5.sh')

        # Change directory to the current modified one.
        os.chdir(path_modified)

        # Load all necessary modules before compilation.
        # Combine all commands into a single shell command.
        # Each subprocess.run() call starts a new shell process, and environment changes (like loading modules) are not shared between these processes.
        module_netcdf = "/opt/cecisw/arch/easybuild/2023b/modules/all/netCDF/"
        module_eigen  = "/opt/cecisw/arch/easybuild/2023b/modules/all/Eigen/"
        lib_netcdf    = "/opt/cecisw/arch/easybuild/2023b/software/netCDF/4.9.2-gompi-2023b/lib/"

        # "g++ -std=c++17 -fopenmp -O3 -I"+module_netcdf+" -I"+module_eigen+" -L"+lib_netcdf+" -lnetcdf -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lyaml-cpp",
        # "g++ -std=c++17 -I"+module_netcdf+" -I"+module_eigen+" -L"+lib_netcdf+" -lnetcdf -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lyaml-cpp",
        # Too agrressive paralelization gives problem in nic5!!! We need: -fopenmp -O1.
        # Compile: nix_solver_omp.cpp to test linear solver alone.
        commands = [
                    "module --force purge",
                    "module load releases/2023b",
                    "module load Eigen/3.4.0-GCCcore-13.2.0",
                    "module load yaml-cpp",
                    "module load netCDF/4.9.2-gompi-2023b",
                    "g++ -std=c++17 -fopenmp -O1 -I"+module_netcdf+" -I"+module_eigen+" -L"+lib_netcdf+" -lnetcdf -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lyaml-cpp",
                    ]

        cmd = " &&\n".join(commands)


    elif config == 'lemaitre4':

        # In clusters, we need a submit.sh file to send job to the queue.
        shutil.copyfile(path_nix+'submit_lemaitre4.sh', path_modified+'submit_lemaitre4.sh')

        # Change directory to the current modified one.
        os.chdir(path_modified)

        # Load all necessary modules before compilation.
        # Combine all commands into a single shell command.
        # Each subprocess.run() call starts a new shell process, and environment changes (like loading modules) are not shared between these processes.
        # THIS NEEDS TO BE UPDATED FOR LEMAITRE4!!!!
        module_netcdf = "/opt/cecisw/arch/easybuild/2023b/modules/all/netCDF/"
        module_eigen  = "/opt/cecisw/arch/easybuild/2023b/modules/all/Eigen/"
        lib_netcdf    = "/opt/cecisw/arch/easybuild/2023b/software/netCDF/4.9.2-gompi-2023b/lib/"

        # "g++ -std=c++17 -fopenmp -O3 -I"+module_netcdf+" -I"+module_eigen+" -L"+lib_netcdf+" -lnetcdf -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lyaml-cpp",
        # "g++ -std=c++17 -I"+module_netcdf+" -I"+module_eigen+" -L"+lib_netcdf+" -lnetcdf -o "+path_modified+"nix.o "+path_output_scr+"nix.cpp -lyaml-cpp",
        # Too agrressive paralelization gives problem in nic5!!! We need: -fopenmp -O1.
        commands = [
                    "module --force purge",
                    "module load releases/2023b",
                    "module load Eigen/3.4.0-GCCcore-13.2.0",
                    "module load yaml-cpp",
                    "module load netCDF/4.9.2-gompi-2023b",
                    "g++ -std=c++17 -fopenmp -O1 -I"+module_netcdf+" -I"+module_eigen+" -L"+lib_netcdf+" -lnetcdf -o "+path_modified+"nix.o "+path_output_scr+"nix_solver.cpp -lyaml-cpp",
                    ]

        cmd = " &&\n".join(commands)

            

    # Create text file for terminal output. "wb" for unbuffered output.
    f = open(path_modified+"out.txt", "wb")

    # Print compiling config.
    print('')
    print('-> Compiling configuration: ', config)
    print('')

        
    # Run Nix in background. Note that the solution is stored in nc file.
    # In Brigit, we need submit.sh to send it to the queue.
    if config == 'brigit':

        # Compile nix with subprocess.
        subprocess.run(cmd, shell=True, check=True, \
                    stdout=f, universal_newlines=True)


        # Try changing working directory and then running sbatch there.
        os.chdir(path_modified)
        cmd_run = "sbatch submit.sh"


    elif config == 'nic5':
        # Compile nix with subprocess.
        subprocess.run(cmd, shell=True, check=True, universal_newlines=True)

        # Necessary to change directory to run therein.
        os.chdir(path_modified)
        #cmd_run = "sbatch submit_ceci.sh"

        cmd_run = "sbatch --chdir="+path_modified+" submit_nic5.sh"

    elif config == 'lemaitre4':
        # Compile nix with subprocess.
        subprocess.run(cmd, shell=True, check=True, universal_newlines=True)

        # Necessary to change directory to run therein.
        os.chdir(path_modified)
        #cmd_run = "sbatch submit_ceci.sh"

        cmd_run = "sbatch --chdir="+path_modified+" submit_lemaitre4.sh"


    elif config == 'parallel' or config == 'iceshelf':
        # Compile nix with subprocess.
        subprocess.run(cmd, shell=True, check=True, \
                    stdout=f, universal_newlines=True)

        cmd_run = path_modified+"nix.o &"


    print('')
    print('-> Nix compiled.')
    print('')

    print('')
    print('-> Running Nix.')
    print('')

    # Run Nix model.
    p = subprocess.Popen(cmd_run, shell=True, \
                            stdout=f, universal_newlines=True)


