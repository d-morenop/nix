#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:23:54 2022

@author: dmoreno
"""

import os
import subprocess
import sys
import shutil
from subprocess import Popen, PIPE, STDOUT



# PYTHON SCRIPT TO COMPILE AND RUN THE FLOWLINE MODEL.
# This script copies flowline.cpp and its dependencies to a desired folder
# where it is compiled and run. The simulation output is then saved in a .nc file
# where the corresponding directory must be specified in write_nc.cpp and it may differ 
# from the compiling/running directory.
# 
# Daniel Moreno Parada (danielm@ucm.es).


# User defined directories.
path_flowline = "/home/dmoreno/c++/git/flowline/"
path_output   = "/home/dmoreno/c++/git/flowline/output/mismip/exp3/exp3_n.250/"


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
    isfile = os.path.isdir(path_output+'*.o')
    if isfile == True:
        subprocess.run("rm "+path_output+"*.o", shell=True, check=True, \
                           stdout=subprocess.PIPE, universal_newlines=True)
    

# Copy main script and write_nc to the output folder for compilation therein.
shutil.copyfile(path_flowline+'flow_line_mismip_4.cpp', path_output+'flow_line_mismip_4.cpp')
shutil.copyfile(path_flowline+'write_nc.cpp', path_output+'write_nc.cpp')


# Compilation configuration. ['local', 'iceberg', 'brigit']
config = 'iceberg'

if config == 'local':
    cmd  = "g++ -I /path/to/eigen3/ -o "+path_output+"flow_line.o "+path_output+"flow_line.cpp -lnetcdf"
elif config == 'iceberg':
    cmd = "g++ -std=c++11 -I /usr/include/eigen3/ -o "+path_output+"flow_line_mismip_4.o "+path_output+"flow_line_mismip_4.cpp -lnetcdf"
elif config == 'brigit':
    cmd = "g++ -std=c++11 -I/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/include/ -I/usr/include/eigen3/ -L/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/lib/ -lnetcdf -o "+path_output+"flow_line.o "+path_output+"flow_line.cpp"


# Create text file for terminal output. 
f = open(path_output+"out.txt", "w")

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
    
# Run flowline in background. Note thatthe  solution is stored in nc file.
p = subprocess.Popen(path_output+"flow_line_mismip_4.o &", shell=True, \
                         stdout=f, universal_newlines=True)




