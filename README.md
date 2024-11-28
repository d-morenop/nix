# Nix 2D ice-sheet model.

Daniel Moreno-Parada
daniel.moreno.parada@ulb.be


Nix is an open source software available under the Creative Commons Attribution 4.0 International license. A thorough model description can be found in: Moreno-Parada, D., Robinson, A., Montoya, M., and Alvarez-Solas, J.: Description and validation of the ice sheet model Nix v1.0, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-2690, 2023. 

The model has been derived from scratch with a clear Application Programming Interface (API). It is written in C/C++ for efficiency and extremely fast computing (see Appendix H) and is readily available to run in any High Performance Computing Cluster. There are two key dependencies: NetCDF (Rew and Davis, 1990; Brown et al., 1993) and Eigen (Guennebauda et al., 2010) libraries. The former handles tasks for convenient community-standard input/output capability, whereas the latter serves to define vectors, matrices and further necessary computations (Fig. 1). Nix users can optionally select parallel computing (supported by Eigen library) simply by enabling OpenMP on the employed compiler, particularly convenient for high resolutions in the Blatter- Pattyn approximation, where large sparse matrices must be inverted. Moreover, it is also possible to use Eigen’s matrices, vectors, and arrays for fixed size within CUDA kernels (Nickolls et al., 2008).

Nix’s design offers a friendly Python wrapper module that handles directory management and compilation, though it can be compiled and run independently. The exact version used to produce the results of this work is archived at a persistent Zenodo repository (Moreno-Parada et al., 2023) while the latest version can be accessed on GitHub at: https://github.com/d-morenop/nix.


NIX INSTALLATION.

Nix installation in any Unix/MacOS system machine is straightforward. If you are a Windows user, please we encourage you to first set up any virtual machine.

Follow these steps on your Ubuntu/MacOS terminal:

Clone GitHub repository:
$ git clone https://github.com/d-morenop/nix

C++ compiler. Nix preferably uses g++ (gcc is also possible, but not recommended):
$ sudo apt install g++

Eigen library (matrices and vector calculations): 
$ sudo apt install libeigen3-dev

NetCDF:
$ sudo apt install libnetcdf-dev

Yaml library to import and read param files.
$ sudo apt install libyaml-cpp-dev

That is all! You have now installed Nix on your computer :)

Tips for your first runs
Nix can be simply run using the python wrapper:
$ python run_nix.py

This command line will handle everything: directories, editing namelists, compilation and executing Nix. Make sure to edit your input/output directories in the corresponding param file located in: ‘nix/par/’. 

If you are running Nix locally, then the config option in “run_nix.py” should be “iceshelf” (default). Otherwise, if you are submitting the job in a cluster, then pick “brigit”.

Large ensemble of simulations can be also run by providing a list of values within the desired experiment. For instance, in the resolution experiment (i.e., exp=’resolution’), the user can modify the total number of horizontal points in the simulation by editing the list taken by the numpy array:

values_0 = np.array( [2**5, 2**6, 2**7, 2**8, 2**9] )

Simulation status can be checked in out.txt. Nix output is saved in nix.nc in the desired directory. 


For furhter and more complex simualtions with active thermodynamics and higher-order velocity solvers, we refer the user to the manuscript (Moreno-Parada et al., 2013; https://doi.org/10.5194/egusphere-2023-2690).
