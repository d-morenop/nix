Nix is an open source software available under the Creative Commons Attribution 4.0 International license. A thorough model description can be found in: Moreno-Parada, D., Robinson, A., Montoya, M., and Alvarez-Solas, J.: Description and validation of the ice sheet model Nix v1.0, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-2690, 2023.

The model has been derived from scratch with a clear Application Programming Interface (API). It is written in C/C++ for efficiency and extremely fast computing (see Appendix H) and is readily available to run in any High Performance Computing Cluster. There are two key dependencies: NetCDF (Rew and Davis, 1990; Brown et al., 1993) and Eigen (Guennebauda et al., 2010) libraries. The former handles tasks for convenient community-standard input/output capability, whereas the latter serves to define vectors, matrices and further necessary computations (Fig. 1). Nix users can optionally select parallel computing (supported by Eigen library) simply by enabling OpenMP on the employed compiler, particularly convenient for high resolutions in the Blatter- Pattyn approximation, where large sparse matrices must be inverted. Moreover, it is also possible to use Eigen’s matrices, vectors, and arrays for fixed size within CUDA kernels (Nickolls et al., 2008).

Nix’s design offers a friendly Python wrapper module that handles directory management and compilation, though it can be compiled and run independently. The exact version used to produce the results of this work is archived at a persistent Zenodo repository (Moreno-Parada et al., 2023) while the latest version can be accessed on GitHub at: https://github.com/d-morenop/nix.

TEST EXPERIMENTS.

All MISMIP experiments (Pattyn et al., 2013) can be run simply with one code line:

$ python run_nix.py

This will create the output directories, compile Nix and run the simulations.

For furhter and more complex simualtions with active thermodynamics and higher-order velocity solvers, we refer the user to the manuscript (Moreno-Parada et al., 2013; https://doi.org/10.5194/egusphere-2023-2690).
