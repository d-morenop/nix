#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netcdf.h>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <unistd.h>
#include <omp.h> // Required for OpenMP

using namespace Eigen;
using namespace std;

#include "nix_readparams.cpp"
#include "nix_tools.cpp"
#include "read-write_nc.cpp"
#include "nix_dynamics.cpp"
#include "nix_thermodynamics.cpp"
#include "nix_material.cpp"
#include "nix_topography.cpp"
#include "nix_friction.cpp"
#include "nix_timestep.cpp"
#include "nix_iceflux.cpp"




// NIX ICE SHEET MODEL.
// Driver method.
int main()
{

    /*omp_set_num_threads(1); // Set the number of threads to 4
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of OPM threads: " << omp_get_num_threads() < std::endl;
    }*/

    // Enable OpenMP if supported by the compiler.
    // Is this needed/ 
    // export OMP_DISPLAY_ENV=TRUE
    // export OMP_NUM_THREADS=8


    // Set Eigen to use multiple threads.
    int num_threads = 1;
    Eigen::setNbThreads(num_threads);
    std::cout << "Using " << Eigen::nbThreads() << " eigen threads.\n";


    // Specify the path to YAML file.
    string yaml_name = "nix_params_parallel_ceci.yaml";
    //string yaml_name = "nix_params_mismip_therm_T_oce.yaml";

    // Assuming the path won't exceed 4096 characters.
    char buffer[4096];

    // Read the symbolic link /proc/self/exe, which points to the executable file of the calling process.
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);

    if (len != -1) 
    {
        // Null-terminate the string.
        buffer[len] = '\0'; 

        // Extract the directory from the full path.
        char *last_slash = strrchr(buffer, '/');
        if (last_slash != nullptr) 
        {
            // Null-terminate at the last slash to get the directory
            *last_slash = '\0'; 
            //std::cout << "Executable directory: " << buffer << std::endl;
        } 
        else 
        {
            std::cerr << "Error extracting directory." << std::endl;
        }
    } 
    else 
    {
        std::cerr << "Error getting executable path." << std::endl;
    }

    // Convert char to string and concatenate full path.
    string path = buffer;
    // Test.
    //path = "/scratch/ulb/glaciol/dmoreno/nix-iceshelf/nix/output/n.50_n_z.25_dt_min.1.0_eps.1e-07";
    string full_path = path+"/par/"+yaml_name;

    cout << "\n full_path = " << full_path;
    
    // Load the YAML file
    YAML::Node config = YAML::LoadFile(full_path);

    // Parse parameters
    NixParams nixParams;
    readParams(config, nixParams);

    // USE PARAM CLASS TO ACCESS SOME PARAMETERS IN YAML.
    auto [bed_exp, exp, n, n_z, grid, grid_exp, bedrock_ews] = nixParams.dom;
    auto [H_0, S_0, u_0, visc_0, theta_0, beta_0]            = nixParams.init;


    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////
    // Initialize Nix.

    // BED GEOMETRY.
    // Following Pattyn et al. (2012) the overdeepening hysterisis uses n = 250.
    // bed_exp = 1: "mismip_1", 3: "mismip_3", 4: "galcier_ews"
    //int const bed_exp = 3;

    // MISMIP EXPERIMENTS FORCING.
    // Number of steps in the A forcing.
    //  Exp_3: 13 (18 long), Exp_1-2: 17. T_air: 17, T_oce: 9. T_oce_f_q: 29
    int n_s;
    double L;

    // Initial position ice sheet depending on bed geometry.
    if ( bed_exp == "mismip_1" )
    {
        L = 694.5e3;
    }
    else if ( bed_exp == "mismip_3" )
    {
        L = 473.1e3;
    }
    else
    {
        cout << "\n Bed geometry not recognised. Please, select: mismip_1... ";
    }

    
    // Prepare variables for forcing.
    // BETTER TO USE AN EVEN NUMBER FOR n_s!!
    if ( exp == "constant_A" || exp == "constant_T_air" )
    {
        n_s = 1;
    }
    else if ( exp == "mismip_1" || exp == "mismip_1_therm" )
    {
        n_s = 17;
    }
    else if ( exp == "mismip_3_A" )
    {
        n_s = 42; // 23, 42
    }  
    else if ( exp == "mismip_3" )
    {
        n_s = 22; //17; 18;
    }  
    else if ( exp == "therm_T_air" )
    {
        n_s = 24; // 5, 8, 13
    }
    else if ( exp == "therm_T_oce" )
    {
        n_s = 46; // 42 
    }
    else if ( exp == "ews_christian" )
    {
        n_s = 2; 
        L   = 50.0e3;
    }
    else if ( exp == "ews_schoof" )
    {
        n_s = 2; 
    }
    else
    {
        cout << "\n Experiment forcing not recognised. Please, select: mismip_1... ";
    }



    // Normalised horizontal dimension.
    // Try an unevenly-spaced horizontal grid to allows for fewer point whilst keeping
    // high resolution at the grounding line.
    ArrayXd sigma = ArrayXd::LinSpaced(n, 0.0, 1.0);      // Dimensionless x-coordinates.
    ArrayXd ds(n);
    ArrayXd ds_inv(n);
    
    //double const n_sigma = 1.0;          // 0.5. Exponent of spacing in horizontal grid (1.0 = evenly-spaced). 
    sigma = pow(sigma, grid_exp);
    
    // Handy definitions for further finite differences.
    // Uneven spacing.
    for (int i=0; i<n-1; i++)
    {
        ds(i) = sigma(i+1) - sigma(i);
    }
    ds(n-1) = ds(n-2);

    ds_inv = 1.0 / ds;



    // We assume a constant viscosity in the first iteration. 1.0e8 Pa yr.
    ArrayXXd visc     = ArrayXXd::Constant(n, n_z, visc_0);            // [Pa yr]
    ArrayXd visc_bar = ArrayXd::Constant(n, visc_0);

    // Intialize ice thickness and SMB.
    ArrayXd H = ArrayXd::Constant(n, H_0); // 10.0
    ArrayXd S = ArrayXd::Constant(n, S_0); // 0.3


    // Initilize vertical discretization.
    ArrayXd dz(n);                                        // Vertical discretization (only x-dependecy for now).                                  
    dz = H / n_z;


    //////////////////////////////////////////////////////////////////////////////////////////
    // Update bedrock with new domain extension L.
    ArrayXd bed = f_bed(L, sigma, ds, 0.0, nixParams.dom);

    ArrayXd dx = L * ds;
    ArrayXd h = bed + H; 
    double L_inv = 1.0 / L;

    // Derivatives at the boundaries O(x).
    ArrayXd dhds = 0.5 * ( H + shift(H,-1,n) ) * ( shift(h,-1,n) - h ) * ds_inv;
    dhds(0)       = 0.5 * ( H(0) + H(1) ) * ( h(1) - h(0) ) * ds_inv(0);
    dhds(n-1) = H(n-1) * ( h(n-1) - h(n-2) ) * ds_inv(n-2); 
        
    // Inhomogeneous term.
    ArrayXd F = nixParams.cnst.rho * nixParams.cnst.g * dhds * L_inv;


    // Wall time for computational speed.
    auto begin = std::chrono::high_resolution_clock::now();

    // Initialize a triplet list to store non-zero entries.
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // Reserve memory for triplets.
    // 5 unkowns in a n*n_z array.
    tripletList.reserve(5 * (n-2) * (n_z-2));  

    ArrayXd dz_2_inv = 1.0 / pow(dz,2);
    ArrayXd gamma    = 4.0 / pow(dx, 2); // 4.0 / pow(dx, 2)

    // Inhomogeneous term: A*x = b.
    VectorXd b = VectorXd::Zero(n * n_z);

    // Original discretization.
    ArrayXXd gamma_mat = gamma.replicate(1, n_z);
    ArrayXXd dz_2_mat  = dz_2_inv.replicate(1, n_z);
 
    ArrayXXd c_x1 = gamma_mat * shift_2D(visc,-1,0);
    ArrayXXd c_x  = shift_2D(gamma_mat,1,0) * visc;

    ArrayXXd c_z1 = dz_2_mat * shift_2D(visc,0,-1); //(i,j+1);
    ArrayXXd c_z  = dz_2_mat * visc;


    // From the staggered grid definition, we should not take points at n_z-1 (j+1)
    // to calculate the velocities at j = n_z-2. 
    c_z1.col(n_z-2) = 0.0;

    ArrayXXd alpha_u = - ( c_x1 + c_x + c_z1 + c_z );
    
    // Loop through grid points
    for (int i=1; i<n-1; i++) 
    {
        for (int j=1; j<n_z-1; j++) 
        {
            // New index.
            int idx = i*n_z + j;

            // Add non-zero entries to the triplet list
            tripletList.push_back(T(idx, idx, alpha_u(i,j)));
            tripletList.push_back(T(idx, idx+n_z, c_x1(i,j)));
            tripletList.push_back(T(idx, idx-n_z, c_x(i,j)));
            tripletList.push_back(T(idx, idx+1, c_z1(i,j)));
            tripletList.push_back(T(idx, idx-1, c_z(i,j)));
            

            // Fill vector b.
            b(idx) = F(i);
        }
    }

    // Set the triplets in the sparse matrix
    // declares a column-major sparse matrix type of double.
    SparseMatrix<double,RowMajor> A_sparse(n*n_z, n*n_z); 
    
    // Define your sparse matrix A_spare from triplets.
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    // Solver.
    BiCGSTAB<SparseMatrix<double,RowMajor> > solver;
    solver.compute(A_sparse);

    // Set tolerance and maximum number of iterations.
    // THIS VALUE IS CRITICAL TO AVOID NUMERICAL INSTABILITIES!!!
    int maxIter = 1000;                   // 1000. Working: 10000. 1000000
    double tol  = 1.0e-6;                // Currently:  Working: 1.0e-8, 1.0e-10
    solver.setMaxIterations(maxIter);
    solver.setTolerance(tol);

    // Solve without guess (assumes x = 0).
    VectorXd x = solver.solve(b);

    // Write solution with desired output frequency.
    // Running time (measures wall time).
    auto end     = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    //double speed = 60 * 60 * 1.0e6 * (a(1) - a(0)) / elapsed.count();

    // Print computational time.
    printf("\n Time measured: %.3f ms.\n", elapsed.count() * 1e-6);

    return 0;
    /////////////////////////////////////////////////////////////////////////////////////////////

}