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



VectorXd f_sol(vector<Eigen::Triplet<double>> triplets, VectorXd b, int N, int maxIter, double tol)
{
    BiCGSTAB<SparseMatrix<double,RowMajor> > solver;
    solver.setMaxIterations(maxIter);
    solver.setTolerance(tol);

    SparseMatrix<double,RowMajor> A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    solver.compute(A);
    VectorXd x = solver.solve(b.segment(0, N));

    return x;
}




// NIX ICE SHEET MODEL.
// Driver method.
int main()
{
    // Wall time for computational speed.
    auto begin = std::chrono::high_resolution_clock::now();
    
    int omp_threads = 1;
    omp_set_num_threads(omp_threads); // Set the number of threads to 4
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of OPM threads: " << omp_get_num_threads() << std::endl;
    }

    // Enable OpenMP if supported by the compiler.
    // Is this needed/ 
    // export OMP_DISPLAY_ENV=TRUE
    // export OMP_NUM_THREADS=8


    // Set Eigen to use multiple threads.
    int num_threads = 1;
    Eigen::setNbThreads(num_threads);
    std::cout << "Using " << Eigen::nbThreads() << " eigen threads.\n";


    // Specify the path to YAML file.
    string yaml_name = "nix_params_parallel_nic5.yaml";
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

    
    int N = n*n_z;
    //int N_1 = 0.5 * N;
    //int N_1 = static_cast<int>(0.5 * N);

    
    // Set tolerance and maximum number of iterations.
    // THIS VALUE IS CRITICAL TO AVOID NUMERICAL INSTABILITIES!!!
    int maxIter = 100;                   // 1000. Working: 10000. 1000000
    double tol  = 1.0e-3;                // Currently:  Working: 1.0e-8, 1.0e-10
    
    

    // Wall time for computational speed.
    auto begin_parallel = std::chrono::high_resolution_clock::now();

    if ( omp_threads == 1 )
    {
        // Solve the sparse problem for each partition (sub-matrix).
        VectorXd x = f_sol(tripletList, b, N, maxIter, tol);
    }

    // Divide the original sparse matrix in smaller partitions to be solved in parallel.
    else
    {
        // Convert Eigen vector to std::vector
        ArrayXd eigen_vector = ArrayXd::LinSpaced(omp_threads, 0, (1.0 - (1.0/double(omp_threads)) ) * double(N));

        // Convert Eigen vector into std::vector.
        std::vector<int> partitions(eigen_vector.data(), eigen_vector.data() + eigen_vector.size());


        //std::vector<int> partitions = {0, N_1, N_2, N_3}; // Define your partitions. {0, N_1, N_2, N_3}
        std::vector<std::vector<Eigen::Triplet<double>>> triplet_subsections(partitions.size(), 
                                                                            std::vector<Eigen::Triplet<double>>());

        // Dive the triplets that contains the priginal sparse matrix in smaller partitions.
        // The total number of partitions equates the number of parallel threads.
        for (const auto& triplet : tripletList) 
        {
            // Determine which subsection the row belongs to
            int row_section = std::distance(partitions.begin(), 
                std::upper_bound(partitions.begin(), partitions.end(), triplet.row())) - 1;

            // Determine which subsection the column belongs to
            int col_section = std::distance(partitions.begin(), 
                std::upper_bound(partitions.begin(), partitions.end(), triplet.col())) - 1;

            // Ensure that the triplet goes into the correct subsection (only square blocks are stored)
            if (row_section == col_section) 
            {
                //cout << "row_section: " << row_section << endl;

                triplet_subsections[row_section].push_back(
                    Eigen::Triplet<double>(
                        triplet.row() - partitions[row_section], 
                        triplet.col() - partitions[col_section], 
                        triplet.value()
                    )
                );
            }
        }

        // Create parallel region to run the partitions of the matrix in parallel.
        #pragma omp parallel
        {
            // Only one thread is creating all the additional threads (avoids overhead).
            #pragma omp single
            {
                std::vector<VectorXd> x_subsections(triplet_subsections.size());

                // Loop over all partitions to create one thread per partition.
                for (size_t i = 0; i < triplet_subsections.size(); ++i) 
                {
                    #pragma omp task firstprivate(i)
                    {
                        // Solve the sparse problem for each partition (sub-matrix).
                        x_subsections[i] = f_sol(triplet_subsections[i], b, partitions[1], maxIter, tol);
                    }
                }
            }
        }
    }
    




    /*
    
    if ( omp_threads == 1 )
    {
        BiCGSTAB<SparseMatrix<double,RowMajor> > solver;
        solver.setMaxIterations(maxIter);
        solver.setTolerance(tol);

        SparseMatrix<double,RowMajor> A_sparse(N, N); 

        // Define your sparse matrix A_spare from triplets.
        A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

        // Solver.
        solver.compute(A_sparse);

        // Solve without guess (assumes x = 0).
        VectorXd x = solver.solve(b);
    }

    if ( omp_threads == 2 )
    {

        std::vector<int> partitions = {0, N_1}; // Define your partitions
        std::vector<std::vector<Eigen::Triplet<double>>> triplet_subsections(partitions.size() - 1, 
                                                                            std::vector<Eigen::Triplet<double>>());


        #pragma omp parallel
        {
            #pragma omp single
            {
                std::vector<VectorXd> x_subsections(triplet_subsections.size());

                for (size_t i = 0; i < triplet_subsections.size(); ++i) 
                {
                    #pragma omp task firstprivate(i)
                    {
                        x_subsections[i] = f_sol(triplet_subsections[i], b, partitions[i + 1] - partitions[i], maxIter, tol);
                    }
                }
            }
        } // End of parallel region

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                VectorXd x_1 = f_sol(triplet_subsections[0], b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_2 = f_sol(triplet_subsections[1], b, N_1, maxIter, tol);
            }
        }


    }


    if ( omp_threads == 4 )
    {
        int N_1 = static_cast<int>(0.25 * N);
        int N_2 = static_cast<int>(0.5 * N);
        int N_3 = static_cast<int>(0.75 * N);

        vector<Eigen::Triplet<double>> triplets_1, triplets_2, triplets_3, triplets_4;

        for (const auto& triplet : tripletList) 
        {
            if (triplet.row() < N_1 && triplet.col() < N_1) 
            {
                triplets_1.push_back(triplet);
            } 

            else if (triplet.row() >= N_1 && triplet.row() < N_2 && triplet.col() >= N_1 && triplet.col() < N_2 ) 
            {
                triplets_2.push_back(Eigen::Triplet<double>(triplet.row() - N_1, triplet.col() - N_1, triplet.value()));
            }

            else if (triplet.row() >= N_2 && triplet.row() < N_3 && triplet.col() >= N_2 && triplet.col() < N_3 ) 
            {
                triplets_3.push_back(Eigen::Triplet<double>(triplet.row() - N_2, triplet.col() - N_2, triplet.value()));
            }

            else if (triplet.row() >= N_3 && triplet.col() >= N_3 ) 
            {
                triplets_4.push_back(Eigen::Triplet<double>(triplet.row() - N_3, triplet.col() - N_3, triplet.value()));
            }

        }


        #pragma omp parallel sections
        {
            #pragma omp section
            {
                VectorXd x_1 = f_sol(triplets_1, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_2 = f_sol(triplets_2, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_3 = f_sol(triplets_3, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_4 = f_sol(triplets_4, b, N_1, maxIter, tol);
            }
        }

 

    }


    if ( omp_threads == 8 )
    {
        int N_1 = static_cast<int>(0.125 * N);
        int N_2 = static_cast<int>(0.25 * N);
        int N_3 = static_cast<int>(0.375 * N);
        int N_4 = static_cast<int>(0.5 * N);
        int N_5 = static_cast<int>(0.625 * N);
        int N_6 = static_cast<int>(0.750 * N);
        int N_7 = static_cast<int>(0.875 * N);

        vector<Eigen::Triplet<double>> triplets_1, triplets_2, triplets_3, triplets_4, triplets_5, triplets_6, triplets_7, triplets_8;

        for (const auto& triplet : tripletList) 
        {
            if (triplet.row() < N_1 && triplet.col() < N_1) 
            {
                triplets_1.push_back(triplet);
            } 

            else if (triplet.row() >= N_1 && triplet.row() < N_2 && triplet.col() >= N_1 && triplet.col() < N_2 ) 
            {
                triplets_2.push_back(Eigen::Triplet<double>(triplet.row() - N_1, triplet.col() - N_1, triplet.value()));
            }

            else if (triplet.row() >= N_2 && triplet.row() < N_3 && triplet.col() >= N_2 && triplet.col() < N_3 ) 
            {
                triplets_3.push_back(Eigen::Triplet<double>(triplet.row() - N_2, triplet.col() - N_2, triplet.value()));
            }

            else if (triplet.row() >= N_3 && triplet.row() < N_4 && triplet.col() >= N_3 && triplet.col() < N_4 ) 
            {
                triplets_4.push_back(Eigen::Triplet<double>(triplet.row() - N_3, triplet.col() - N_3, triplet.value()));
            }

            else if (triplet.row() >= N_4 && triplet.row() < N_5 && triplet.col() >= N_4 && triplet.col() < N_5 ) 
            {
                triplets_5.push_back(Eigen::Triplet<double>(triplet.row() - N_4, triplet.col() - N_4, triplet.value()));
            }

            else if (triplet.row() >= N_5 && triplet.row() < N_6 && triplet.col() >= N_5 && triplet.col() < N_6 ) 
            {
                triplets_6.push_back(Eigen::Triplet<double>(triplet.row() - N_5, triplet.col() - N_5, triplet.value()));
            }

            else if (triplet.row() >= N_6 && triplet.row() < N_7 && triplet.col() >= N_6 && triplet.col() < N_7 ) 
            {
                triplets_7.push_back(Eigen::Triplet<double>(triplet.row() - N_6, triplet.col() - N_6, triplet.value()));
            }

            else if (triplet.row() >= N_7 && triplet.col() >= N_7 ) 
            {
                triplets_8.push_back(Eigen::Triplet<double>(triplet.row() - N_7, triplet.col() - N_7, triplet.value()));
            }

        }
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                VectorXd x_1 = f_sol(triplets_1, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_2 = f_sol(triplets_2, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_3 = f_sol(triplets_3, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_4 = f_sol(triplets_4, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_5 = f_sol(triplets_5, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_6 = f_sol(triplets_6, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_7 = f_sol(triplets_7, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_8 = f_sol(triplets_8, b, N_1, maxIter, tol);
            }
        }

    }


    if ( omp_threads == 16 )
    {
        int N_1  = static_cast<int>(1 * N / 16);
        int N_2  = static_cast<int>(2 * N / 16);
        int N_3  = static_cast<int>(3 * N / 16);
        int N_4  = static_cast<int>(4 * N / 16);
        int N_5  = static_cast<int>(5 * N / 16);
        int N_6  = static_cast<int>(6 * N / 16);
        int N_7  = static_cast<int>(7 * N / 16);
        int N_8  = static_cast<int>(8 * N / 16);
        int N_9  = static_cast<int>(9 * N / 16);
        int N_10 = static_cast<int>(10 * N / 16);
        int N_11 = static_cast<int>(11 * N / 16);
        int N_12 = static_cast<int>(12 * N / 16);
        int N_13 = static_cast<int>(13 * N / 16);
        int N_14 = static_cast<int>(14 * N / 16);
        int N_15 = static_cast<int>(15 * N / 16);

        vector<Eigen::Triplet<double>> triplets_1, triplets_2, triplets_3, triplets_4,  \
                                       triplets_5, triplets_6, triplets_7, triplets_8, \
                                       triplets_9, triplets_10, triplets_11, triplets_12, \
                                       triplets_13, triplets_14, triplets_15, triplets_16;

        for (const auto& triplet : tripletList) 
        {
            if (triplet.row() < N_1 && triplet.col() < N_1) 
            {
                triplets_1.push_back(triplet);
            } 

            else if (triplet.row() >= N_1 && triplet.row() < N_2 && triplet.col() >= N_1 && triplet.col() < N_2 ) 
            {
                triplets_2.push_back(Eigen::Triplet<double>(triplet.row() - N_1, triplet.col() - N_1, triplet.value()));
            }

            else if (triplet.row() >= N_2 && triplet.row() < N_3 && triplet.col() >= N_2 && triplet.col() < N_3 ) 
            {
                triplets_3.push_back(Eigen::Triplet<double>(triplet.row() - N_2, triplet.col() - N_2, triplet.value()));
            }

            else if (triplet.row() >= N_3 && triplet.row() < N_4 && triplet.col() >= N_3 && triplet.col() < N_4 ) 
            {
                triplets_4.push_back(Eigen::Triplet<double>(triplet.row() - N_3, triplet.col() - N_3, triplet.value()));
            }

            else if (triplet.row() >= N_4 && triplet.row() < N_5 && triplet.col() >= N_4 && triplet.col() < N_5 ) 
            {
                triplets_5.push_back(Eigen::Triplet<double>(triplet.row() - N_4, triplet.col() - N_4, triplet.value()));
            }

            else if (triplet.row() >= N_5 && triplet.row() < N_6 && triplet.col() >= N_5 && triplet.col() < N_6 ) 
            {
                triplets_6.push_back(Eigen::Triplet<double>(triplet.row() - N_5, triplet.col() - N_5, triplet.value()));
            }

            else if (triplet.row() >= N_6 && triplet.row() < N_7 && triplet.col() >= N_6 && triplet.col() < N_7 ) 
            {
                triplets_7.push_back(Eigen::Triplet<double>(triplet.row() - N_6, triplet.col() - N_6, triplet.value()));
            }

            else if (triplet.row() >= N_7 && triplet.row() < N_8 && triplet.col() >= N_7 && triplet.col() < N_8 ) 
            {
                triplets_8.push_back(Eigen::Triplet<double>(triplet.row() - N_7, triplet.col() - N_7, triplet.value()));
            }

            else if (triplet.row() >= N_8 && triplet.row() < N_9 && triplet.col() >= N_8 && triplet.col() < N_9 ) 
            {
                triplets_9.push_back(Eigen::Triplet<double>(triplet.row() - N_8, triplet.col() - N_8, triplet.value()));
            }

            else if (triplet.row() >= N_9 && triplet.row() < N_10 && triplet.col() >= N_9 && triplet.col() < N_10 ) 
            {
                triplets_10.push_back(Eigen::Triplet<double>(triplet.row() - N_9, triplet.col() - N_9, triplet.value()));
            }

            else if (triplet.row() >= N_10 && triplet.row() < N_11 && triplet.col() >= N_10 && triplet.col() < N_11 ) 
            {
                triplets_11.push_back(Eigen::Triplet<double>(triplet.row() - N_10, triplet.col() - N_10, triplet.value()));
            }

            else if (triplet.row() >= N_11 && triplet.row() < N_12 && triplet.col() >= N_11 && triplet.col() < N_12 ) 
            {
                triplets_12.push_back(Eigen::Triplet<double>(triplet.row() - N_11, triplet.col() - N_11, triplet.value()));
            }

            else if (triplet.row() >= N_12 && triplet.row() < N_13 && triplet.col() >= N_12 && triplet.col() < N_13 ) 
            {
                triplets_13.push_back(Eigen::Triplet<double>(triplet.row() - N_12, triplet.col() - N_12, triplet.value()));
            }

            else if (triplet.row() >= N_13 && triplet.row() < N_14 && triplet.col() >= N_13 && triplet.col() < N_14 ) 
            {
                triplets_14.push_back(Eigen::Triplet<double>(triplet.row() - N_13, triplet.col() - N_13, triplet.value()));
            }

            else if (triplet.row() >= N_14 && triplet.row() < N_15 && triplet.col() >= N_14 && triplet.col() < N_15 ) 
            {
                triplets_15.push_back(Eigen::Triplet<double>(triplet.row() - N_14, triplet.col() - N_14, triplet.value()));
            }

            else if (triplet.row() >= N_15 && triplet.col() >= N_15 ) 
            {
                triplets_16.push_back(Eigen::Triplet<double>(triplet.row() - N_15, triplet.col() - N_15, triplet.value()));
            }

        }
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                VectorXd x_1 = f_sol(triplets_1, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_2 = f_sol(triplets_2, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_3 = f_sol(triplets_3, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_4 = f_sol(triplets_4, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_5 = f_sol(triplets_5, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_6 = f_sol(triplets_6, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_7 = f_sol(triplets_7, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_8 = f_sol(triplets_8, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_9 = f_sol(triplets_9, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_10 = f_sol(triplets_10, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_11 = f_sol(triplets_11, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_12 = f_sol(triplets_12, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_13 = f_sol(triplets_13, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_14 = f_sol(triplets_14, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_15 = f_sol(triplets_15, b, N_1, maxIter, tol);
            }

            #pragma omp section
            {
                VectorXd x_16 = f_sol(triplets_16, b, N_1, maxIter, tol);
            }
        }

    }
    */

    // Write solution with desired output frequency.
    // Running time (measures wall time).
    auto end     = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    auto elapsed_parallel = chrono::duration_cast<chrono::nanoseconds>(end - begin_parallel);

    // Print computational time.
    printf("\n Time measured         : %.3f ms.\n", elapsed.count() * 1e-6);
    printf("\n Time measured in potential parallel region: %.3f ms.\n", elapsed_parallel.count() * 1e-6);
    printf("\n Fraction of execution time that can benefit from paralelization, p = %.3f \n", elapsed_parallel.count() / elapsed.count());

    return 0;
    /////////////////////////////////////////////////////////////////////////////////////////////

}