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

    omp_set_num_threads(1); // Set the number of threads to 4
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
    }

    // Enable OpenMP if supported by the compiler.
    // Is this needed/ 
    // export OMP_DISPLAY_ENV=TRUE
    // export OMP_NUM_THREADS=8


    // Set Eigen to use multiple threads
    int num_threads = 8;
    Eigen::setNbThreads(num_threads);
    std::cout << "Using " << Eigen::nbThreads() << " threads.\n";

    // Example sparse matrix size and number of non-zeros
    /*int n1 = 1024; // Matrix dimensions
    int n_z1 = 1024; // Example non-zero density factor
    SparseMatrix<double, RowMajor> A_sparse(n1 * n_z1, n1 * n_z1);

    // Define the triplets for sparse matrix initialization
    std::vector<Triplet<double>> tripletList;
    for (int i = 0; i < n1 * n_z1; ++i) {
        tripletList.emplace_back(i, i, 4.0); // Diagonal elements
        if (i + 1 < n1 * n_z1) tripletList.emplace_back(i, i + 1, -1.0); // Off-diagonal
        if (i - 1 >= 0) tripletList.emplace_back(i, i - 1, -1.0); // Off-diagonal
    }
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    // Define the solver
    BiCGSTAB<SparseMatrix<double>> solver;

    // Define the preconditioner
    IncompleteLUT<double> preconditioner;
    preconditioner.setDroptol(1.0e-4);
    solver.preconditioner().compute(A_sparse);

    // Compute the solver
    solver.compute(A_sparse);

    // Solver parameters
    solver.setMaxIterations(100);
    solver.setTolerance(1.0e-4);

    // Define the right-hand side vector
    VectorXd b = VectorXd::Random(n1 * n_z1);

    // Solve the system
    auto start1 = std::chrono::high_resolution_clock::now();
    VectorXd x = solver.solve(b);
    auto end1 = std::chrono::high_resolution_clock::now();

    // Output results
    std::cout << "Solver completed in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() 
              << " ms.\n";
    std::cout << "Converged in " << solver.iterations() << " iterations with error " 
              << solver.error() << ".\n";*/

    
    
    
    //Eigen::setNbThreads(12); // Set the number of threads. 4
    //cout << "Number of Eigen threads: " << Eigen::nbThreads() << " threads.\n";


    /*Eigen::MatrixXd A1 = Eigen::MatrixXd::Random(10000, 10000);
    Eigen::MatrixXd A2 = Eigen::MatrixXd::Random(10000, 10000);
    Eigen::VectorXd b = Eigen::VectorXd::Random(10000);


    // Solve without guess (assumes x = 0).
    auto start1 = std::chrono::high_resolution_clock::now();
    MatrixXd W = ( A1 * (A1 + A2) ) * A2; 
    Eigen::VectorXd x = W.partialPivLu().solve(b);
    auto end1 = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix solution in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count()
              << " ms.\n";


    //Eigen::setNbThreads(1); // Set the number of threads. 4
    cout << "Number of Eigen threads: " << Eigen::nbThreads() << " threads.\n";*/



    /*#ifdef _OPENMP
    {
        Eigen::setNbThreads(12); // Set the number of threads. 4
        cout << "Using Eigen with " << Eigen::nbThreads() << " threads.\n";
    }
    #endif*/

    /*omp_set_num_threads(12); // Set the number of threads to 4
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
    }*/


    //Eigen::setNbThreads(8);
    //std::cout << "Using " << Eigen::nbThreads() << " threads.\n";

    // Specify the path to YAML file.
    string yaml_name = "nix_params_parallel.yaml";
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
    string full_path = path+"/par/"+yaml_name;

    cout << "\n full_path = " << full_path;
    
    // Load the YAML file
    YAML::Node config = YAML::LoadFile(full_path);

    // Parse parameters
    NixParams nixParams;
    readParams(config, nixParams);

    // USE PARAM CLASS TO ACCESS SOME PARAMETERS IN YAML.
    auto [bed_exp, exp, n, n_z, grid, grid_exp, bedrock_ews] = nixParams.dom;
    auto [t0, tf, t_eq, output]                              = nixParams.tm;
    auto [t_n, out_hr]                                       = nixParams.tm.output;
    auto [n_picard, picard_tol, omega_1 , omega_2]           = nixParams.pcrd;
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
    if ( exp == "mismip_1" || exp == "mismip_1_therm" )
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


    // Prepare scalar variables.
    double t;                            // Time variable [yr].
    double dt;                           // Time step [yr].
    double dL_dt;                        // GL migration rate [m/yr]. 
    double m_stoch;                      // Stochatic contribution to calving [m/yr].     
    double smb_stoch;                    // Stochatic contribution to smb [m/yr].
    double T_oce_det;                    // Deterministic contribution to ocean temperature anomalies [K].
    double T_oce_stoch;                  // Stochatic contribution to ocean temperature anomalies [K].
    double alpha;                        // Fraction time to apply forcing trends [0,1].
    int t_stoch;                         // Stochactic time index.
    double T_air;                        // Current value of atmospheric temperature forcing [K] 
    double T_oce;                        // Current value of ocean temperature anomaly forcing [K]   
    double M;                      // Calving value from oceanic interaction [m/yr].
    
    // LATERAL BOUNDARY CONDITION.
    double D;                                        // Depth below the sea level [m].
    double u_x_bc;                                   // Boundary condition on u2 = dub/dx.
    double u2_dif;                                   // Difference between analytical and numerical.

    // Ice rate factor.
    double A, B;

    // PICARD ITERATION
    double error;                           // Norm of the velocity difference between iterations.
    double omega;                           // Angle between two consecutive velocities [rad]. 
    double mu;                              // Relaxation method within Picard iteration. 
    
    int c_picard;                           // Counter of Picard iterations.
    double speed;

    // PREPARE VARIABLES.
    ArrayXd H(n);                        // Ice thickness [m].
    ArrayXd ub(n);                       // Sliding Velocity [m/yr].
    ArrayXd u_bar(n);                    // Depth-integrated Velocity [m/yr].
    ArrayXd u_bar_x(n);                       // Velocity first derivative [1/yr].
    ArrayXd q(n);                        // Ice flux [m²/yr].
    ArrayXd bed(n);                      // Bedrock elevation [m].
    ArrayXd visc_bar(n);                 // Vertically-averaged ice viscosity [Pa·s].
    ArrayXd C_ref(n);                    // Reference riction coefficient [Pa m^-1/3 s^1/3].
    ArrayXd C_bed(n);                    // Friction coefficient [Pa m^-1/3 s^1/3].
    ArrayXd Q_fric(n);                   // Frictional heat [W / m^2].
    ArrayXd S(n);                        // Surface accumulation equivalent [mm/day].
    ArrayXd tau_b(n);                    // Basal friction [Pa]
    ArrayXd beta(n);                     // Basal friction [Pa m^-1 yr]
    ArrayXd tau_d(n);                    // Driving stress [Pa]
    ArrayXd u_bar_old_1(n);  
    ArrayXd u_bar_old_2(n);  
    ArrayXd u2_0_vec(n);                 // Ranged sampled of u2_0 for a certain iteration.
    ArrayXd u2_dif_vec(n);               // Difference with analytical BC.
    //ArrayXd w(n);                        // Synthetic vertical velocity.
    ArrayXd F_1(n);                      // Integral for DIVA solver (Arthern et al., 2015)
    ArrayXd F_2(n);                      // Integral for DIVA solver (Arthern et al., 2015)
    ArrayXd b_melt(n);                    // Basal melt [m/yr]
    
    // Stochastic matrices and vectors.
    int n_stoch = 3;                                           // Number of stochastic variables forcing.
    ArrayXXd noise(n_stoch, nixParams.stoch.N);               // Matrix to allocate stochastic BC.
    Array3d noise_now;                                        // Three-entry vector with current stochastic noise.
    ArrayXd t_vec = ArrayXd::LinSpaced(nixParams.stoch.N, 0.0, nixParams.stoch.N);  // Time vector with dt_noise time step.
    
    // Vectors to compute norm.
    VectorXd u_bar_vec(n); 
    VectorXd c_u_bar_1(n);                   // Correction vector Picard relaxed iteration.
    VectorXd c_u_bar_2(n);

    // FRONTAL ABLATION.
    ArrayXd M_s(n_s); 

    // MISMIP FORCING.
    ArrayXd A_s(n_s);                    // Rate factor values for MISMIP exp.
    ArrayXd t_s(n_s);                    // Time length for each step of A.
    ArrayXd T_air_s(n_s);                // Corresponding air temperature values to A_s. 
    ArrayXd T_oce_s(n_s);                // Ocean temperature forcing.   

    // MATRICES.
    ArrayXXd sol(n,n_z+1);               // Matrix output. sol(2*n+1,2*n_z+1);
    ArrayXXd sol_thrm(n,n_z+1);  
    ArrayXXd u(n,n_z);                   // Full velocity u(x,z) [m/yr].  
    ArrayXXd u_old(n,n_z);
    ArrayXXd u_z(n,n_z);                 // Full vertical vel derivative [1/yr]
    ArrayXXd u_x(n,n_z);                 // Velocity horizontal derivative [1/yr].
    ArrayXXd strain_2d(n,n_z);           // Strain ratefrom DIVA solver.
    ArrayXXd visc_all(n,5*n_z+2);        // Output ice viscosity function [Pa·s]. (n,n_z+2)
    ArrayXXd visc(n,n_z);                // Ice viscosity [Pa·s]. 
    ArrayXXd theta(n,n_z);               // Temperature field [K].
    ArrayXXd A_theta(n,n_z);             // Temperature dependent ice rate factor [Pa^-3 yr^-1]
    ArrayXXd fric_all(n,4);              // Basal friction output.(n,n_z+3)
    ArrayXXd lmbd(n,n_z);                // Matrix with stress vertical derivatives d(visc du/dz)/dz. 
    
    // Function outputs.
    Array2d L_out;                    // Grounding line function output.
    Array2d dt_out;                   // Time step function output.

    // Normalised horizontal dimension.
    // Try an unevenly-spaced horizontal grid to allows for fewer point whilst keeping
    // high resolution at the grounding line.
    ArrayXd sigma = ArrayXd::LinSpaced(n, 0.0, 1.0);      // Dimensionless x-coordinates.
    ArrayXd ds(n);
    ArrayXd ds_inv(n);
    ArrayXd ds_sym(n);
    
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

    //////////////////////////////////////////////////////////////////////////
    // Uneven spacing for staggered u-grid.
    ArrayXd ds_u(n);
    ArrayXd ds_u_inv(n);
    ArrayXd ds_u_sym(n);

    for (int i=0; i<n-2; i++)
    {
        //  0.5 * ( sigma(i+2) + sigma(i+1) ) - 0.5 * ( sigma(i+1) + sigma(i) )
        ds_u(i) = 0.5 * ( sigma(i+2) - sigma(i) );
    }
    ds_u(n-2) = sigma(n-1) - sigma(n-2);
    ds_u(n-1) = ds_u(n-2);

    ds_u_inv = 1.0 / ds_u;

    // For symmetric finite differences schemes we sum two consecutive grid spacings.
    for (int i=1; i<n; i++)
    {
        ds_u_sym(i) = ds_u(i) + ds_u(i-1);
    }
    //////////////////////////////////////////////////////////////////////////

    

    // For symmetric finite differences schemes we sum two consecutive grid spacings.
    for (int i=1; i<n; i++)
    {
        ds_sym(i) = ds(i) + ds(i-1);
    }


    // Time steps in which the solution is saved. 
    ArrayXd a    = ArrayXd::LinSpaced(t_n, t0, tf);       // Array with output time frames.
    ArrayXd a_hr = ArrayXd::LinSpaced(int(tf), t0, tf);   // Array with high resolution time frames.    

    // EXPERIMENT. Christian et al (2022): 7.0e6
    // Constant friction coeff. 7.624e6, 7.0e6 [Pa m^-1/3 s^1/3]
    C_ref = ArrayXd::Constant(n, nixParams.fric.C_ref_0);    // [Pa m^-1/3 yr^1/3] 7.0e6, MISMIP: 7.624e6

    // We assume a constant viscosity in the first iteration. 1.0e8 Pa yr.
    visc     = ArrayXXd::Constant(n, n_z, visc_0);            // [Pa yr]
    visc_bar = ArrayXd::Constant(n, visc_0);


    // Implicit initialization.
    ub          = ArrayXd::Constant(n, u_0);               // [m / yr] 
    u_bar       = ArrayXd::Constant(n, u_0);               // [m / yr]
    u_bar_old_2 = ArrayXd::Constant(n, u_0); 
    u           = ArrayXXd::Constant(n, n_z, u_0);         // [m / yr]
    beta        = ArrayXd::Constant(n, beta_0);             // [Pa yr / m]
    tau_b       = beta * ub;

    // Thermodynamical initial conditions (-25ºC).
    theta  = ArrayXXd::Constant(n, n_z, theta_0);
    b_melt = ArrayXd::Zero(n);

    // Intialize ice thickness and SMB.
    H = ArrayXd::Constant(n, H_0); // 10.0
    S = ArrayXd::Constant(n, S_0); // 0.3

    // Melting for oceanic interaction is initialised to zero.
    M = 0.0;

    // Initilize vertical discretization.
    ArrayXd dz(n);                                        // Vertical discretization (only x-dependecy for now).                                  
    dz = H / n_z;

    // Initialize vertical velocity (only x-dependency).
    // USE MATRIX FROM FLOW INCOMPRESSIBILITY.
    ArrayXXd w(n,n_z);    

    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    // MISMIP EXPERIMENTS 1-2 FORCING.
    if ( exp == "mismip_1" )
    {
        // Exps 1-2 forcing.
        // Rate factor [Pa^-3 s^-1].
        A_s << 4.6416e-24, 2.1544e-24, 1.0e-24, 4.6416e-25, 2.1544e-25, 1.0e-25,
                4.6416e-26, 2.1544e-26, 1.0e-26,
                2.1544e-26, 4.6416e-26, 1.0e-25, 2.1544e-25, 4.6416e-25, 1.0e-24,
                2.1544e-24, 4.6416e-24;

        // Time length for a certain A value. 
        t_s << 3.0e4, 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4,
               30.0e4, 33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4;
    
        // Unit conversion: [Pa^-3 s^-1] --> [Pa^-3 yr^-1].
        A_s = A_s * nixParams.cnst.sec_year;     

        // Initialization.
        A       = A_s(0);
        A_theta = ArrayXXd::Constant(n, n_z, A);
    }

    // MISMIP EXPERIMENT 3 FORCING.
    else if ( exp == "mismip_3" )
    {
        // Exps 3 forcing.
        // Rate factor [Pa^-3 s^-1].
        // Exps 3 hysteresis forcing.
        /*
        A_s << 3.0e-25, 2.5e-25, 2.0e-25, 1.5e-25, 1.0e-25, 5.0e-26, 2.5e-26, 
               5.0e-26, 1.0e-25, 1.5e-25, 2.0e-25, 2.5e-25, 3.0e-25; 

        t_s << 3.0e4, 4.5e4, 6.0e4, 7.5e4, 9.0e4, 12.0e4, 15.0e4, 16.5e4, 18.5e4,
                21.5e4, 24.5e4, 27.5e4, 29.0e4;
        */

        // Time length for a certain A value.
        A_s << 5.0e-25, 4.0e-25, 3.0e-25, 2.5e-25, 2.0e-25, 1.5e-25, 1.0e-25, 5.0e-26,
                2.5e-26, 1.2e-26,
               5.0e-26, 1.0e-25, 1.5e-25, 2.0e-25, 2.5e-25, 3.0e-25, 4.0e-25, 5.0e-25;

        t_s << 3.0e4, 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4,
               30.0e4, 33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4;
        
    
        // Unit conversion: [Pa^-3 s^-1] --> [Pa^-3 yr^-1].
        A_s = A_s * nixParams.cnst.sec_year;     

        // Initialization.
        A       = A_s(0);
        A_theta = ArrayXXd::Constant(n, n_z, A);
    }

    // MISMIP EXPERIMENT 3 WITH OCEANIC FORCING AND NO THERMODYNAMICS.
    else if ( exp == "mismip_3_A" )
    {
        // Full simulation. 44double
        /*t_s <<  6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4, 33.0e4,
               36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4, 57.0e4, 60.0e4, 63.0e4,
               66.0e4, 69.0e4, 72.0e4, 75.0e4, 78.0e4, 81.0e4, 84.0e4, 87.0e4, 90.0e4, 93.0e4,
               96.0e4, 99.0e4, 102.0e4, 105.0e4, 108.0e4, 111.0e4, 114.0e4, 117.0e4, 120.0e4, 123.0e4, 
               126.0e4, 129.0e4;

        T_oce_s << 273.15, 273.65, 274.15, 274.65, 275.15, 275.65, 276.15, 276.65, 277.15, 277.65,
                   278.15, 278.65, 279.15, 279.65, 280.15, 280.65, 281.15, 281.65, 282.15, 282.65,
                   283.15, 283.15, 282.65, 282.15, 281.65, 281.15, 280.65, 280.15, 279.65, 279.15, 278.68,
                   278.15, 277.65, 277.15, 276.65, 276.15, 275.65, 275.15, 274.65, 274.15, 273.65, 
                   273.15;
                   */

        // Initial and step length of forcing.
        double dt_forcing = 4.0e4;
        double t_forc_0   = 10.0e4;

        t_s = ArrayXd::LinSpaced(n_s, t_forc_0, t_forc_0 + double(dt_forcing*n_s));

        cout << "\n t_s = " << t_s;

        // Oceanic forcing.
        double T_oce_min = 273.15;
        double T_oce_max = 283.15;

        int n_s_hlf = int(0.5*n_s);

        T_oce_s.block(0,0,n_s_hlf,1)       = ArrayXd::LinSpaced(n_s_hlf, T_oce_min, T_oce_max);
        T_oce_s.block(n_s_hlf,0,n_s_hlf,1) = ArrayXd::LinSpaced(n_s_hlf, T_oce_max, T_oce_min);
        
        
        // Short simulation. 23
        /*t_s << 4.0e4, 6.0e4, 9.0e4, 
               12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4, 33.0e4, 36.0e4,
               39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4, 57.0e4, 60.0e4, 63.0e4,
               66.0e4, 69.0e4;
        
        T_oce_s << 273.15, 273.15, 273.15, 
                   274.15, 275.15, 276.15, 277.15, 278.15, 279.15, 280.15, 281.15, 282.15, 
                   283.15, 282.15, 281.15, 280.15, 279.15, 278.15, 277.15, 276.15, 275.15, 
                   274.15, 273.15;*/

        // Short test.
        /*t_s     << 4.0e4, 6.0e4, 8.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4;
        T_oce_s << 273.15, 273.65, 274.15, 274.65, 275.15, 275.65, 276.15, 276.65, 277.15, 277.65;*/


        // Unit conversion: [Pa^-3 s^-1] --> [Pa^-3 yr^-1].
        A_s = ArrayXd::Constant(n_s, 0.8e-26); // 1.0e-26
        A_s = A_s * nixParams.cnst.sec_year; 


        // This should not be necessary as thermodynamics is off.
        T_air_s = ArrayXd::Constant(n_s, 188.15); // 193.15

        // Initialization.
        T_air   = T_air_s(0);
        T_oce   = T_oce_s(0);
        A       = A_s(0);
        A_theta = ArrayXXd::Constant(n, n_z, A);
    }
    
    // MISMIP THERMODYNAMICS. AIR TEMPERATURES FORCING
    else if ( exp == "therm_T_air" )
    {
        // AIR TEMPERATURES FORCING.
        // Stable forcing.
        //t_s << 3.0e4, 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4,
        //       30.0e4, 33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4;

        //T_air_s << 253.15, 243.15, 233.15, 223.15, 213.15, 203.15, 198.15, 193.15, 193.15,
        //           198.15, 203.15, 213.15, 223.15, 233.15, 243.15, 253.15, 253.15;

        
        /*t_s << 6.0e4, 10.0e4, 14.0e4, 18.0e4, 22.0e4, 26.0e4, 30.0e4, 34.0e4;
        T_air_s << 263.15, 253.15, 243.15, 233.15, 243.15, 253.15, 263.15, 273.15;*/


        // Short.
        /*t_s << 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 
                27.0e4, 30.0e4, 33.0e4, 36.0e4, 39.0e4, 42.0e4;
        
        T_air_s << 273.15, 268.15, 263.15, 258.15, 253.15, 248.15, 243.15, 
                    248.15, 253.15, 258.15, 263.15, 268.15, 273.15;*/

        
        
        // High resolution.
        /*t_s << 6.0e4, 10.0e4, 14.0e4, 16.0e4, 20.0e4, 24.0e4, 28.0e4, 32.0e4, 36.0e4, 40.0e4,
               44.0e4, 48.0e4, 52.0e4, 56.0e4, 60.0e4, 64.0e4, 68.0e4, 72.0e4, 76.0e4, 80.0e4,
               84.0e4, 88.0e4, 92.0e4, 96.0e4, 100.0e4, 104.0e4;*/

        
        double dt_forcing = 4.0e4;
        double t_forc_0   = 10.0e4;

        double T_air_min = 243.15;
        double T_air_max = 273.15;

        t_s = ArrayXd::LinSpaced(n_s, t_forc_0, t_forc_0 + double(dt_forcing*n_s));

        int n_s_hlf = int(0.5*n_s);
        
        T_air_s.block(0,0,n_s_hlf,1)       = ArrayXd::LinSpaced(n_s_hlf, T_air_max, T_air_min);
        T_air_s.block(n_s_hlf,0,n_s_hlf,1) = ArrayXd::LinSpaced(n_s_hlf, T_air_min, T_air_max);
        
        
        /*T_air_s << 273.15, 268.15, 263.15, 258.15, 253.15, 248.15, 243.15, 
                    248.15, 253.15, 258.15, 263.15, 268.15, 273.15;*/
        
        /*T_air_s << 253.15, 248.15, 243.15, 238.15, 233.15, 228.15, 223.15, 218.15, 213.15, 208.15,
                   203.15, 198.15, 193.15, 193.15, 198.15, 203.15, 208.15, 213.15, 218.15, 223.15,
                   228.15, 233.15, 238.15, 243.15, 248.15, 253.15;*/
        
        

        // Initialization.
        T_air   = T_air_s(0);
        A       = A_s(0);
        A_theta = ArrayXXd::Constant(n, n_z, A);

    }

    // MISMIP THERMODYNAMICS. OCEAN TEMPERATURES FORCING.
    else if ( exp == "therm_T_oce" )
    {
        // OCEAN TEMPERATURES ANOMALIES FORCING.
        // Change of sign in (T_0-T_oce) to produce advance/retreate.
        // Make sure length of positive/negative anomalies is the same
        // to retireve the initial state. Close hysteresis loop.
        /*t_s <<  6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4, 33.0e4,
               36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4, 57.0e4, 60.0e4, 63.0e4,
               66.0e4, 69.0e4, 72.0e4, 75.0e4, 78.0e4, 81.0e4, 84.0e4, 87.0e4, 90.0e4, 93.0e4,
               96.0e4, 99.0e4, 102.0e4, 105.0e4, 108.0e4, 111.0e4, 114.0e4, 117.0e4, 120.0e4, 123.0e4, 
               126.0e4, 129.0e4;
        
        T_oce_s << 273.15, 273.65, 274.15, 274.65, 275.15, 275.65, 276.15, 276.65, 277.15, 277.65,
                   278.15, 278.65, 279.15, 279.65, 280.15, 280.65, 281.15, 281.65, 282.15, 282.65,
                   283.15, 283.15, 282.65, 282.15, 281.65, 281.15, 280.65, 280.15, 279.65, 279.15, 278.68,
                   278.15, 277.65, 277.15, 276.65, 276.15, 275.65, 275.15, 274.65, 274.15, 273.65, 
                   273.15;*/


        double dt_forcing = 4.0e4;
        double t_forc_0   = 10.0e4;

        double T_oce_min = 273.15;
        double T_oce_max = 283.15;

        //t_s = ArrayXd::LinSpaced(n_s, 6.0e4, double(dt_forcing*n_s));
        //t_s.block(4,0,n_s-4,1) = ArrayXd::LinSpaced(n_s-4, 10.0e4, double(dt_forcing*n_s));
        t_s.block(4,0,n_s-4,1) = ArrayXd::LinSpaced(n_s-4, t_forc_0, t_forc_0 + double(dt_forcing*(n_s-4)));

        t_s(0) = 3.0e4;
        t_s(1) = 4.0e4;
        t_s(2) = 5.0e4;
        t_s(3) = 6.0e4;

        int n_s_hlf = int(0.5*(n_s-4));
        
        T_oce_s.block(4,0,n_s_hlf,1)         = ArrayXd::LinSpaced(n_s_hlf, T_oce_min, T_oce_max);
        T_oce_s.block(4+n_s_hlf,0,n_s_hlf,1) = ArrayXd::LinSpaced(n_s_hlf, T_oce_max, T_oce_min);
        T_oce_s(0) = 273.15;
        T_oce_s(1) = 273.15;
        T_oce_s(2) = 273.15;
        T_oce_s(3) = 273.15;
        
        
        // Short simulation. 23
        /*t_s << 4.0e4, 6.0e4, 
               9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4, 33.0e4, 
               36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4, 57.0e4, 60.0e4, 
               63.0e4, 66.0e4;
        
        T_oce_s << 273.15, 273.15, 
                   274.15, 275.15, 276.15, 277.15, 278.15, 279.15, 280.15, 281.15, 282.15, 
                   283.15, 282.15, 281.15, 280.15, 279.15, 278.15, 277.15, 276.15, 275.15, 
                   274.15, 273.15;*/
        
        // Constant value given in the param file but smooth transition to the cold value.
        // CHECK IF THIS VALUE OF T_AIR IS ENOUGH TO ADVANCE AS MUCH AS THE NON-THERMAL SIM.
        T_air_s    = ArrayXd::Constant(n_s, nixParams.bc.therm.T_air); // 193.15
        T_air_s(0) = 253.15; // Equilibration with a medium value!! 253.15
        T_air_s(1) = 258.15; 
        T_air_s(2) = 253.15; 
        T_air_s(3) = 243.15; 

        // Initialization.
        T_air   = T_air_s(0);
        T_oce   = T_oce_s(0);
        A       = A_s(0);
        A_theta = ArrayXXd::Constant(n, n_z, A);
    }

    // TRANSITION INDICATORS EXPERIMENTS. OCEAN TEMPERATURES FORCING (T_air const).
    else if ( exp == "ews_schoof" )
    {
        // OCEAN TEMPERATURES ANOMALIES FORCING.
        // Beggining and ending points.
        T_oce_s << 273.15, 283.15;  // 283.15 is probably too high for the transition indicators (overshooting).
        T_air_s << 253.15, 233.15; // 253.15, 233.15


        // Test no therm.
        //A_s = ArrayXd::Constant(n_s, 5.0e-25);
        //A_s = A_s * nixParams.cnst.sec_year;  

        // Initialization.
        T_air   = T_air_s(0);
        T_oce   = T_oce_s(0);
        A       = A_s(0);
        A_theta = ArrayXXd::Constant(n, n_z, A);
    }
    
    // TRANSITION INDICATORS EXPERIMENTS.
    else if ( exp == "ews_christian" )
    {
        // WE NEED TO TUNE THIS NUMBER TOGEHTER WITH THE FLUX DISCRETIAZTION TO OBTAIN THE SAME EXTENT.
        //A_s << 0.5e-26, 5.0e-25; //(0.5e-26, 5.0e-25)
        // Decrease A_s values for all peaks to be reduced.
        A_s << 3.0e-26, 5.0e-25; //(0.5e-26, 5.0e-25)


        // Unit conversion: [Pa^-3 s^-1] --> [Pa^-3 yr^-1].
        A_s = A_s * nixParams.cnst.sec_year;     

        // FORCING CAN BE IMPOSED DIRECTLY ON A_s (i.e., an increase in temperature) or
        // on the calving at the front as a consequence of ocean warming.
    }

    // EXPERIMENT NOT DEFINED.
    else
    {
        cout << "\n Experiment not defined. Please, do so: mismip_1...";
    }

    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    // Print spatial and time dimensions.
    cout << " \n Nix ice-shet model v1.0";
    cout << " \n Experiment = " << exp;
    cout << " \n n          = " << n;
    cout << " \n n_sigma    = " << nixParams.dom.grid_exp;
    cout << " \n tf         = " << tf;

    // Call nc read function.
    if ( nixParams.stoch.stoch == true )
    {
        cout << "\n Stochastic data successfully read from = " << nixParams.path.read;
        noise = f_nc_read(nixParams.stoch.N, nixParams.path.read);
    }
    

    // Call nc write functions.
    f_nc(n, n_z, nixParams.path.out);
    f_nc_hr(n, n_z, nixParams.path.out);


    // Wall time for computational speed.
    auto begin_0 = std::chrono::high_resolution_clock::now();
    auto begin = begin_0;

    // Counters to write solution.
    int c    = 0;
    int c_hr = 0;

    // Counter for MISMIP ice factor A forcing.
    int c_s = 0;

    // Initialize time.
    t  = t0;
    dt = nixParams.tmstep.dt_min;


    // TIME INTEGRATION.
    while (t < tf)
    {

        // STOCHASTIC TIME-DEPENDENT BOUNDARY CONDITION. 
        // Update time-dependent boundary conditions after equilibration.
        if ( nixParams.stoch.stoch == true )
        {
            if ( t < nixParams.stoch.t0 )
            {
                m_stoch     = 0.0;
                smb_stoch   = 0.0;
                T_oce_stoch = 0.0;
            }
            else
            {
                // Christian's spin-up also considers stochastic anomalies?
                // Lower bound of zero in m to avoid numerical issues.
                // Start counting time when stoch is applied.
                t_stoch = floor(max(0.0, t - nixParams.stoch.t0));
                        
                noise_now = noise.col(t_stoch);

                //m_stoch   = max(0.0, noise_now(0)); 
                m_stoch     = noise_now(0); 
                smb_stoch   = noise_now(1);
                T_oce_stoch = noise_now(2);

                // Update SMB considering new domain extension and current stochastic term.
                //S = f_smb(sigma, L, t, smb_stoch, nixParams.bc, \
                //            nixParams.dom, nixParams.tm);
            }
        }
        

        // MISMIP EXPERIMENTS 1, 3 and 3.
        if ( exp == "mismip_1" || exp == "mismip_3" )
        {
            // Update rate factor value.
            if ( t > t_s(c_s) )
            {
                c_s = min(c_s+1, n_s-1);
            }
            
            // Ice hardness.
            A = A_s(c_s);
            B = pow(A, (-1 / nixParams.vis.n_gln) );

            // OSCILLATIONS.
            //A = A_s(0);
            //B = pow(A, (-1 / nixParams.vis.n_gln) );

        }

        // MISMIP-THERM EXPERIMENTS.
        else if ( exp == "mismip_3_A" || exp == "therm_T_air" || exp == "therm_T_oce" )
        {
            // Update rate factor and T_air value.
            if ( t > t_s(c_s) )
            {
                c_s = min(c_s+1, n_s-1);

                // Ice hardness.
                A = A_s(c_s);
                B = pow(A, (-1 / nixParams.vis.n_gln) );

                // Air temperature as a BC.
                T_air = T_air_s(c_s);

                // Ocean temperature as a BC.
                T_oce = T_oce_s(c_s);
                //cout << " \n T_oce = " << T_oce;

                // Directly on frontal ablation.
                M = M_s(c_s);
            }

        }

        // TRANSITION INDICATORS EXPERIMENTS.
        else if ( exp == "ews_schoof" || exp == "ews_christian" )
        {
            // Constant A throughout the sim.
            if ( nixParams.bc.trend.type == "none" )
            {
                A = A_s(0);
                
                // Ensure positive values of frontal ablation as Christian et al. (2022).
                m_stoch = max(0.0, m_stoch);
            }

            // Oceanic temperature anomalies forcing.
            else if ( nixParams.bc.trend.type == "T_oce" )
            {
                // Gradually decrease T_air to advance ice sheet (no oceanic forcing).
                if ( t < 0.5*nixParams.bc.trend.t0 )
                {
                    // Gradual decrease in T_air for equilibration.
                    T_air = T_air_s(0) + ( T_air_s(1) - T_air_s(0) ) \
                                * ( 2.0 * t / nixParams.bc.trend.t0 );
                    
                    T_oce_det = T_oce_s(0);
                }

                // Equilibration with cold T_air (no oceanic forcing).
                else if ( t >= 0.5*nixParams.bc.trend.t0 && t < nixParams.bc.trend.t0 )
                {
                    // Fixed value of air temperature.
                    T_air     = T_air_s(1);
                    T_oce_det = T_oce_s(0);
                }
                
                // Start oceanic forcing while keeping a constant air temperature.
                else if ( t >= nixParams.bc.trend.t0 && t <= nixParams.bc.trend.tf )
                {
                    // Fixed value of air temperature.
                    T_air = T_air_s(1);

                    // OCEANIC FORCING. Gradually increasing temperature anomaly.
                    T_oce_det = T_oce_s(0) + ( T_oce_s(1) - T_oce_s(0) ) * (t - nixParams.bc.trend.t0) \
                                / (nixParams.bc.trend.tf - nixParams.bc.trend.t0);
                }

                else if ( t > nixParams.bc.trend.tf )
                {
                    // Fixed value of air temperature.
                    T_air     = T_air_s(1);
                    T_oce_det = T_oce_s(1);
                }

                // Total T_oce is the deterministic forcing plus the stochastic contribution.
                T_oce = T_oce_det + T_oce_stoch;
                T_oce = max(0.0, T_oce);
            }

            // Forcing in A.
            else if ( nixParams.bc.trend.type == "rate_factor" )
            {
                // Equilibration with constant A.
                if ( t < nixParams.bc.trend.t0 )
                {
                    A = A_s(0);
                }
                
                else if ( t >= nixParams.bc.trend.t0 && t <= nixParams.bc.trend.tf )
                {
                    A = A_s(0) + ( A_s(1) - A_s(0) ) * (t - nixParams.bc.trend.t0) \
                                / (nixParams.bc.trend.tf - nixParams.bc.trend.t0);
                }

                else if ( t > nixParams.bc.trend.tf )
                {
                    A = A_s(1);
                }

                // Ensure positive values of frontal ablation as Christian et al. (2022).
                m_stoch = max(0.0, m_stoch);
            }

            // Forcing in frontal ablation m_stoch. (Christian et al., 2022)
            else if ( nixParams.bc.trend.type == "ablation" )
            {
                // Temperature is fixed for now via a constant ice rate factor.
                A = A_s(0);

                // We need to add a value to the stochastic perturbation. The added value
                // corresponds to a linear increase with a maximum value of 80% of the mean original
                // melting, i.e. M_0 = 30 m/yr. At t=tf, we will have the perturbation plus 24 m/yr.
                
                // Increase from trend is only applied to non-zero perturbations (Fig. 5a in Christian et al., 2022).
                if ( t >= nixParams.bc.trend.t0 && t < nixParams.bc.trend.tf )
                {
                    // The trend increases linearly with time and it is an additional fraction M_f 
                    // of the original mean frontal ablation M_0 (Fig. 5, Christian et al., 2022).
                    alpha   = (t - nixParams.bc.trend.t0) / (nixParams.bc.trend.tf - nixParams.bc.trend.t0);
                    m_stoch = m_stoch + alpha * nixParams.bc.trend.M_f * nixParams.bc.trend.M_0;
                }

                else if ( t >= nixParams.bc.trend.tf )
                {
                    m_stoch = m_stoch + nixParams.bc.trend.M_f * nixParams.bc.trend.M_0;
                }

                // Ensure positive values of frontal ablation as Christian et al. (2022).
                m_stoch = max(0.0, m_stoch);
            }


            // Ice hardness.
            B = pow(A, ( -1 / nixParams.vis.n_gln ) );
        }
    

        // Update SMB considering new domain extension and current stochastic term.
        S = f_smb(sigma, L, t, smb_stoch, \
                    nixParams.bc, nixParams.dom, nixParams.tm);

        // Update bedrock with new domain extension L.
        bed = f_bed(L, sigma, ds, t, nixParams.dom);

        // Friction coefficient.
        C_bed = f_C_bed(C_ref, theta, H, t, nixParams.dom, \
                            nixParams.cnst, nixParams.tm, nixParams.fric);


        
        // UNDER DEVELOPMENT.
        // Homothopy ontinuation in viscosity.
        /*double eps;
        double eps_old;
        double rel = 0.1; // 0.25
        double eps_min = 1.0e-7; // 1.0e-7
        double eps_max = 1.0e-4; // 1.0e-5


        int c_eps = 0;
        //error    = 1.0;
        while ( c_eps < 1 )
        {
            // Update eps for Blatter-Pattyn.
            eps_old = eps;
            if ( t < 1.0 * nixParams.tm.t_eq )
            {
                eps = nixParams.vis.eps;
            }
            else
            {
                eps = ( 1.0 - pow( min(error, nixParams.pcrd.tol)/nixParams.pcrd.tol, 2) ) * \
                             ( eps_min - eps_max ) + eps_max;

                eps = rel * eps_old + (1.0 - rel) * eps;
            }
            
            // Picard initialization.
            error    = 1.0;
            c_picard = 0;
            
            // Implicit velocity solver. Picard iteration for non-linear viscosity and beta.
            // Loop over the vertical level for Blatter-Pattyn.
            while ( error > nixParams.pcrd.tol && c_picard < nixParams.pcrd.n_picard )
            {
                // Save previous iteration solution.
                u_bar_old_1 = u_bar;
                u_old       = u;
                
                // Implicit solver.
                // If SSA solver ub = u_bar.
                sol = vel_solver(H, ds, ds_inv, ds_u_inv, dz, visc_bar, bed, L, \
                                    C_bed, t, beta, A, A_theta, visc, u, u_z, \
                                        nixParams.dyn, nixParams.dom, \
                                            nixParams.cnst, nixParams.vis);
                
                // Allocate variables. sol(n+1,n_z+1)
                u_bar  = sol.block(0,0,n,1);
                u      = sol.block(0,1,n,n_z);

                
                // Update beta with new velocity.
                fric_all = f_u(u, u_bar, beta, C_bed, visc, H, dz, t, \
                                    nixParams.dom, nixParams.dyn, \
                                        nixParams.fric, nixParams.cnst);
                beta = fric_all.col(0);

                // Update viscosity with new velocity.
                visc_all = f_visc(theta, u, visc, H, tau_b, u_bar, dz, \
                                    ds, ds_u, ds_sym, L, t, A, eps, \
                                        nixParams.dom, nixParams.thrmdyn, \
                                            nixParams.vis, nixParams.tm, \
                                                nixParams.dyn, nixParams.init);


                // Allocate variables.
                //ArrayXXd visc_old = visc;
                visc     = visc_all.block(0,0,n,n_z);
                visc_bar = visc_all.col(n_z);
                
                
                // Current error (vector class required to compute norm). 
                // Eq. 12 (De-Smedt et al., 2010).
                c_u_bar_1 = u_bar - u_bar_old_1;
                u_bar_vec = u_bar;
                error     = c_u_bar_1.norm() / u_bar_vec.norm();
                
                // New relaxed Picard iteration. Pattyn (2003). 
                // Necessary to deal with the nonlinear velocity dependence on both viscosity and beta.
                // Just update beta and visc, not tau_b!
                // We have previously intialize u_bar_old_2 for t = 0.
                // Difference between iter (i-1) and (i-2).
                c_u_bar_2 = u_bar_old_1 - u_bar_old_2;
                
                // Angle defined between two consecutive vel solutions.
                omega = acos( c_u_bar_1.dot(c_u_bar_2) / \
                                ( c_u_bar_1.norm() * c_u_bar_2.norm() ) );
                

                // De Smedt et al. (2010). Eq. 10.
                if (omega <= omega_1 || c_u_bar_1.norm() == 0.0)
                {
                    mu = 2.5; // De Smedt.
                    //mu = 1.0; // To avoid negative velocities?
                    //mu = 0.7; // Daniel
                }
                else if (omega > omega_1 & omega < omega_2)
                {
                    mu = 1.0; // De Smedt.
                    //mu = 0.7; // Daniel
                }
                else
                {
                    mu = 0.7; // De Smedt.
                    //mu = 0.5; // Daniel
                }

                
                // New velocity guess based on updated mu.
                u_bar = u_bar_old_1 + mu * c_u_bar_1.array();
                u     = u_old + mu * ( u - u_old ); 

                //double rel = 0.95; // 0.8 works extremely well.
                //visc = visc_old * rel + ( 1.0 - rel ) * visc; 
                //visc     = visc_old + mu * ( visc - visc_old ); 
                
                // Update multistep variables.
                u_bar_old_2 = u_bar_old_1;

                // Update number of iterations.
                ++c_picard;
            }
        
            ++c_eps;
        }*/




        // Implicit velocity solver. Picard iteration for non-linear viscosity and beta.
        // Loop over the vertical level for Blatter-Pattyn.
        // Picard initialization.
        error    = 1.0;
        c_picard = 0;

        ArrayXd beta_old(n);
        
        while ( error > nixParams.pcrd.tol && c_picard < nixParams.pcrd.n_picard )
        {
            // Save previous iteration solution.
            u_bar_old_1 = u_bar;
            u_old       = u;

            beta_old = beta;
            
            // Implicit solver.
            // If SSA solver ub = u_bar.
            sol = vel_solver(H, ds, ds_inv, ds_u_inv, dz, visc_bar, bed, L, \
                                C_bed, t, beta, A, A_theta, visc, u, u_z, \
                                    nixParams.dyn, nixParams.dom, \
                                        nixParams.cnst, nixParams.vis);
            
            // Allocate variables. sol(n+1,n_z+1)
            u_bar  = sol.block(0,0,n,1);
            u      = sol.block(0,1,n,n_z);

            
            // Update beta with new velocity.
            fric_all = f_u(u, u_bar, beta, C_bed, visc, H, dz, t, \
                                nixParams.dom, nixParams.dyn, \
                                    nixParams.fric, nixParams.cnst);
            beta = fric_all.col(0);

            // Update viscosity with new velocity.
            visc_all = f_visc(theta, u, visc, H, tau_b, u_bar, dz, \
                                ds, ds_u, ds_sym, L, t, A, \
                                    nixParams.dom, nixParams.thrmdyn, \
                                        nixParams.vis, nixParams.tm, \
                                            nixParams.dyn, nixParams.init);

            // Allocate variables.
            visc     = visc_all.block(0,0,n,n_z);
            visc_bar = visc_all.col(n_z);

            // Execute functions in parallel using OpenMP
            /*#pragma omp parallel sections
            {
                #pragma omp section
                {
                    // Implicit solver.
                    // If SSA solver ub = u_bar.
                    sol = vel_solver(H, ds, ds_inv, ds_u_inv, dz, visc_bar, bed, L, \
                                        C_bed, t, beta, A, A_theta, visc, u, u_z, \
                                            nixParams.dyn, nixParams.dom, \
                                                nixParams.cnst, nixParams.vis);
                    
                    // Allocate variables. sol(n+1,n_z+1)
                    u_bar  = sol.block(0,0,n,1);
                    u      = sol.block(0,1,n,n_z);

                    // Update beta with new velocity.
                    fric_all = f_u(u, u_bar, beta_old, C_bed, visc, H, dz, t, \
                                        nixParams.dom, nixParams.dyn, \
                                            nixParams.fric, nixParams.cnst);
                    beta = fric_all.col(0);
                    
                }


                #pragma omp section
                {
                    // Update viscosity with new velocity.
                    visc_all = f_visc(theta, u, visc, H, tau_b, u_bar, dz, \
                                        ds, ds_u, ds_sym, L, t, A, \
                                            nixParams.dom, nixParams.thrmdyn, \
                                                nixParams.vis, nixParams.tm, \
                                                    nixParams.dyn, nixParams.init);

                    // Allocate variables.
                    visc     = visc_all.block(0,0,n,n_z);
                    visc_bar = visc_all.col(n_z);
                }

            }*/

            /*cout << "\n c_picard =  " << c_picard;
            cout << "\n u_bar =  " << u_bar;
            cout << "\n beta =  " << beta;
            cout << "\n visc_bar =  " << visc_bar;*/

            
            
            // Current error (vector class required to compute norm). 
            // Eq. 12 (De-Smedt et al., 2010).
            c_u_bar_1 = u_bar - u_bar_old_1;
            u_bar_vec = u_bar;
            error     = c_u_bar_1.norm() / u_bar_vec.norm();
            
            // New relaxed Picard iteration. Pattyn (2003). 
            // Necessary to deal with the nonlinear velocity dependence on both viscosity and beta.
            // Just update beta and visc, not tau_b!
            // We have previously intialize u_bar_old_2 for t = 0.
            // Difference between iter (i-1) and (i-2).
            c_u_bar_2 = u_bar_old_1 - u_bar_old_2;
            
            // Angle defined between two consecutive vel solutions.
            omega = acos( c_u_bar_1.dot(c_u_bar_2) / \
                            ( c_u_bar_1.norm() * c_u_bar_2.norm() ) );
            

            // De Smedt et al. (2010). Eq. 10.
            if (omega <= omega_1 || c_u_bar_1.norm() == 0.0)
            {
                mu = 2.5; // De Smedt.
                //mu = 1.0; // To avoid negative velocities?
                //mu = 0.7; // Daniel
            }
            else if (omega > omega_1 & omega < omega_2)
            {
                mu = 1.0; // De Smedt.
                //mu = 0.7; // Daniel
            }
            else
            {
                mu = 0.7; // De Smedt.
                //mu = 0.5; // Daniel
            }

            
            // New velocity guess based on updated mu.
            u_bar = u_bar_old_1 + mu * c_u_bar_1.array();
            u     = u_old + mu * ( u - u_old ); 
            
            // Update multistep variables.
            u_bar_old_2 = u_bar_old_1;

            // Update number of iterations.
            ++c_picard;
        }
        
        

        
        // Allocate variables from converged solution.
        tau_b     = fric_all.col(1);
        Q_fric    = fric_all.col(2);
        ub        = fric_all.col(3);
        u_bar_x   = visc_all.col(n_z+1);
        u_x       = visc_all.block(0,n_z+2,n,n_z);
        u_z       = visc_all.block(0,2*n_z+2,n,n_z);
        strain_2d = visc_all.block(0,3*n_z+2,n,n_z);
        A_theta   = visc_all.block(0,4*n_z+2,n,n_z);


        // CONSISTENCY CHECK. Search for NaN values.
        // Count number of true positions in u_bar.isnan().    
        if ( u_bar.isNaN().count() != 0 )
        {
            cout << "\n NaN found.";
            cout << "\n Saving variables in nc file. \n ";

            // Save previous iteration solution (before NaN encountered).
            f_write(c, u_bar_old_1, ub, u_bar_x, H, visc_bar, S, tau_b, beta, tau_d, bed, \
                    C_bed, Q_fric, u2_dif_vec, u2_0_vec, L, t, u_x_bc, u2_dif, \
                    error, dt, c_picard, mu, omega, theta, visc, u_z, u_x, u, w, A, dL_dt, \
                    F_1, F_2, m_stoch, smb_stoch, A_theta, T_oce_det, T_oce_stoch, T_air, \
                    lmbd, speed);

            // Close nc file. 
            if ((retval = nc_close(ncid)))
            ERR(retval);
            printf("\n *** %s file has been successfully written \n", nixParams.path.out.c_str());
                
            // Abort flowline.
            return 0;
        }
        
        
        // Update sub-shelf melt.
        if ( nixParams.calv.sub_shelf_melt.shelf_melt == true )
        {
            M = f_melt(T_oce, nixParams.calv.sub_shelf_melt, nixParams.cnst);
        }
        
        
        
        // Ice flux calculation. Flotation thickness H_f.
        q = f_q(u_bar, H, bed, t, m_stoch, M, nixParams.dom, \
                    nixParams.cnst, nixParams.tm, nixParams.calv);
        
        // Update grounding line position with new velocity field.
        L_out = f_L(H, q, S, bed, dt, L, ds, nixParams.dom, nixParams.cnst);
        L     = L_out(0);
        dL_dt = L_out(1);

        

        // Write solution with desired output frequency.
        if ( c == 0 || t > a(c) )
        {
            // Running time (measures wall time).
            auto end     = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
            double speed = 60 * 60 * 1.0e6 * (a(1) - a(0)) / elapsed.count();

            if ( c == 0 )
            {
                speed = 0.0;
                error = 0.0;
            }

            // std::cout is typically buffered by default.
            // By using std::flush or std::endl, you ensure that the data is 
            // immediately written to the output device. 
            cout << "\n t                 = " << t << std::flush;
            cout << "\n dt                = " << dt << std::flush;
            cout << "\n dx_min            = " << 1.0e-3*L*ds(n-2) << std::flush;
            cout << "\n Picard iterations = " << c_picard << std::flush;
            //cout << "\n Visc eps:         = " << eps << std::flush;
            //cout << "\n Estimated error:     " << error << std::flush;
            
            //cout << "\n noise_now(0) = " << noise_now(0);
            //cout << " b_melt = " << b_melt;

            // Write solution in nc.
            // Sub-shelf melting as forcing of MISMIP+thermodynamics.
            //m_stoch = M;
            

            f_write(c, u_bar, ub, u_bar_x, H, visc_bar, S, tau_b, beta, tau_d, bed, \
                    C_bed, Q_fric, u2_dif_vec, u2_0_vec, L, t, u_x_bc, u2_dif, \
                    error, dt, c_picard, mu, omega, theta, visc, u_z, u_x, u, w, A, dL_dt, \
                    F_1, F_2, m_stoch, smb_stoch, A_theta, T_oce_det, T_oce_stoch, T_air, lmbd, \
                    speed);

            ++c;

            // Wall time for computational speed.
            begin = std::chrono::high_resolution_clock::now();
        }  
        

        // Write solution with high resolution output frequency.
        if ( out_hr == true && t > a_hr(c_hr) )
        {
            //cout << "\n t_hr = " << t;
            // Write solution in nc.
            f_write_hr(c_hr, u_bar(n-1), H(n-1), L, t, u_x_bc, u2_dif, \
                       error, dt, c_picard, mu, omega, A, dL_dt, m_stoch, \
                       smb_stoch, T_air, T_oce_det, T_oce_stoch);

            ++c_hr;
        }  
        

        

        // Integrate ice thickness forward in time.
        H = f_H(u_bar, H, S, sigma, dt, ds, ds_inv, ds_sym, ds_u_inv, \
                  L, D, dL_dt, bed, q, M, t, \
                    nixParams.dom, nixParams.tm, nixParams.adv);
        
        // Update vertical discretization.
        dz = H / n_z;

        

        // THERMODYNAMICS.
        // Vertical advection is the key to obtain oscillations.
        // It provides with a feedback to cool down the ice base and balance frictional heat.
        if ( nixParams.thrmdyn.therm == false || t < nixParams.tm.t_eq )
        {
            theta = ArrayXXd::Constant(n, n_z, theta_0);
        }

    
        // Obtain vertical velocities and integrate Fourier heat equation.
        else if ( nixParams.thrmdyn.therm == true && t >= nixParams.tm.t_eq )
        {
            // Vertical velocity from incompressibility of ice flow.
            w = f_w(u_bar_x, H, dz, b_melt, u_bar, bed, ds, L, nixParams.dom);

            // Integrate heat equation and calculate basal melt.
            sol_thrm = f_theta(theta, ub, H, tau_b, Q_fric, bed, sigma, dz, \
                                dt, ds, L, T_air, dL_dt, t, w, strain_2d, \
                                    nixParams.dom, nixParams.thrmdyn, nixParams.dyn, \
                                        nixParams.bc, nixParams.cnst, nixParams.calv);

            // Allocate variables.
            theta  = sol_thrm.block(0,0,n,n_z);
            b_melt = sol_thrm.block(0,n_z,n,1);

        }


        // Update timestep and current time.
        dt_out = f_dt(L, t, dt, u_bar.maxCoeff(), ds.minCoeff(), error, \
                        nixParams.tmstep, nixParams.tm, nixParams.pcrd);
        t  = dt_out(0);
        dt = dt_out(1);
        
    }
    

    // Running time (measures wall time).
    auto end     = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin_0);
    
    // Print computational time.
    printf("\n Time measured: %.3f minutes.\n", elapsed.count() * 1e-9 / 60.0);
    //printf("\n Computational speed: %.3f kyr/hr.\n", speed);

    // Close nc files. 
    if ((retval = nc_close(ncid)))
    ERR(retval);
    if ((retval = nc_close(ncid_hr)))
    ERR(retval);
 
    printf("\n *** %s file has been successfully written \n", nixParams.path.out.c_str());
    


    return 0;
}