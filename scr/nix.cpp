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
    // USE YAML FILE TO READ PARAMETERS.
    /*string experiment = "mismip_1";
    if (experiment == "mismip_1")
    {
        cout << "\n Experiment = " << experiment;
    }*/

    // Specify the path to your YAML file
    string file_path = "/home/dmoreno/scr/nix/par/nix_params.yaml";

    // Load the YAML file
    YAML::Node config = YAML::LoadFile(file_path);

    // Access constants.
    /*double const g        = config["constants"]["g"].as<double>();
    double const rho      = config["constants"]["rho"].as<double>();
    double const rho_w    = config["constants"]["rho_w"].as<double>();
    double const sec_year = config["constants"]["sec_year"].as<double>();
    */

    // Parse parameters
    NixParams nixParams;
    readParams(config, nixParams);

    // Access parameters
    std::cout << "t0: " << nixParams.time.t0 << std::endl;
    std::cout << "g: " << nixParams.constants.g << std::endl;



    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////
    // Initialize flowline.

    // SELECT EXPERIMENT.
    // 1: Exp. 1-2 MISMIP, 3: Exp. 3 MISMIP, 4: MISMIP+THERM, 5: TRANSITION INDICATORS.
    int const exp = 3;

    // BED GEOMETRY.
    // Following Pattyn et al. (2012) the overdeepening hysterisis uses n = 250.
    // bed_exp = 1: "mismip_1", 3: "mismip_3", 4: "galcier_ews"
    int const bed_exp = 3;

    // MISMIP EXPERIMENTS FORCING.
    // Number of steps in the A forcing.
    //  Exp_3: 13 (18 long), Exp_1-2: 17. T_air: 17, T_oce: 9. T_oce_f_q: 29
    int const n_s = 18;  

    // GENERAL PARAMETERS.
    double const sec_year = 3.154e7;                // Seconds in a year.

    // PHYSICAL CONSTANTS.
    double const u_0   = 150.0 / sec_year;
    double const g     = 9.8;                      // Gravitational acceleration [m/s²].
    double const rho   = 917.0;                     // Ice density [kg/m³]. 900.0, 917.0
    double const rho_w = 1028.0;                    // Water denisity [kg/m³]. 1000.0, 1028.0
    
   
    // GROUNDING LINE. exp1 = 694.5e3, exp3 = 479.1e3
    double L = 479.1e3;                              // Grounding line position [m] exp1 = 694.5e3, exp3 = 479.1e3, ews = 50.0e3
    double dL_dt;                                   // GL migration rate [m/yr]. 
    
    // ICE VISCOSITY: visc.
    double const n_gln = 3.0;                             // Glen flow low exponent.
    double const n_exp = (1.0 - n_gln) / (2.0 * n_gln);   // De-smedt et al.
    //double const n_exp = (1.0 - n_gln) / n_gln;         // Pattyn.

    // VISCOSITY REGULARIZATION TERM.
    // eps is fundamental for GL, velocities, thickness, etc. 1.0e-10, 1.0e-5
    // Values above 1.0e-7 are unstable for the ewr domain (much smaller than MISMIP).
    // DIVA: 1.0e-6, 1.0e-5
    // For Transition Indicators:
    //double const eps = 1.0e-9; 
    // MISMIP: 1.0e-8
    double const eps = 1.0e-8; // 1.0e-6                      

    // Basal friction exponent.
    double const m = 1.0 / 3.0;                      // Friction exponent.


    // Spatial resolution.
    int const n   = 500;                             // 100. 250. Number of horizontal points 350, 500, 1000, 1500
    int const n_z = 10;                              // 10. Number vertical layers. 25 (for T_air forcing!)
    //double const ds     = 1.0 / n;                   // Normalized spatial resolution.
    //double const ds_inv = n;


    // Time variables.
    double const t0   = 0.0;                         // Starting time [yr].
    double const tf   = 1.0e3;                       // 57.0e3, 54.0, 32.0e4, MISMIP-therm: 3.0e4, 90.0e4, Ending time [yr]. EWR: 5.0e3
    double const t_eq = 1.5e3;                   // 2.0e3 Equilibration time: visc, vel, theta, etc. 0.2 * tf
    double t;                                        // Time variable [yr].

    
    // A rate forcing for Transition Indicators.
    bool const A_rate = false;                            // Boolean to apply forcing.
    double const t0_A     = 2.0e4;                   // Start time to apply increase in ice rate factor. 3.0e4
    double const tf_A     = 3.0e4;
    double const t0_stoch = 3.0e3;                   // Start time to apply stochastic BC. 2.0e4

    // VELOCITY SOLVER.
    // 0 = cte, 1 = SSA, 2 = DIVA, 3 = Blatter-Pattyn.
    //int const vel_meth = 3;                          // Vel solver choice: 
    int vel_meth = 1;

    // TIME STEPING. Quite sensitive (use fixed dt in case of doubt).
    // For stochastic perturbations. dt = 0.1 and n = 250.
    int const dt_meth = 1;                           // Time-stepping method. Fixed, 0; adapt, 1.
    double dt;                                       // Time step [yr].
    double dt_CFL;                                   // Courant-Friedrichs-Lewis condition [yr].
    double dt_tilde;                                 // New timestep. 
    double const t_eq_dt = 2.0 * t_eq;               // Eq. time until adaptative timestep is applied.
    double const dt_min = 0.1;                       // Minimum time step [yr]. 0.1
    double const dt_max = 5.0;                       // Maximum time step [yr]. 2.0, 5.0
    double const rel = 0.7;                          // Relaxation between interations [0,1]. 0.5
    
    // INPUT DEFINITIONS.   
    int const N = 50000;                             // Number of time points in BC (input from noise.nc in glacier_ews), equivalent to time length..
    double const dt_noise = 1.0;                     // Assumed time step in stochastic_anom.py 

    // OUTPUT DEFINITIONS.
    bool const out_hr = false;                       // Allow high resolution output.
    int const t_n = 100;                             // Number of output frames. 30.

    // BEDROCK
    // Glacier ews option.
    int const smooth_bed = 0;                        // Apply smooth (gaussian filter or running mean) on bed topography. 
    double const sigma_gauss = 10.0;                  // Sigma gaussian filter. 
    double const t0_gauss = 1.5e4;                   // Init time to apply gaussian filter.
    double const x_1 = 346.0e3;                      // Peak beginning [m].
    double const x_2 = 350.0e3;                      // Peak end [m]. 88.0
    double const y_p = 88.0;                         // Peak height [m]. 44.0, 88.0, 176.0
    double const y_0 = 70.0;                         // Initial bedrock elevation (x=0) [m].
    
    // SURFACE MASS BALANCE.
    bool const stoch = false;                        // Stochastic SBM.
    double const S_0      = 0.3;                     // SMB at x = 0 (and during equilibration) [m/yr]. 0.7
    double const dlta_smb = -4.0;                    // Difference between interior and terminus SMB [m/yr]. 
    double const x_acc    = 300.0e3;                 // Position at which accumulation starts decreasing [m]. 300.0, 355.0
    double const x_mid    = 3.5e5;                   // Position of middle of SMB sigmoid  [m]. 365.0, 375.0
    double const x_sca    = 4.0e4;                   // Length scale of area where SMB changing. [m]
    double const x_varmid = 2.0e5;                   // Position of middle of SMB variability sigmoid [m].
    double const x_varsca = 8.0e4;                   // Length scale of area where SMB varaibility changing. [m]
    double const var_mult = 0.25;                    // Factor by which inland variability is less than max.
    double m_stoch;
    double smb_stoch;
    int t_stoch;          


    // THERMODYNAMICS.
    // Vertical advection is the key to obtain oscillations.
    // It provides with a feedback to cool down the ice base and balance frictional heat.
    bool const thermodynamics  = false;              // Apply thermodynamic solver at each time step.
    int const thermodynamics_w = 1;                  // Vertical advection in therm. 0: no advection, 1: constant/linear adv.
    double const k = 2.0;                            // Thermal conductivity of ice [W / m · ºC].
    double const G = 0.05;                           // Geothermal heat flow [W / m^2] = [J / s · m^2].
    double const G_k = G / k;                        // [K / m] 
    double const kappa = 1.4e-6 * sec_year;          // Thermal diffusivity of ice [m^2/s] --> [m^2/yr].
    double const theta_max = 273.15;                 // Max temperature of ice [K].
    double const theta_act = 263.15;                 // Threshold temperature for the two regimes in activation energy [K]
                     
    double const R = 8.314;                          // Universal gas constant [J / K mol]
    double T_air = 253.15;                           // BC: prescribed air temperature. 253.15
    double const w_min = -0.3;                      // Prescribed vertical advection at x=0 in theta. 0.25
    double const w_max = 0.0;                        // Prescribed vertical advection at x=L in theta. 0.0
    

    // BEDROCK PARAMETRIZATION: f_C_bed.
    int const fric_therm = 0;                        // Temperature-dependent friction.
    double const theta_frz = 268.15;
    double const C_frz = 7.624e6 / pow(sec_year, m); // Frozen friction coeff. 7.624e6 [Pa m^-1/3 yr^1/3]
    double const C_thw = 0.5 * C_frz;                // Thawed friction coeff. [Pa m^-1/3 yr^1/3]

    // VISCOSITY-THERM
    bool const visc_therm = false;                   // Temperature-dependent viscosity. 0: no; 1: yes.
    double const t_eq_A_theta = t_eq;                // Eq. time to start applying Arrhenius dependency.
    double const A_act = 4.9e-25 * sec_year;         // Threshold rate factor for the two regimes in activation energy [Pa^-3 s^-1] 
    
    double const visc_0 = 1.0e8;                     // Initial viscosity [Pa·yr]
    double const visc_min = 1.0e6;
    double const visc_max = 1.0e11;                  // 1.0e8
    
    Array2d Q_act, A_0; 
    Q_act << 60.0, 139.0;                            // Activation energies [kJ/mol].
    Q_act = 1.0e3 * Q_act;                           // [kJ/mol] --> [J/mol] 
    A_0 << 3.985e-13, 1.916e3;                       // Pre-exponential constants [Pa^-3 s^-1]
    A_0 = A_0 * sec_year;                            // [Pa^-3 s^-1] --> [Pa^-3 yr^-1]
    
    // AVECTION EQUATION.
    int const H_meth = 0;                              // Solver scheme: 0, explicit; 1, implicit.

    // LATERAL BOUNDARY CONDITION.
    double D;                                        // Depth below the sea level [m].
    double u_x_bc;                                   // Boundary condition on u2 = dub/dx.
    double u2_dif;                                   // Difference between analytical and numerical.

    // CALVING.
    int const calving_meth = 0;                      // 0, no calving; 1, Christian et al. (2022), 2: deterministic Favier et al. (2019)
    double const m_dot = 30.0;                       // Mean frontal ablation [m/yr]. 30.0
    double H_f;

    // Sub-shelf melt.
    bool const shelf_melt = false; 
    int const melt_meth  = 0;                         // Melt parametrization. 0: linear; 1: quadratic.
    double const t0_oce  = 2.5e4;                    // Start time of ocean warming.
    double const tf_oce  = 27.5e4;                   // End time of applied ocean forcing.
    double const c_po    = 3974.0;                    // J / (kg K)
    double const L_i     = 3.34e5;                    // J / kg 
    double const gamma_T = 2.2e-5 * sec_year;        // Linear: 2.0e-5, Quad: 36.23e-5. [m/s] --> [m/yr]
    double const T_0     = 273.15;                    // K
    double T_oce; 
    double const delta_T_oce = 2.0;                    // Amplitude of ocean temperature anomalies. 
    double M = 0.0;                                   // Sub-shelf melt [m/yr]. 
    double const delta_M = 150.0;                      // Amplitude of sub-shelf melting.                         


    // Ice rate factor.
    double A, B;

    // PICARD ITERATION
    double error;                           // Norm of the velocity difference between iterations.
    double omega;                           // Angle between two consecutive velocities [rad]. 
    double mu;                              // Relaxation method within Picard iteration. 
    
    int c_picard;                           // Counter of Picard iterations.
    int const n_picard = 10;                // Max number iter. Good results: 10.
    
    double const picard_tol = 1.0e-4;              // 1.0e-4, Convergence tolerance within Picard iteration. 1.0e-5
    double const omega_1 = 0.125 * M_PI;           // De Smedt et al. (2010) Eq. 10.
    double const omega_2 = (19.0 / 20.0) * M_PI;




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
    ArrayXd w(n);                        // Synthetic vertical velocity.
    ArrayXd F_1(n);                      // Integral for DIVA solver (Arthern et al., 2015)
    ArrayXd F_2(n);                      // Integral for DIVA solver (Arthern et al., 2015)
    
    // Stochastic matrices and vectors.
    ArrayXXd noise(2,N);                            // Matrix to allocate stochastic BC.
    Array2d noise_now;                              // Two-entry vector with current stochastic noise.
    ArrayXd t_vec = ArrayXd::LinSpaced(N, 0.0, N);  // Time vector with dt_noise time step.
    
    // Vectors to compute norm.
    VectorXd u_bar_vec(n); 
    VectorXd c_u_bar_1(n);                   // Correction vector Picard relaxed iteration.
    VectorXd c_u_bar_2(n);
    //VectorXd x_0(n*n_z);                 // Initial guess for linear solver in Blatter-Pattyn solution.

    // FRONTAL ABLATION.
    ArrayXd M_s(n_s); 

    // MISMIP FORCING.
    ArrayXd A_s(n_s);                    // Rate factor values for MISMIP exp.
    ArrayXd t_s(n_s);                    // Time length for each step of A.
    ArrayXd T_air_s(n_s);                // Corresponding air temperature values to A_s. 
    ArrayXd T_oce_s(n_s);                // Ocean temperature forcing.   

    // MATRICES.
    ArrayXXd sol(n,n_z+1);                     // Matrix output. sol(2*n+1,2*n_z+1);
    ArrayXXd u(n,n_z);                     // Full velocity u(x,z) [m/yr].  
    ArrayXXd u_old(n,n_z);
    ArrayXXd u_z(n,n_z);                   // Full vertical vel derivative [1/yr]
    ArrayXXd u_x(n,n_z);              // Velocity horizontal derivative [1/yr].
    ArrayXXd strain_2d(n,n_z);           // Strain ratefrom DIVA solver.
    ArrayXXd visc_all(n,5*n_z+2);          // Output ice viscosity function [Pa·s]. (n,n_z+2)
    ArrayXXd visc(n,n_z);                  // Ice viscosity [Pa·s]. 
    ArrayXXd theta(n,n_z);                 // Temperature field [K].
    ArrayXXd A_theta(n,n_z);               // Temperature dependent ice rate factor [Pa^-3 yr^-1]
    ArrayXXd fric_all(n,4);            // Basal friction output.(n,n_z+3)
    ArrayXXd lmbd(n,n_z);                  // Matrix with stress vertical derivatives d(visc du/dz)/dz. 
    

    // Function outputs.
    Array2d L_out;                    // Grounding line function output.
    Array2d dt_out;                   // Time step function output.

    // Normalised horizontal dimension.
    // Try an unevenly-spaced horizontal grid to allows for fewer point whilst keeping
    // high resolution at the grounding line.
    ArrayXd sigma = ArrayXd::LinSpaced(n, 0.0, 1.0);      // Dimensionless x-coordinates.
    ArrayXd ds(n-1);
    ArrayXd ds_inv(n-1);
    ArrayXd ds_sym(n-1);
    
    double const n_sigma = 1.0;          // 0.5. Exponent of spacing in horizontal grid (1.0 = evenly-spaced). 
    sigma = pow(sigma, n_sigma);
    
    // Uneven spacing.
    for (int i=0; i<n-1; i++)
    {
        ds(i) = sigma(i+1) - sigma(i);
    }

    ds_inv = 1.0 / ds;

    // For symmetric diferences we sum two consecutive grid spacings.
    for (int i=1; i<n-1; i++)
    {
        ds_sym(i) = ds(i) + ds(i-1);
    }


    // Time steps in which the solution is saved. 
    ArrayXd a    = ArrayXd::LinSpaced(t_n, t0, tf);       // Array with output time frames.
    ArrayXd a_hr = ArrayXd::LinSpaced(int(tf), t0, tf);   // Array with high resolution time frames.    

    // EXPERIMENT. Christian et al (2022): 7.0e6
    // Constant friction coeff. 7.624e6, 7.0e6 [Pa m^-1/3 s^1/3]
    C_ref = ArrayXd::Constant(n, 7.624e6/ pow(sec_year, m) );    // [Pa m^-1/3 yr^1/3] 7.0e6, MISMIP: 7.624e6

    // We assume a constant viscosity in the first iteration. 1.0e8 Pa yr.
    visc     = ArrayXXd::Constant(n, n_z, visc_0);            // [Pa yr]
    visc_bar = ArrayXd::Constant(n, visc_0);


    // Implicit initialization.
    ub          = ArrayXd::Constant(n, 1.0);               // [m / yr] 
    u_bar       = ArrayXd::Constant(n, 1.0);               // [m / yr]
    u_bar_old_2 = ArrayXd::Constant(n, 1.0); 
    u           = ArrayXXd::Constant(n, n_z, 1.0);         // [m / yr]
    beta        = ArrayXd::Constant(n, 5.0e3);             // [Pa yr / m]
    tau_b       = beta * ub;

    // Temperature initial conditions (-25ºC).
    theta = ArrayXXd::Constant(n, n_z, 253.15);

    // Intialize ice thickness and SMB.
    H = ArrayXd::Constant(n, 10.0); // 10.0
    S = ArrayXd::Constant(n, 0.3); // 0.3

    // Initilize vertical discretization.
    ArrayXd dz(n);                                        // Vertical discretization (only x-dependecy for now).                                  
    dz = H / n_z;

    // Initialize vertical velocity (only x-dependency).
    if ( thermodynamics_w == 0 )
    {
        w = ArrayXd::Zero(n);
    }
    else if ( thermodynamics_w == 1 )
    {
        //w = ArrayXd::Constant(n_z, 0.6);
        w = ArrayXd::LinSpaced(n_z, 0.0, 1.0);
        w = 0.3 * pow(w, 1.0/3.0);  // 1.0/3.0

        //w = ArrayXd::LinSpaced(n_z, w_min, w_max);
        // Positive vertical direction defined downwards.
        w = - w;
        //cout << "\n w = " << w;
    }
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    // MISMIP EXPERIMENTS 1-2 FORCING.
    if ( exp == 1 )
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
        A_s = A_s * sec_year;     
    }

    // MISMIP EXPERIMENT 3 FORCING.
    if ( exp == 3 )
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
        A_s = A_s * sec_year;     
    }
    
    // MISMIP THERMODYNAMICS. 
    else if ( exp == 4 )
    {
        // Time length for each forcing step. 


        // ICE RATE FACTOR FORCING. [Pa^-3 s^-1].
        // Exps 1-2 hysteresis forcing.
        /*
        t_s << 3.0e4, 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4,
               30.0e4, 33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4;

        A_s << 4.6416e-24, 2.1544e-24, 1.0e-24, 4.6416e-25, 2.1544e-25, 1.0e-25,
                4.6416e-26, 2.1544e-26, 1.0e-26,
                2.1544e-26, 4.6416e-26, 1.0e-25, 2.1544e-25, 4.6416e-25, 1.0e-24,
                2.1544e-24, 4.6416e-24; 
        
        A_s = A_s * sec_year;   
        */
        
        /*
        // Exp 3 hysteresis forcing.
        A_s << 3.0e-25, 2.5e-25, 2.0e-25, 1.5e-25, 1.0e-25, 5.0e-26, 2.5e-26, 5.0e-26, 1.0e-25, 
               1.5e-25, 2.0e-25, 2.5e-25, 3.0e-25; 

        // Time length for a certain A value. 
        t_s << 3.0e4, 4.5e4, 6.0e4, 7.5e4, 9.0e4, 12.0e4, 15.0e4, 16.5e4, 18.0e4, 
               21.0e4, 24.0e4, 27.0e4, 28.5e4;

        // Unit conversion: [Pa^-3 s^-1] --> [Pa^-3 yr^-1].
        A_s = A_s * sec_year;   
        */
        
        
        // ICE HARDNESS CONVERSION TO TEMPERATURE.
        // Corresponding temperature amplitude is not wide enough for advance/retreate.
        /*
        for (int i=0; i<n_s; i++)
        {
            if ( A_s(i) < A_act )
            {
                T_air_s(i) = - Q_act(0) / ( R * log(A_s(i) / A_0(0)) );
            }
            else
            {
                T_air_s(i) = - Q_act(1) / ( R * log(A_s(i) / A_0(1)) );
            }
        }
        */


        
        // OCEAN TEMPERATURES ANOMALIES FORCING.
        // Change of sign in (T_0-T_oce) to produce advance/retreate.
        // Make sure length of positive/negative anomalies is the same
        // to retireve the initial state. Close hysteresis loop.
        /*
        t_s << 6.0e4, 7.5e4, 9.0e4, 10.5e4, 12.0e4, 13.5e4, 15.0e4, 16.5e4, 18.0e4, 19.5e4;
        T_oce_s << 273.15, 273.65, 274.15, 274.65, 275.15, 275.65,
                   274.65, 274.15, 273.65, 273.15;
        */

        /*
        t_s << 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4;
        T_oce_s << 273.15, 274.15, 275.15, 276.15, 277.15, 276.15, 275.15, 274.15, 273.15;
        T_air_s = ArrayXd::Constant(n_s, 203.15);
        */
        
        
        t_s << 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4,
               33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4, 57.0e4,
               60.0e4, 63.0e4, 66.0e4, 69.0e4, 72.0e4, 75.0e4, 78.0e4, 81.0e4, 84.0e4, 
               87.0e4, 90.0e4;
        
        T_oce_s << 273.15, 273.65, 274.15, 274.65, 275.15, 275.65, 276.15, 276.65, 277.15, \
                   277.65, 278.15, 278.65, 279.15, 279.65, 280.15,
                   279.65, 279.15, 278.68, 278.15, 277.65, 277.15, 
                   276.65, 276.15, 275.65, 275.15, 274.65, 274.15, 273.65, 273.15;

        T_air_s = ArrayXd::Constant(n_s, 193.15); // 193.15
        


        // FORCING DIRECTLY ON MELTING AND THEN TRANSFORM TO TEMPERATURE TO PLOT.
        /*
        t_s << 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4,
               33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4, 57.0e4,
               60.0e4, 63.0e4, 66.0e4, 69.0e4, 72.0e4, 75.0e4, 78.0e4;
        
        M_s << 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 
               60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0;
        T_air_s = ArrayXd::Constant(n_s, 193.15); // 193.15
        */
        

        // COMBINED OCEAN + AIR FORCING.
        // Factor of 0.25 for ocean temperatures compared to air (Golledge) (1/0.25).
        /*
        t_s << 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4;
        T_oce_s << 273.15, 274.15, 275.15, 276.15, 277.15, 276.15, 275.15, 274.15, 273.15;

        T_air_s = 200.15 + ( 4.0 * ( T_oce_s - T_0 ) );
        //T_air_s = 193.15 + ( 4.0 * ( T_oce_s - T_0 ) );
        */
        

       /*
        t_s << 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4, 30.0e4, \
               33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4, 54.0e4;
        T_oce_s << 273.15, 273.65, 274.15, 274.65, 275.15, 275.65, 276.15, 276.65, 277.15, \
                   276.65, 276.15, 275.65, 275.15, 274.65, 274.15, 273.65, 273.15;
        T_air_s = 193.15 + ( 4.0 * ( T_oce_s - T_0 ) );
        */
        
        /*
        // TRY THE OCEAN FORCING WITHOUT REFREEZING. JUST POSITIVE ANOMALIES.
        // START THE DIAGRAM FROM THE REGITH WITH VERY COLD TEMPERATURES!
        T_oce_s << 273.15, 272.65, 272.15, 271.65, 271.15, 
                   271.65, 272.15, 272.65, 273.15, 273.65, 274.15, 274.65, 
                   275.15, 274.65, 274.15, 273.65, 273.15;


        // Constant air temperature for the oceanic forcing.
        T_air_s = ArrayXd::Constant(n_s, 243.15);
        */

        // AIR TEMPERATURES FORCING.
        // Stable forcing.
        /*
        t_s << 3.0e4, 6.0e4, 9.0e4, 12.0e4, 15.0e4, 18.0e4, 21.0e4, 24.0e4, 27.0e4,
               30.0e4, 33.0e4, 36.0e4, 39.0e4, 42.0e4, 45.0e4, 48.0e4, 51.0e4;

        T_air_s << 253.15, 243.15, 233.15, 223.15, 213.15, 203.15, 198.15, 193.15, 193.15,
                   198.15, 203.15, 213.15, 223.15, 233.15, 243.15, 253.15, 253.15;
        */
        

       // Temperature initial conditions (-25ºC).
       //theta = ArrayXXd::Constant(n, n_z, T_air_s(0));

       // Convert to kelvin.
       //cout << "\n Air temperatures = " << T_air_s;
       //cout << "\n ocean temperatures = " << T_oce_s;

        // Initialization.
        T_air   = T_air_s(0);
        T_oce   = T_oce_s(0);
        A       = A_s(0);
        A_theta = ArrayXXd::Constant(n, n_z, A);

    }   

    // TRANSITION INDICATORS EXPERIMENTS.
    else if ( exp == 5 )
    {
        //int const A_rate = 1;         // 0: constant A, 1: linear increase in A.

        //ArrayXd A_s(2);  
        //ArrayXd t_s(1); 
        //A_s << 2.0e-25, 20.0e-25; // Christian: 4.23e-25. 2.0e-25 is right at the peak for ewr.
        
        // WE NEED TO TUNE THIS NUMBER TOGEHTER WITH THE FLUX DISCRETIAZTION TO OBTAIN THE SAME EXTENT.
        A_s << 0.5e-26, 5.0e-25; // 4.227e-25, (0.5e-26, 5.0e-25)
        //t_s << 2.0e4;

        // FORCING CAN BE IMPOSED DIRECTLY ON A_s (i.e., an increase in temperature) or
        // on the calving at the front as a consequence of ocean warming.
    }
    
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    // Print spatial and time dimensions.
    cout << " \n n = " << n;
    cout << " \n n_sigma = " << n_sigma;
    cout << " \n tf = " << tf;

    // Call nc read function.
    if ( stoch == true )
    {
        noise = f_nc_read(N);
        //cout << "\n noise_ocn = " << noise;
    }
    

    // Call nc write functions.
    f_nc(n, n_z);
    //f_nc_hr(n, n_z);


    // Wall time for computational speed.
    auto begin = std::chrono::high_resolution_clock::now();

    // Counters to write solution.
    int c    = 0;
    int c_hr = 0;

    // Counter for MISMIP ice factor A forcing.
    int c_s = 0;

    // Initialize time.
    t  = t0;
    dt = dt_min;

    

    // TIME INTEGRATION.
    while (t < tf)
    {
        // MISMIP EXPERIMENTS 1, 3 and 3.
        if ( exp == 1 || exp == 3 )
        {
            // Update rate factor value.
            if ( t > t_s(c_s) )
            {
                c_s = min(c_s+1, n_s-1);
            }
            
            // Ice hardness.
            A = A_s(c_s);
            B = pow(A, (-1 / n_gln) );
        }

        // MISMIP-THERM EXPERIMENTS.
        else if ( exp == 4 )
        {
            // Update rate factor and T_air value.
            if ( t > t_s(c_s) )
            {
                c_s = min(c_s+1, n_s-1);

                // Ice hardness.
                A = A_s(c_s);
                B = pow(A, (-1 / n_gln) );

                // Air temperature as a BC.
                T_air = T_air_s(c_s);

                // Ocean temperature as a BC.
                T_oce = T_oce_s(c_s);
                //cout << " \n T_oce = " << T_oce;

                // Directly on frontal ablation.
                M = M_s(c_s);
            }

            // Ocean temperature as a BC (quadratic on time).
            // Factor 10 for an amplitude of 1ºC.
            /*
            if ( t < t0_oce || t > tf_oce )
            {
                //T_oce = T_0;
                M = 0.0;
            }
            else
            {
                //T_oce = T_0 + 10.0 * delta_T_oce * 0.25 * ( t - t0_oce ) * ( tf_oce - t ) \
                //                / pow(tf_oce-t0_oce, 2);
                M = 10.0 * delta_M * 0.25 * ( t - t0_oce ) * ( tf_oce - t ) \
                                / pow(tf_oce-t0_oce, 2);
            }
            */
            
            
        }

        // TRANSITION INDICATORS EXPERIMENTS.
        else if ( exp == 5 )
        {
            // Constant A throughout the sim.
            if ( A_rate == false )
            {
                A = A_s(0);
            }

            // Forcing in A.
            else
            {
                // Equilibration with constant A.
                if ( t < t0_A )
                {
                    A = A_s(0);
                }
                
                else if ( t >= t0_A && t <= tf_A )
                {
                    A = A_s(0) + ( A_s(1) - A_s(0) ) * (t - t0_A) / (tf_A - t0_A);
                }

                else if ( t > tf_A )
                {
                    A = A_s(1);
                }
            }

            // Ice hardness.
            A = A * sec_year;
            B = pow(A, ( -1 / n_gln ) );
        }
    

        // Update bedrock with new domain extension L.
        bed = f_bed(L, n, bed_exp, y_0, y_p, x_1, x_2, smooth_bed, sigma_gauss, sigma);

        // Friction coefficient.
        C_bed = f_C_bed(C_ref, theta, H, t, t_eq, theta_max, \
                        theta_frz, C_frz, C_thw, rho, g, fric_therm, n);

        // Stochastic configuration. 
        // Update time-dependent boundary conditions after equilibration.
        if ( stoch == true )
        {
            // Christian's spin-up also considers stochastic anomalies?
            // Lower bound of zero in m to avoid numerical issues.
            // Start counting time when stoch is applied (t0_stoch).

            // TIME IS NOW WORKING CORRECTLY, BUT IT IS NOT TAKING THE STOCHASTIC
            // VALUES CORRECTLY!!! FILES VALUES LOOK OK WITH NCVIEW THOUGH.
            t_stoch = floor(max(0.0, t-t0_stoch));
            //cout << "\n t_sotch = " << t_stoch;
            
            noise_now = noise.col(t_stoch);
            m_stoch   = max(0.0, noise_now(0)); 
            smb_stoch = noise_now(1);

            //cout << "\n m_stoch = " << m_stoch;
            //cout << "\n smb_sotch = " << smb_stoch;

            // Update SMB considering new domain extension and current stochastic term.
            S = f_smb(sigma, L, S_0, x_mid, x_sca, x_varmid, \
                      x_varsca, dlta_smb, var_mult, smb_stoch, t, t0_stoch , n, stoch);
        }

        // Picard initialization.
        error    = 1.0;
        c_picard = 0;
        
        // Implicit velocity solver. Picard iteration for non-linear viscosity and beta.
        // Loop over the vertical level for Blatter-Pattyn.
        // We solve one tridiagonal solver for each vertical level.
        while (error > picard_tol && c_picard < n_picard)
        {
            // Save previous iteration solution.
            u_bar_old_1 = u_bar;
            u_old       = u;
            
            // Implicit solver.
            // If SSA solver ub = u_bar.
            sol = vel_solver(H, ds, ds_inv, ds_sym, dz, n, n_z, visc_bar, bed, rho, rho_w, g, L, \
                                C_bed, t, beta, A, A_theta, n_gln, visc, u, \
                                    u_z, visc_therm, vel_meth);
            
            // Allocate variables. sol(n+1,n_z+1)
            u_bar  = sol.block(0,0,n,1);
            u      = sol.block(0,1,n,n_z);

            
            // Update beta with new velocity.
            fric_all = f_u(u, u_bar, beta, C_bed, visc, H, dz, sec_year, t, t_eq, m, vel_meth, n_z, n);
            beta     = fric_all.col(0);

            // Update viscosity with new velocity.
            visc_all = f_visc(theta, u, visc, H, tau_b, u_bar, dz, \
                                theta_act, ds, ds_inv, ds_sym, L, Q_act, A_0, n_gln, R, B, n_exp, \
                                    eps, t, t_eq, sec_year, n, n_z, vel_meth, A, \
                                        visc_therm, visc_0);
            
            // Allocate variables.
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
                    error, dt, c_picard, mu, omega, theta, visc, u_z, u_x, u, A, dL_dt, \
                    F_1, F_2, m_stoch, smb_stoch, A_theta, T_oce, lmbd);

            // Close nc file. 
            if ((retval = nc_close(ncid)))
            ERR(retval);
            printf("\n *** %s file has been successfully written \n", FILE_NAME);
                
            // Abort flowline.
            return 0;
        }
        
        
        // Update sub-shelf melt.
        if ( shelf_melt == true )
        {
            M = f_melt(T_oce, T_0, rho, rho_w, c_po, L_i, gamma_T, melt_meth);
        }
        
        // Ice flux calculation. Flotation thickness H_f.
        H_f = D * ( rho_w / rho );
        q   = f_q(u_bar, H, H_f, t, t_eq, rho, rho_w, m_stoch, M, calving_meth, n);
        
        // Update grounding line position with new velocity field.
        L_out = f_L(H, q, S, bed, dt, L, ds, n, rho, rho_w, M);
        L     = L_out(0);
        dL_dt = L_out(1);
        
        // Write solution with desired output frequency.
        if ( c == 0 || t > a(c) )
        {
            // std::cout is typically buffered by default.
            // By using std::flush or std::endl, you ensure that the data is 
            // immediately written to the output device. 
            cout << "\n t =                  " << t << std::flush;
            cout << "\n dx_min =             " << 1.0e-3*L*ds(n-2) << std::flush;
            cout << "\n #Picard iterations:  " << c_picard << std::flush;
            cout << "\n Estimated error:     " << error << std::flush;
            
            //cout << "\n noise_now(0) = " << noise_now(0);

            // Write solution in nc.
            // Sub-shelf melting as forcing of MISMIP+thermodynamics.
            //m_stoch = M;

            f_write(c, u_bar, ub, u_bar_x, H, visc_bar, S, tau_b, beta, tau_d, bed, \
                    C_bed, Q_fric, u2_dif_vec, u2_0_vec, L, t, u_x_bc, u2_dif, \
                    error, dt, c_picard, mu, omega, theta, visc, u_z, u_x, u, A, dL_dt, \
                    F_1, F_2, m_stoch, smb_stoch, A_theta, T_oce, lmbd);

            ++c;
        }  
        /*
        // Write solution with high resolution output frequency.
        else if ( out_hr == true && c_hr == 0 || t > a_hr(c_hr) )
        {
            // Write solution in nc.
            f_write_hr(c_hr, u_bar(n-1), H(n-1), L, t, u_x_bc, u2_dif, \
                       error, dt, c_picard, mu, omega, A, dL_dt, m_stoch, smb_stoch);

            ++c_hr;
        }  
        */

    
        // Integrate ice thickness forward in time.
        H = f_H(u_bar, H, S, sigma, dt, ds, ds_inv, ds_sym, n, \
                    L, D, rho, rho_w, dL_dt, bed, q, M, H_meth, t, t_eq);
        
        // Update vertical discretization.
        dz = H / n_z;

        // Apply thermodynamic solver if desired.
        if ( thermodynamics == false || t < t_eq )
        {
            theta = ArrayXXd::Constant(n, n_z, 253.15);
        }
        
        // Integrate Fourier heat equation.
        else if ( thermodynamics == true && t >= t_eq )
        {
            theta = f_theta(theta, ub, H, tau_b, Q_fric, sigma, dz, theta_max, T_air, kappa, \
                            k, dt, G_k, ds, L, dL_dt, t, t_eq, w, n, n_z, vel_meth, strain_2d);
        }

        // Courant-Friedrichs-Lewis condition.
        // Factor 0.5 is faster since it yields fewer Picard's iterations.
        dt_CFL = 0.5 * ds.minCoeff() * L / u_bar.maxCoeff();

        // Update timestep and current time.
        dt_out = f_dt(error, picard_tol, dt_meth, t, dt, \
                        t_eq_dt, dt_min, dt_max, dt_CFL, rel);
        t  = dt_out(0);
        dt = dt_out(1);
        
    }
    

    // Running time (measures wall time).
    auto end     = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    
    // Print computational time.
    printf("\n Time measured: %.3f minutes.\n", elapsed.count() * 1e-9 / 60.0);
    printf("\n Computational speed: %.3f kyr/hr.\n", \
            60 * 60 * (1.0e-3 * tf) /  (elapsed.count() * 1e-9) );

    // Close nc files. 
    if ((retval = nc_close(ncid)))
    ERR(retval);
    //if ((retval = nc_close(ncid_hr)))
    //ERR(retval);
 
    printf("\n *** %s file has been successfully written \n", FILE_NAME);
    printf("\n *** %s file has been successfully written \n", FILE_NAME_HR);
    


    return 0;
}