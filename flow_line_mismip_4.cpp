#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netcdf.h>
#include <chrono>

using namespace Eigen;
using namespace std;

#include "write_nc.cpp"

// TEST BRANCH UNIT_YEARS.

// Our flow line model uses netcdf and Eigen libraries. Make sure both are installed.
// Eigen: https://eigen.tuXdamily.org.
// Directory where eigen is installed: $ dpkg -L libeigen3-devcd
// Directory of netcdf libraries: $ nc-config --libdir

// LOCAL: 
// (old: g++ -I /usr/include/eigen3/ test.cpp -o test.o)
// g++ -I /usr/include/eigen3/ -o rungeKutta_adv_nc_t_N_array.o rungeKutta_adv_nc_t_N_array.cpp -lnetcdf

// ICEBERG COMPUTER:
// g++ -std=c++11 -I /usr/include/eigen3/ -o rungeKutta_adv_nc_t_N.o rungeKutta_adv_nc_t_N.cpp -lnetcdf
// /home/dmoren07/c++/eigen3

// BRIGIT:
// Modules required (module load <modulename>): nectcdf for c++ and gnu compiler
// module load gnu8
// If intel compiler is loaded:
// module swap intel gnu8/8.3.0

/* g++ -std=c++11 -I/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/include/ -I/usr/include/eigen3/ -L/opt/ohpc/pub/libs/gnu8/impi/netcdf/4.6.3/lib/ 
-lnetcdf -o rungeKutta_adv_nc_t_N_optim.o rungeKutta_adv_nc_t_N_optim.cpp */

// To run program: 
// ./flow_line.o

// Eigen fixed and dynamic sizes:
// Use fixed sizes for very small sizes where you can, and use dynamic sizes for larger 
// sizes or where you have to. For small sizes, especially for sizes smaller than
// (roughly) 16, using fixed sizes is hugely beneficial to performance, as it allows Eigen 
// to avoid dynamic memory allocation and to unroll loops.

// double can contain up to seven digits in total, not just following the decimal point.



// FLOWLINE MODEL STRUCTURE. LIST OF FUNCTIONS AND TOOLS.
/*
TOOLS.
gauss_filter --->  Returns a gaussian smooth by using Weierstrass transform.
half_step    --->  Returns variables evaluated in i+1/2 (n-1 dimensions).
deriv_sigma  --->  Returns numerical derivative O(h²) in sigma coordinates.     
FLOWLINE FUNCTIONS.
f_bed        --->  Returns the topography in the current step from domain length L.
f_L          --->  Returns the new grounding line positin L from explicit 
                   integration.
f_h_flux     --->  Returns new ice thickness H by integrating the 
                   advection eq (sigma).
f_dhdx       --->  Returns net driving contribution: bed slope + ice thickness slope.
f_visc       --->  Returns ice viscosity from ice temperature and du/dx. 
f_calv       --->  Returns maximum ice thickness permitted at the calving front
                   from yield strngth of ice and hydrostatic balance.
f_theta      --->  Thermodynamic solver. Returns new temperature field (theta, in K)
                   considering vertical difussion and horizontal advection.
f_C_bed      --->  Friction coefficient. Two-valued weighed average regarding 
                   the thermal state of the base (thawed/frozen).
f_du_ds      --->  Derivatives in the system of two 1st-order differential eqs.
                   Notation: u1 = u (velocity) and u2 = du1/ds (s denotes sigma).
                   Vectorial magnitude: f_du_dz(0) = du1/ds and f_du_ds(1) = du2/ds.
rungeKutta   --->  4th-order Runge-Kutta integration scheme for the SSA stress 
                   balance. Spatial resolution is ds to ensure consistency with 
                   derivatives. 
*/


// WE USE RUNGE-KUTTA FOR EXPLICIT INTEGRATION IN THE FIRST TIME STEP.
// THEN WE PROCEED WITH AN IMPLICIT SCHEME TO ENSURE BOUNDARY CONDITIONS.


// GENERAL PARAMETERS.
double const sec_year = 3.154e7;              // Seconds in a year.

// PHYSICAL CONSTANTS.
double const u_0   = 150.0 / sec_year;
double const g     = 9.8;                    // 9.81
double const rho   = 900.0;
double const rho_w = 1000.0;                 // 1028.8

// BEDROCK PARAMETRIZATION: f_C_bed.
double const C_thaw = 7.624e6;                  // 1.75e6 [Pa m^-1/3 s^1/3].
double const C_froz = 7.624e6;                  // 2.0e6 [Pa m^-1/3 s^1/3].

// GROUNDING LINE.
double L     = 479.1e3;                    // Grounding line position [m] (702.3e3;)
double L_old = 479.1e3;                    // Previous timestep.
double L_new;                              // New GL position.
double dL_dt;                              // GL migration [m/yr]. 

// ICE VISCOSITY: f_visc.
double const n_gln = 3.0;
double const n_exp = (1.0 - n_gln) / n_gln;      // Pattyn.

// VISCOSITY REGULARIZATION TERM.
// eps is fundamental for GL, velocities, thickness, etc.
double const eps = 1.0e-10;                            // Final: 1.0e-30. Current 1.0e-7. Yelmo: 1.0e-6. 2.5e-9 good GL but bad H.
//double const eps = 1.0e-30;                               // 1.0e-7

// BASAL FRICTION.
double const m = 1.0 / 3.0;                  // Friction exponent.

// SIMULATION PARAMETERS.
double const t0    = 0.0;                // Starting time [s].
double const tf    = 5.0e3;             // 56.5e4, Ending time [yr] * [s/yr]
double t     = t0;                       // Time initialization [s].
double dt;                               // Time step [s].
double dt_CFL;                           // Courant-Friedrichs-Lewis condition
double const dt_max = 0.1;              // Maximum time step = 0.25 years [s].
//double const dt_min = 1.0;
double const t_eq = 0.0;                 // 20.0. Length of explicit scheme. 
//double const t_bc = 10.0;                // 1.0e3. Implicit scheme spin-up.

// OUTPUT DEFINITIONS.
int const t_n = 100;                        // Number of output frames. 30.

ArrayXd a = ArrayXd::LinSpaced(t_n, t0, tf);      // Time steps in which the solution is saved. 


// COORDINATES.
// HIGHLY SENSITIVE TO THE PARTICULAR CHOICE OF dt and n.
// TRY AN ADAPTATIVE TIMESTEP THAT CONSIDERS THE ERROR IN THE PICARD ITERATION?
int const n = 250;                     // 1000. Number of horizontal points 250, 500, 1000, 2000
double const ds = 1.0 / n;               // Normalized spatial resolution.
double const ds_inv = n;

int const   n_z = 10;                   // Number vertical layers. 10, 20.
double const dz = 1.0 / n_z;             // Normalized vertical resolution.
double const dz_inv = n_z;

ArrayXd sigma = ArrayXd::LinSpaced(n, 0.0, 1.0);    // Dimensionless x-coordinates. 

// Auxiliar definitions.
ArrayXd zeros = ArrayXd::Zero(n);


// BEDROCK
// Glacier ews option.
double const x_1 = 346.0;
double const x_2 = 350.0;
double const y_0 = 0.1;
double const y_p = 90.0e-3;


// THERMODYNAMICS.
double const k = 2.0;              // Thermal conductivity of ice [W / m · ºC].
double const G = 0.05;             // Geothermal heat flow [W / m^2] = [J / s · m^2].
double const G_k = G / k;          // [K / m] 
double const kappa = 1.4e-6;       // Thermal diffusivity of ice [m^2/s].
double const theta_max = 273.15;   // Max temperature of ice [K].

// LATERAL BOUNDARY CONDITION.
double D;                               // Depth below the sea level [m].
double u2_bc;                           // Boundary condition on u2 = du1/dx.
double u2_dif;                          // Difference between analytical and numerical.

// CALVING.
int const calv = 1;
double const m_dot = 30.0;              // Mean frontal ablation [m/yr]


// Runge-Kutta boundary conditions.
double u1_0 = 0.0;                      // Velocity in x0 (i.e., u(x0)).
double u2_0 = 0.0;                      // Velocity first derivative in x0 (i.e., du/dx(x0)).

// MISMIP EXPERIMENT CHOICE.
// Following Pattyn et al. (2012) the overdeepening hysterisis uses n = 250.
// exp = "mismip_1", "mismip_3", "galcier_ews"
//int const mismip = 1;
char const exp = "glacier_ews"
double A, B;

// PICARD ITERATION
double error;                           // Norm of the velocity difference between iterations.
double const picard_tol = 1.0e-4;       // 1.0e-5. Convergence tolerance within Picard iteration.
int const n_picard = 10;                // Max number iter. Good results: 10, 20.
int c_picard;                           // Number of Picard iterations.
double omega, mu;                 
double alpha;                           // Relaxation method within Picard iteration. 0.5, 0.7
double const alpha_max = 1.0;
double const omega_1 = 0.125 * M_PI;          // De Smedt et al. (2010) Eq. 10.
double const omega_2 = (19.0 / 20.0) * M_PI;


// MISMIP EXPERIMENTS FORCING.
// Number of steps in the A forcing.
int const n_s = 21;  // 3, 21


// PREPARE VARIABLES.
ArrayXd H(n);                        // Ice thickness [m].
ArrayXd u1(n);                       // Velocity [m/yr].
ArrayXd u2(n);                       // Velocity first derivative [1/yr].
ArrayXd q(n);                        // Ice flux [m²/yr].
ArrayXd bed(n);                      // Bedrock elevation [m].
ArrayXd C_bed(n);                    // Friction coefficient [Pa m^-1/3 s^1/3].
ArrayXd visc(n);                     // Ice viscosity [Pa·s].
ArrayXd S(n);                        // Surface accumulation equivalent [mm/day].
ArrayXd H_c(n);                      // Maxium ice thickness permitted at calving front [m].
ArrayXd u1_plot(n);                  // Saved ice velocity [m/yr]
ArrayXd u2_plot(n);                  // Saved ice velocity derivative [1/yr]
ArrayXd tau_b(n);                    // Basal friction [Pa]
ArrayXd beta(n);                    // Basal friction [Pa m^-1 yr]
ArrayXd tau_d(n);                    // Driving stress [Pa]
ArrayXd filt(n);                     // 
ArrayXd u1_old_1(n);  
ArrayXd u1_old_2(n);  
ArrayXd u2_0_vec(n);                 // Ranged sampled of u2_0 for a certain iteration.
ArrayXd u2_dif_vec(n);               // Difference with analytical BC.
ArrayXi u1_nan(n);
VectorXd u1_vec(n); 
VectorXd u2_vec(n); 
VectorXd c_u1_1(n);                   // Correction vector Picard relaxed iteration.
VectorXd c_u1_2(n);
VectorXd c_u1_dif(n);

ArrayXd A_s(n_s);                    // Rarte factor values for MISMIP exp.
ArrayXd t_s(n_s);                    // Time length for each step of A.


//ArrayXd smth(n);                     // Smooth field.
//MatrixXd smth(3,n);                // Smooth field.
MatrixXd u(7,n);                     // Matrix output.

ArrayXXd theta(n,n_z);               // Temperature field [K].
ArrayXXd theta_now(n,n_z);           // Current temperature field [K].
//MatrixXd u_old(1,n);                 // Store previous iteration for relaxation.




////////////////////////////////////////////////////
////////////////////////////////////////////////////
// TOOLS. USEFUL FUNCTIONS.

ArrayXd gaussian_filter(ArrayXd w, ArrayXd zeros, \
                        double sigma, double L, double ds, int n)
{
    ArrayXd smth(n), summ(n);
    //ArrayXd summ = ArrayXd::Zero(n);

    double x, y;
    double h_L = ds * L;
    double A = 1.0 / (sqrt(2.0 * M_PI) * sigma);

    summ = zeros;
 
    // Weierstrass transform.
    for (int i=0; i<n; i++) 
    {
        x = i * h_L;
        for (int j=0; j<n; j++)
        {
            y = j * h_L;
            summ(i) = summ(i) + w(j) * \
                                exp( - pow((x - y) / sigma, 2) / 2.0 ) * h_L;
        }
    }
 
    // Normalising Kernel.
    smth = A * summ;

    return smth;
}

ArrayXd running_mean(ArrayXd x, int p, int n)
{
    ArrayXd y(n);

    double sum, k;

    // Assign values at the borders f(p).
    y = x;

    // Average.
    k = 1.0 / ( p + 1.0 );
 
    // Loop.
    for (int i=p; i<n-p; i++) 
    {
        sum = 0;
        for (int j=i-p; j<i+p+1; j++) 
        {
            sum = sum + x(j);
        }
        y(i) = k * sum;
    }
 
    return y;
}


// half_step(dhdx, visc, dH, d_visc, c1, c2, C_bed, H, n);
MatrixXd half_step(ArrayXd dhdx, ArrayXd visc, \
                   ArrayXd dH, ArrayXd d_visc, ArrayXd c1, \
                   ArrayXd c2, ArrayXd C_bed, ArrayXd H, int n)
{
    // Function that return a vector of length n-1 with the average of 
    // every contiguous elements.
    MatrixXd x(n+1,8), x_n2(n+1,8), half_n2(n+1,8), half(n-1,8); 
    
    // Allocate x matrix (every variable in a column).
    x.block(0,0,n,1) = dhdx;
    x.block(0,1,n,1) = visc;
    x.block(0,2,n,1) = dH;
    x.block(0,3,n,1) = d_visc;
    x.block(0,4,n,1) = c1;
    x.block(0,5,n,1) = c2;
    x.block(0,6,n,1) = C_bed;
    x.block(0,7,n,1) = H;

    // Allocate shifted matrix (one row below).
    x_n2.block(1, 0, n, 8) = x.block(0, 0, n, 8);

    half_n2 = 0.5 * (x + x_n2);
    half    = half_n2.block(1, 0, n-1, 8);

    return half;
}


MatrixXd deriv_sigma(ArrayXd x, ArrayXd y, ArrayXd z,\
                     int n, double ds_inv, double L)
{   
    MatrixXd grad(n, 3), x_min(n+2, 3), \
             x_plu(n+2, 3), grad_n2(n+2, 3);
    
    //double dx = L * ds;           // ds = 1 / n.
    //double dx_inv = 1.0 / dx;
 
    x_plu.block(0,0,n,1) = x;
    x_plu.block(0,1,n,1) = y;
    x_plu.block(0,2,n,1) = z;

    x_min.block(2,0,n,1) = x;
    x_min.block(2,1,n,1) = y;
    x_min.block(2,2,n,1) = z;

    // CORRECTION INCLUDED:
    // These are d/d(sigma) derivatives, O(h²). No L corrections!
    grad_n2 = 0.5 * (x_plu - x_min);

    grad.block(1, 0, n-2, 3) = grad_n2.block(2, 0, n-2, 3);

    // Derivatives at the boundaries sigma = 0, 1. O(h).
    grad.block(0,0,1,3)   = x_plu.block(1,0,1,3) - x_plu.block(0,0,1,3);
    grad.block(n-1,0,1,3) = x_plu.block(n-1,0,1,3) - x_plu.block(n-2,0,1,3);

    // Horizontal resolution in the derivative.
    grad = grad * ds_inv;
    //grad = grad * dx_inv;

	return grad;
}


ArrayXd tridiagonal_solver(ArrayXd A, ArrayXd B, ArrayXd C, \
                           ArrayXd F, int n)
{
    ArrayXd P(n), Q(n), x(n);
    double m;
    int j;
    
    // This allows us to perform O(n) iterations, rather than 0(n³).
    // A subdiagonal, B diagonal and C uppder-diagonal.
    
    // Ensure tridiagonal definition is satisfied.
    A(0)   = 0.0;
    C(n-1) = 0.0;

    // First element. Ensure that B(0) is non-zero!
    P(0) = - C(0) / B(0);
    Q(0) = F(0) / B(0);

    // Forward elimination.
    for (int i=1; i<n; i++)
    {
        m    = 1.0 / ( B(i) + A(i) * P(i-1) );
        P(i) = - C(i) * m;
        Q(i) = ( F(i) - A(i) * Q(i-1) ) * m;
    }

    // From notes: x(n-1) = Q(n-1).    
    x(n-1) = Q(n-1);      
    //cout << "\n u1(n-1) = " << u1(n-1);
    
    // Back substitution.
    for (int j = n-2; j>0; --j)
    {
        x(j) = P(j) * x(j+1) + Q(j);
    }

    return x;
}


////////////////////////////////////////////////////
////////////////////////////////////////////////////
// Flow line functions.

ArrayXd f_bed(ArrayXd sigma, double L, int n, char exp, double x_1, double x_2)
{
    ArrayXd bed(n), x_scal(n);

    // Same bedrock as Schoof (2007).
    // sigma = x / L. L in metres!
    x_scal = sigma * L / 750.0e3; 

    // MISMIP experiments bedrock.
    // Inverse sign to get a decreasing bedrock elevation.
    if (exp == "mismip_1")
    {
        bed = 720.0 - 778.5 * x_scal;
    }
    else if (exp == "misimip_3")
    {
        bed = 729.0 - 2148.8 * pow(x_scal, 2) + \
                      1031.72 * pow(x_scal, 4) + \
                    - 151.72 * pow(x_scal, 6);
    }
    // REVISE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    else if (exp = "glacier_ews")
    {
        // Variables.
        int c_x1 = 0;
        int c_x2 = 0;
        double m_bed = y_p / ( x_2 - x_1 );

        // Extension in km.
        ArrayXd x = ArrayXd::LinSpaced(n, 0.0, 1.0e-3*L); 


        for (int i=0; i<n; i++)
        {
            // First part.
		    if ( x(i) <= x_1 )
            {
                bed(i) = y_0 - 1.5e-3 * x(i)
            }
		
		    // Second.
            else if ( x_tilde(i) >= x_1 and x(i) <= x_2 )
                {
                    // Save index of last point in the previous interval.
                    if ( c_x1 == 0 )
                    {
                        y_1  = bed(i-1)
                        c_x1 = c_x1 + 1	
                    }
                    
                    // Bedrock function.
                    bed(i) = y_1 + m_bed * ( x(i) - x_1 )
                }
                
            // Third.
            else if ( x_tilde(i) > x_2 )
            {
                // Save index of last point in the previous interval.
                if (c_x2 == 0)
                {
                    y_2  = bed(i-1)
                    c_x2 = c_x2 + 1	
                }
                    
                // Bedrock function.
                bed(i) = y_2 - 5.0e-3 * ( x(i) - x_2 )
            } 

        }

    }



/*
# Bedrock.
	bed = np.empty(n)

	# x_tilde in metres.
	x_tilde = np.linspace(0, 400.0, n)

	# Horizontal domain extenstion [km].
	x_1 = 346.0
	x_2 = 350.0

	# Initial bedrock elevation (x = 0) [km].
	y_0 = 0.1

	# Peak height [km].
	y_p = 90.0e-3

	# Intermideiate slope.
	m_bed = y_p / ( x_2 - x_1 )

	# Counters.
	c_x1 = 0
	c_x2 = 0

	for i in range(n):

		# First part.
		if x_tilde[i] <= x_1:
			bed[i] = y_0 - 1.5e-3 * x_tilde[i]
		
		# Second.
		elif x_tilde[i] >= x_1 and x_tilde[i] <= x_2:
			
			# Save index of last point in the previous interval.
			if c_x1 == 0:
				y_1  = bed[i-1]
				c_x1 = c_x1 + 1	
			
			# Bedrock function.
			bed[i] = y_1 + m_bed * ( x_tilde[i] - x_1 )
		
		# Third.
		elif x_tilde[i] > x_2:
			# Save index of last point in the previous interval.
			if c_x2 == 0:
				y_2  = bed[i-1]
				c_x2 = c_x2 + 1	
			
			# Bedrock function.
			bed[i] = y_2 - 5.0e-3 * ( x_tilde[i] - x_2 )
*/

    return bed;
}



double f_L(ArrayXd u1, ArrayXd H, ArrayXd S, ArrayXd H_c, \
          double dt, double L, double ds, int n, \
          ArrayXd bed, double rho, double rho_w, ArrayXd q)
{
    int num_opt, den_opt;
    double num, den, dL_dt, D_5;
    
    
    // DISCRETIZATION OPTIONS.
    // MIT finite difference calculator: https://web.media.mit.edu/~crtaylor/calculator.html
    // den_opt = 4 rises advance/retreat problems.
    num_opt = 1;
    den_opt = 3;
    
    
    // Accumulation minus flux (reverse sign). 

    // Two-point backward-difference.
    if ( num_opt == 1)
    {
        num = - L * ds * S(n-1) + ( q(n-1) - q(n-2) );
    }
    // Three-point asymmetric backward-diference for bedrock. Unstable!
    else if ( num_opt == 2 )
    {
        num = - L * ds * S(n-1) + 0.5 * ( 3.0 * q(n-1) - 4.0 * q(n-2) + q(n-3) );
    }


    // Purely flotation condition (Following Schoof, 2007).
    // Sign before db/dx is correct. Otherwise, it migrates uphill.


    // Two-point backward-diference for bedrock and ice thickness.
    if ( den_opt == 1 )
    {
        den = H(n-1) - H(n-2) + ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );
    }
    // Three-point asymmetric backward-diference for bedrock.
    else if ( den_opt == 2 )
    {
        den = H(n-1) - H(n-2) + ( rho_w / rho ) * \
              0.5 * ( 3.0 * bed(n-1) - 4.0 * bed(n-2) + bed(n-3) );
    }
    // / Three-point asymmetric backward-diference bedrock and thickness.
    else if ( den_opt == 3 ) 
    {
        den = 3.0 * H(n-1) - 4.0 * H(n-2) + H(n-3) + ( rho_w / rho ) * \
              0.5 * ( 3.0 * bed(n-1) - 4.0 * bed(n-2) + bed(n-3) );
    }
    // Five-point asymmetric backward-difference deffinition. Unstable!
    else if ( den_opt == 4 ) 
    {
        D_5 = ( 137.0 * H(n-1) - 300.0 * H(n-2) + 300.0 * H(n-3) + \
                - 200.0 * H(n-4) + 75.0 * H(n-5) - 12.0 * H(n-6) ) / 60.0;
        
        // Simple bedrock slope.
        den = D_5 + ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );
    }


    // Grounding line time derivative.
    dL_dt = num / den;

    return dL_dt;
} 


// First order scheme. New variable sigma.
ArrayXd f_H_flux(ArrayXd u1, ArrayXd H, ArrayXd S, ArrayXd sigma, \
                    double dt, double ds, double ds_inv, int n, double L_new, \
                    double L, ArrayXd H_c, double D, \
                    double rho, double rho_w, double dL_dt, \
                    ArrayXd bed, ArrayXd q, int meth)
{
    ArrayXd H_now(n), A(n), B(n), C(n), F(n);
    double gamma, L_inv;

    L_inv  = 1.0 / L;
    gamma  = dt / ( 2.0 * ds * L );

    // Solution to the modified advection equation considering a streched coordinate
    // system sigma. Two schemes are available, explicit and implicit, noted as
    // meth = 0, 1 respectively. 
    //  Right now, explicit seems more stable since the 
    // implicit crasher earlier. However, neither of them is fully successful.

    /* Solver choice:
    Explicit ---> 0.
    Implicit ---> 1. */ 
    meth = 0;

    if ( meth == 0 )
    {
        // Explicit scheme. Centred dH in the sigma_L term.
        for (int i=1; i<n-1; i++)
        {
            // Centred in sigma, upwind in flux.
            H_now(i) = H(i) + dt * ( ds_inv * L_inv * \
                                    ( sigma(i) * dL_dt * \
                                        0.5 * ( H(i+1) - H(i-1) ) + \
                                            - ( q(i) - q(i-1) ) ) + S(i) );
        }
        
        // Symmetry at the ice divide (i = 1).
        H_now(0) = H_now(2);
        //H_now(0) = ( 4.0 * H_now(1) - H_now(2) ) / 3.0;
        
        // Lateral boundary: sigma(n-1) = 1.
        H_now(n-1) = H(n-1) + dt * ( ds_inv * L_inv * \
                                        (  dL_dt * ( H(n-1) - H(n-2) ) + \
                                            - ( q(n-1) - q(n-2) ) ) + S(n-1) );
        
        
    }
    else if ( meth == 1 )
    {
        // Implicit scheme. REVISE TRIDIAGONAL MATRIX.
        for (int i=1; i<n-1; i++)
        {
            // Tridiagonal vectors.
            A(i) = - u1(i-1) + sigma(i) * dL_dt;
            B(i) = 1.0 + gamma * ( u1(i) - u1(i-1) );
            C(i) = u1(i) - sigma(i) * dL_dt;

            // Inhomogeneous term.
            F(i) = H(i) + S(i) * dt;
        }

        // Vectors at the boundary.
        A(0) = 0.0;
        B(0) = 1.0 + gamma * u1(0);
        //B(0) = 1.0 + gamma * ( u1(1) - u1(0) );
        C(0) = u1(0);                  

        A(n-1) = - u1(n-2) + sigma(n-1) * dL_dt;
        B(n-1) = 1.0 + gamma * ( u1(n-1) - u1(n-2) );
        C(n-1) = 0.0;

        F(0)   = H(0) + S(0) * dt;
        F(n-1) = H(n-1) + S(n-1) * dt;

        // Discretization factor.
        A = gamma * A;
        C = gamma * C;

        // Tridiagonal solver.
        H_now = tridiagonal_solver(A, B, C, F, n);

        // Boundary conditons.
        H_now(0)   = H_now(2);
        //H_now(n-1) = ( F(n-1) - A(n-1) * H_now(n-2) ) / B(n-1);
        
        H_now(n-1) = H(n-1) + S(n-1) * dt - gamma * ( 2.0 * dL_dt * H_now(n-2) \
                    - u1(n-2) * H_now(n-2) ) / ( 1.0 + gamma * ( - 2.0 * dL_dt \
                                                + u1(n-1) - u1(n-2) ) ); 
        
        //H_now(n-1) = H(n-1) + S(n-1) * dt - gamma * ( 2.0 * dL_dt * H_now(n-2) \
                    - u1(n-2) * H_now(n-2) ) / ( 1.0 + gamma * ( - 2.0 * dL_dt \
                                                - u1(n-2) ) ); 
        
        //H_now(n-1) = min( D * ( rho_w / rho ), H_now(n-1) );
        
        
        //H_now(n-1) = D * ( rho_w / rho );
        //H_now(n-1) = min( D * ( rho_w / rho ), H(n-2) );

        //cout << "\n H_now = " << H_now;   
    }

    return H_now; 
}


ArrayXd f_dhdx(ArrayXd dH, ArrayXd b, int n)
{
    ArrayXd dhdx(n);

    // Ice surface elevation term plus bedrock contribution. \
    Ice surface elevation h = H + b.
    dhdx = dH + b;

	return dhdx;
}


ArrayXd f_visc(ArrayXd u2, double B, double n_exp, \
                double eps, double L, int n)
{
    ArrayXd u2_eps(n), visc(n); 

    // u2 comes from derivation of u1 (u2 > 0).

    // Regularitazion term to avoid division by 0. 
    u2_eps = u2 + eps;

    // n_exp = (1-n)/n
    visc = 0.5 * B * pow(u2_eps, n_exp);
    //cout << "\n visc = " << visc * sec_year;

    // Constant viscosity experiment:
    //visc = ArrayXd::Constant(n, 0.5e17); // 1.0e15, 0.5e17
    
	return visc;
}	

ArrayXd f_calv(ArrayXd tau, double D, \
               double rho, double rho_w, double g, ArrayXd bed)
{
    ArrayXd H_c(n), tau_tilde(n), bed_tilde(n);

    // Following Bassis et al. (2011; 2017)
    // Maximum ice thickness permitted at the calving front H_c.
    tau_tilde = tau / (rho * g);

    // Bassis and Walker (2012) Eq. 2.10, Bassis et al. (2017). 
    H_c = tau_tilde + sqrt( pow(tau_tilde, 2) \
                            + (rho_w / rho) * pow(bed, 2) );

    return H_c;
} 


ArrayXXd f_theta(ArrayXXd theta, ArrayXd u, ArrayXd H, ArrayXd tau_b, \
                double theta_max, double kappa, double k, double dt, double G_k, \
                double ds, double L, int n, int n_z)
{
    MatrixXd theta_now(n,n_z);
    ArrayXd dz(n), dz_2_inv(n), Q_f_k(n);

    double dx_inv;
 
    dz       = H / n_z;
    dz_2_inv = 1.0 / pow(dz, 2);
    dx_inv   = 1.0 / (ds * L);
    Q_f_k   = u * tau_b / k;      // [Pa m s^-1] = [J s^-1 m-2] = [W m-2] => [K / m]
    //Q_f_k    = Q_fric / k;      // [W m-2] => [K / m]

    for (int i=1; i<n; i++)
    {
        for (int j=1; j<n_z-1; j++)
        {
            theta_now(i,j) = theta(i,j) + dt * ( kappa * dz_2_inv(i) * \
                             ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                             - u(i) * ( theta(i,j) - theta(i-1,j) ) * dx_inv );
        }
        
        // Boundary conditions. Geothermal heat flow at the base \
        and prescribed theta (-20ºC) at the surface.
        // We add friciton heat contribution Q_f_k.
        theta_now(i,0)     = theta_now(i,1) + dz(i) * ( G_k + Q_f_k(i) ); 
        theta_now(i,n_z-1) = 248.0;

        // Pressure melting point as the uppder bound.
        theta_now(i,0) = min(theta_now(i,0), theta_max);
    }
    // OPTIMIZE THIS!!!!!!!!
    // Vertical loop for x = 0.
    for (int j=1; j<n_z-1; j++)
    {
        theta_now(0,j) = theta(0,j) + dt * kappa * dz_2_inv(0) * \
                            ( theta(0,j+1) - 2.0 * theta(0,j) + theta(0,j-1) ) ;
    }
    // Boundary condition at x = 0.
    theta_now(0,0)     = theta_now(0,1) + dz(0) * G_k; 
    theta_now(0,n_z-1) = 253.0;
    theta_now(0,0)     = min(theta_now(0,0), theta_max);

    return theta_now;
}

ArrayXd f_C_bed(ArrayXXd theta, double theta_max, \
                double C_thaw, double C_froz, int n)
{
    ArrayXd C_bed(n), theta_norm(n);

    // Normalized basal temperature with pressure melting point [0,1].
    theta_norm = theta.block(0,0,n,1) / theta_max;
    C_bed      = C_thaw * theta_norm + C_froz * ( 1.0 - theta_norm );

    return C_bed;
}
	

Array2d du2_ds(double u_1, double u_2, double dh_dx, double visc,\
               double c_1, double c_2, double u_0, double m, double L, \
               double C_bed, double dH, double d_visc, double H)
{
    Array2d out;
    double du2, tau_b; 

    // Notation: du2 = du2/dx and u2 = du1/dx
    tau_b = C_bed * pow(u_1, m);

    // 1D SSA stress balance. L correction accounts for sigma derivatives.
    // We change sign to H and bed derivative so that negative values \
    contribute to positive change in du/dx. With d_visc_H this is a problem! \
    dH must be changed of sign but visc shouldn't! Separate derivatives?
    //du2   = c_1 * ( - L * ( c_2 * dh_dx + L * tau_b) - u_2 * d_visc_H );
    du2   = c_1 * ( - ( c_2 * dh_dx + tau_b) \
                    - u_2 * 4.0 * ( d_visc * H + visc * (-dH) ) );

    out(0) = u_2;
    out(1) = du2;

    return out;
}


MatrixXd rungeKutta(double u1_0, double u2_0, double u_0, ArrayXd H, \
                    double ds, double ds_inv, int n, ArrayXd visc, \
                    ArrayXd bed, double rho, double g, double L, double m, \
                    ArrayXd C_bed, double t_eq)
{
    ArrayXd u1(n), u2(n), dhds(n), dvisc_H(n), visc_dot_H(n), \
            c1(n), c2(n), h(n), dH(n), d_visc(n), \
            dhds_h2(n-1), d_visc_h2(n-1), visc_h2(n), H_h2(n), dH_h2(n), \
            c1_h2(n-1), c2_h2(n-1), u1_h2(n-1), u2_h2(n-1), C_bed_h2(n-1);

    MatrixXd dff(n,3), out(7,n), half(n-1,8);

    Array2d k1, k2, k3, k4, u_sol;
    double pre, D, u2_bc, vareps, tol, u2_dif, u2_dif_now;


    // Define for convenience.
    visc_dot_H = 4.0 * visc * H;
    c1         = 1.0 / visc_dot_H;      
    c2         = rho * g * H;
	pre        = 1.0 / 6.0;
    h          = bed + H;           // Ice surface elevation.
    
    // Numerical differentiation (sigma coord.) of H, bed and 4 * visc * H.
    dff     = deriv_sigma(h, H, visc, n, ds_inv, L);
    dhds    = dff.col(0);
    dH      = dff.col(1);
    d_visc  = dff.col(2);

    // Prepare variables in half-step (i + 1/2).
    half       = half_step(dhds, visc, dH, d_visc, c1, c2, C_bed, H, n);
    dhds_h2    = half.col(0);
    visc_h2    = half.col(1);
    dH_h2      = half.col(2);
    d_visc_h2  = half.col(3);
    c1_h2      = half.col(4);
    c2_h2      = half.col(5);
    C_bed_h2   = half.col(6);
    H_h2       = half.col(7);

    // Boundary condition at sigma = 1 (x = L).
    // Following Bassis et al. (2017).
    D = abs( min(0.0, bed(n-1)) );      // u_min is just double32 0.0.

    // Equivalent (Greve and Blatter 6.64).
    // Original:
    //u2_bc = 0.125 * g * H(n-1) * L * rho * ( rho_w - rho ) / ( rho_w * visc(n-1) );
    // New:
    u2_bc = 0.5 * c1(n-1) * g * L * ( rho * pow(H(n-1),2) - rho_w * pow(D,2) );


    // We include a shooting method. Let now u2(x0) be a degree of freedom 
    // so that we can sample it to match the BC at x = L. We perform RK4 with a given 
    // threshold in the solution at x = L. We also add a max number of iterations 
    // to scape loop and continue with the following time step.

    int c = 0;
    int s = 0;

    int n_c = 1; // 5

    ArrayXd u2_0_vec   = ArrayXd::Zero(n);
    ArrayXd u2_dif_vec = ArrayXd::Zero(n);

    while ( c < n_c ) // c < 10, 15.
    {
        // Allocate initial conditions.
        u_sol(0) = u1_0;
        u_sol(1) = u2_0;   // u2_0
        u1(0)    = u1_0;
        u2(0)    = u2_0;   // u2_0
        
        // Runge-Kutta 4th order iteration.
        for (int i=0; i<n-1; i++)
        {
            // Apply Runge-Kutta scheme. u1 = u_sol(0), u2 = u_sol(1).
            // k1(0) = du1/ds, k1(1) = du2/ds. L * 
            k1 = ds * du2_ds(u_sol(0), u_sol(1), dhds(i), visc(i), \
		    			        c1(i), c2(i), u_0, m, L, C_bed(i), \
                                    dH(i), d_visc(i), H(i));
            
            k2 = ds * du2_ds(u_sol(0) + 0.5 * k1(0), u_sol(1) + 0.5 * k1(1), \
                                dhds_h2(i), visc_h2(i), c1_h2(i), \
                                    c2_h2(i), u_0, m, L, C_bed_h2(i), \
                                        dH_h2(i), d_visc_h2(i), H_h2(i));
            
            k3 = ds * du2_ds(u_sol(0) + 0.5 * k2(0), u_sol(1) + 0.5 * k2(1), \
                                dhds_h2(i), visc_h2(i), c1_h2(i), \
                                    c2_h2(i), u_0, m, L, C_bed_h2(i), \
                                        dH_h2(i), d_visc_h2(i), H_h2(i));
            
            k4 = ds * du2_ds(u_sol(0) + k3(0), u_sol(1) + k3(1), dhds(i+1), \
                                visc(i+1), c1(i+1), c2(i+1), u_0, \
                                m, L, C_bed(i+1), dH(i+1), d_visc(i+1), H(i+1));

            u_sol = u_sol + pre * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            
            u1(i+1) = max(0.0, u_sol(0));
            u2(i+1) = u_sol(1); 
        }


        // Update IVP perturbation.
        u2_dif_now = u2(n-1) - u2_bc;
        vareps     = 2.0e-8;           // Perturbation in u2_0. 1.0e-8
        tol        = 1.0e-7;           // 1.0e-7

        // See iteration plots. u2_0 is always zero. \
        Thus, u2_dif increases with time. Maybe we can just sample a range of \
        u2_0 for every t and get the value that minimizes u2_dif.

        /*if ( t > t_eq )
        {
            if ( u2_dif_now < - tol )
            {
                u2_0 = u2_0 + vareps;
            }
            else if ( u2_dif_now > tol )
            {
                u2_0 = max(u2_0 - vareps, u_min);
            }
        }
        else
        {
            break;
        }*/

        // Save initial condition and difference in current iteration.
        u2_dif_vec(c) = u2_dif_now;
        u2_0_vec(c)   = u2_0;

        c = c + 1;
    }

    // Update shear stress from current velocity.
    tau_b = C_bed * pow(u1, m);
    
    // Allocate solutions.
    out.row(0) = u1;
    out.row(1) = u2;
    out.row(2) = tau_b;
    out.row(3) = c2 * dhds;
    out(4,0)   = D;
    out(4,1)   = u2_bc;
    out(4,2)   = u2_dif_now;
    
    // Shooting convergence.
    out.row(5) = u2_0_vec;
    out.row(6) = u2_dif_vec;
    
    return out;
}



MatrixXd vel_solver(ArrayXd H, double ds, double ds_inv, int n, ArrayXd visc, \
                    ArrayXd bed, double rho, double g, double L, ArrayXd C_bed, \
                    ArrayXd tau_b, double t, double u2_RK, ArrayXd beta, double A_ice)
{
    ArrayXd u2(n), dhds(n), visc_H(n), c1(n), c2(n), h(n), \
            A(n), B(n), C(n), F(n);

    MatrixXd out(5,n);

    double D, u2_bc, ds_inv_2, L_inv, gamma;

    L_inv    = 1.0 / L;
    ds_inv_2 = pow(ds_inv, 2);
    gamma    = 4.0 * ds_inv_2 * pow(L_inv, 2); // Factor 2 difference from Vieli and Payne (2005).

    // Defined for convenience.
    visc_H = visc * H;
    h      = bed + H;           // Ice surface elevation.

    ///////////////////////////////////////
    ///////////////////////////////////////
    // Staggered grid (Vieli and Payne solutions, appendix).

    for (int i=1; i<n-1; i++)
    {
        // Surface elevation gradient. Centred stencil.
        dhds(i) = 0.5 * ( H(i) + H(i+1) ) * ( h(i+1) - h(i) );

        // Diagonal, B; lower diagonal, A; upper diagonal, C.
        A(i) = visc_H(i);
        B(i) = - gamma * ( visc_H(i) + visc_H(i+1) ) - beta(i);
        C(i) = visc_H(i+1);
    }

    // Derivatives at the boundaries O(x).
    dhds(0)   = 0.5 * ( H(0) + H(1) ) * ( h(1) - h(0) );
    dhds(n-1) = H(n-1) * ( h(n-1) - h(n-2) ); // without a factor 2???
    //dhds(n-1) = H(n-1) * 0.5 * ( 3.0 * h(n-1) - 4.0 * h(n-2) + h(n-3) );

    // Tridiagonal boundary values. Correct????
    A(0) = 0.0;
    B(0) = - gamma * ( visc_H(0) + visc_H(1) ) - beta(0);
    C(0) = visc_H(1);

    A(n-1) = visc_H(n-1);
    B(n-1) = - gamma * visc_H(n-1) - beta(n-1);
    C(n-1) = 0.0;
    
    F = rho * g * dhds * L_inv * ds_inv;

    A = gamma * A;
    C = gamma * C;

    // Grounding line sigma = 1 (x = L). 
    D = abs( min(0.0, bed(n-1)) );   

    // Lateral boundary condition (Greve and Blatter Eq. 6.64).
    //cout << "\n visc(n-1) = " << visc(n-1);
    // Now:
    //u2_bc = 0.125 * g * H(n-1) * L * rho * ( rho_w - rho ) / ( rho_w * visc(n-1) );
    // Old:
    //u2_bc = 0.5 * c1(n-1) * g * L * ( rho * pow(H(n-1),2) - rho_w * pow(D,2) );

    // Pattyn.
    u2_bc = L * A_ice * pow( 0.25 * ( rho * g * H(n-1) * (1.0 - rho / rho_w) ), n_gln);

    // TRIDIAGONAL SOLVER.
    u1 = tridiagonal_solver(A, B, C, F, n);

    // Boundary conditions.
    // Ice divide: symmetry x = 0.
    u1(0)   = - u1(1);
    u1(n-1) = u1(n-2) + ds * u2_bc;

    // Centred derivative to calculate viscosity.
    for (int i=1; i<n-1; i++)
    {
        // Ensure positive velocity here? It solves the problem!
        if ( u1(i) < 0.0 )
        {
            u1(i) = u1(i+1); // 0.5 factor ?
        }

        u2(i) = 0.5 * ( u1(i+1) - u1(i-1) );
    }   
    u2(0)   = u1(1) - u1(0);
    u2(n-1) = u1(n-1) - u1(n-2);

    // Sigma coordinates transformation.
    u2 = abs(u2) / (ds * L);

    // Allocate solutions.
    out.row(0) = u1;
    out.row(1) = u2;
    out.row(2) = F;
    //out.row(3) = tau_b;
    out(4,0)   = D;
    out(4,1)   = u2_bc;
    
    return out;
}


// RUN FLOW LINE MODEL.
// Driver method
int main()
{
    cout << " \n h = " << ds;
    cout << " \n n = " << n;
    cout << " \n tf = " << tf;

    // Call nc write function.
    f_nc(n, n_z);

    // Wall time for computational speed.
    auto begin = std::chrono::high_resolution_clock::now();

    // Initilize first iteration values.
    for (int i=0; i<n; i++)
    {
        // Initial ice thickness H0.
        //H(i) = 1.8e3 * (1.0 - pow(sigma(i), 2.5)) + 1.2e3;

        // Surface mass accumulation.
        //S(i) = 0.0;
        //S(i)    = 0.30 * pow(sigma(i), 2) ;      // 0.31 * pow(sigma(i), 2)

        // MISMIP EXPERIMENTS.
        H(i) = 10.0;             // Initial ice thickness [m].
        S(i) = 0.3;              // Snow accumulation [m/yr].
    }
    
    // Units consistency.
    //S = S / sec_year; 

    // Temperature initial conditions (-25ºC).
    //ArrayXXd theta = ArrayXXd::Constant(n, n_z, 248.0);


    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    // MISMIP experiments.

    // Constant friction coeff. 7.624e6 [Pa m^-1/3 s^1/3]
    ArrayXd C_bed = ArrayXd::Constant(n, 7.0e6);    // [Pa m^-1/3 s^1/3]
    C_bed = C_bed / pow(sec_year, m);                 // [Pa m^-1/3 yr^1/3]
    
    // Viscosity from constant A value. u2 = 0 initialization. \
    // 4.6416e-24, 2.1544e-24. [Pa^-3 s^-1] ==> [Pa^-3 yr^-1]
    A = 4.23e-25 * sec_year;               
    B = pow(A, ( -1 / n_gln ) );

    // We assume a constant viscosity in the first iteration.
    ArrayXd visc = ArrayXd::Constant(n, 1.0e13);    // [Pa s]
    visc = visc / sec_year;                         // [Pa yr]


    // Implicit initialization.
    beta = ArrayXd::Constant(n, 5.0e3);    // [Pa yr / m]

    // Exp 3 hysteresis forcing.
    // Rate factor.
    /*A_s << 5.0e-25, 4.0e-25, 3.0e-25, 2.5e-25, 2.0e-25, 1.5e-25, 1.0e-25, 5.0e-26, 2.5e-26, 
           5.0e-26, 1.0e-25, 1.5e-25, 2.0e-25, 2.5e-25, 3.0e-25, 3.5e-25, 4.0e-25, 4.25e-25, 
           4.5e-25, 4.75e-25, 5.0e-25; */

    // Time length for a certain A value. 3.0e4,
    /*t_s << 3.0e4, 6.0e4, 9.0e4, 11.5e4, 14.0e4, 16.5e4, 19.0e4, 21.5e4, 24.0e4, 
           26.5e4, 29.0e4, 31.5e4, 34.0e4, 36.5e4, 39.0e4, 41.5e4, 44.0e4, 46.5e4, 
           49.0e4, 51.5e4, 54.0e4;*/

    /*t_s << 3.0e4, 6.0e4, 9.0e4, 10.5e4, 12.0e4, 13.5e4, 15.0e4, 18.0e4, 21.0e4, 
           22.5e4, 24.0e4, 27.0e4, 30.0e4, 33.0e4, 34.5e4, 36.0e4, 37.5e4, 39.0e4, 
           40.5e4, 43.5e4, 46.5e4;*/

    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////

   
    // Counter to write solution.
    int c = 0;

    // Counter for MISMIP ice factor A forcing.
    int c_s = 0;

    // Initialize time.
    t  = 0.0;

    // Main loop.
    while (t < tf)
    {
        // EXPERIMENT 3 MISMIP.

        // Update rate factor value.
        /*if ( t <= t_s(c_s) )
        {
            A = A_s(c_s) * sec_year;
        }
        else
        {
            // Ensure we don't exceed time vector lenght.
            c_s = min(c_s + 1, n_s - 1);

            A = A_s(c_s) * sec_year;
            //cout << " \n A = " << A;
        }
        
        // Ice hardness.
        B = pow(A, ( -1 / n_gln ) );*/

        // Update bedrock with new domain extension L.
        bed = f_bed(sigma, L, n, exp);

        // Friction coefficient from temperaure.
        //C_bed = f_C_bed(theta, theta_max, C_thaw, C_froz, n);

        // First, explcit scheme. Then, implicit using explicit guess.
        if (t < t_eq)
        {
            // First iterations using an explicit scheme RK-4.
            //cout << " \n Runge-Kutta \n ";
            u = rungeKutta(u1_0, u2_0, u_0, H, ds, ds_inv, n, \
                                visc, bed, rho, g, L, m, C_bed, t_eq);

            // Allocate variables.
            u1    = u.row(0);
            u2    = u.row(1);
            tau_b = u.row(2);
            tau_d = u.row(3);
            D     = u(4,0);
            u2_bc = u(4,1);
            u2_dif = u(4,2);
            u2_0_vec   = u.row(5);
            u2_dif_vec = u.row(6);

            //cout << "\n u1 = " << u1;
        }
        else
        {
            // Implicit velocity solver.
            // Picard iteration for non-linear viscosity and beta.
            
            // Update basal friction with previous step velocity. Out of Picard iteration?
            //tau_b = C_bed * pow(u1, m);
            tau_b = beta * u1;
            // Symmetry.
            tau_b(0) = tau_b(1);

            // Flux definition. Staggered grid.
            for (int i=0; i<n-1; i++)
            {
                q(i) = u1(i) * 0.5 * ( H(i+1) + H(i) );
            }
            
            // GL flux definition (Vieli and Payne, 2005).
            if ( calv == 0 )
            {
                q(n-1) = u1(n-1) * H(n-1);
            } 
            // Additional calving term (Christian et al., 2022).
            else if ( calv == 1 )
            {
                q(n-1) = ( u1(n-1) + m_dot ) * H(n-1);
            }
            

            // Picard initialization.
            error    = 1.0;
            c_picard = 0;

            while (error > picard_tol & c_picard < n_picard)
            {
                // Save previous iteration solution.
                u1_old_1 = u1;

                // Implicit solver.
                u = vel_solver(H, ds, ds_inv, n, visc, bed, rho, g, L, \
                               C_bed, tau_b, t, u2_bc, beta, A);

                // Allocate variables.
                u1    = u.row(0);
                u2    = u.row(1);
                //tau_d = u.row(2);
                D     = u(4,0);
                u2_bc = u(4,1);

                // Beta definition: tau_b = beta * u.
                beta    = C_bed * pow(u1, m - 1.0);
                beta(0) = beta(1);

                // Current error (vector class required to compute norm). 
                // Eq. 12 (De-Smedt et al., 2010).
                c_u1_1 = u1 - u1_old_1;
                u1_vec = u1;
                error  = c_u1_1.norm() / u1_vec.norm();
                
                // New relaxed Picard iteration. Pattyn (2003). 
                // Necessary to deal with the nonlinear velocity dependence
                // in both viscosity and beta.
                if (c_picard > 0)
                {
                    // Difference in iter i-2.
                    c_u1_2   = u1_old_1 - u1_old_2;
                    c_u1_dif = c_u1_1 - c_u1_2;
                    
                    omega = acos( c_u1_1.dot(c_u1_2) / \
                                 ( c_u1_1.norm() * c_u1_2.norm() ) );
                    

                    // De Smedt et al. (2010). Eq. 10.
                    if (omega <= omega_1 || c_u1_1.norm() == 0.0)
                    {
                        //mu = 2.5; // De Smedt.
                        mu = 1.0; // To avoid negative velocities?
                        //mu = 0.7;
                    }
                    else if (omega > omega_1 & omega < omega_2)
                    {
                        mu = 1.0; // De Smedt.
                        //mu = 0.7;
                    }
                    else
                    {
                        mu = 0.5; // De Smedt.
                        //mu = 0.7;
                    }

                    // New velocity guess based on updated omega.
                    u1 = u1_old_1 + mu * c_u1_1.array();

                    // Update beta with new u1.
                    beta    = C_bed * pow(u1, m - 1.0);
                    beta(0) = beta(1); // Symmetry

                    // Update viscosity with new u2 field.
                    for (int i=1; i<n-1; i++)
                    {
                        // Centred stencil.
                        u2(i) = 0.5 * ( u1(i+1) - u1(i-1) );
                    }
                    u2(0)   = u1(1) - u1(0);
                    u2(n-1) = u1(n-1) - u1(n-2);

                    u2 = abs(u2) / (ds * L);
                    visc = f_visc(u2, B, n_exp, eps, L, n);
                    
                }

                // Update multistep variables.
                u1_old_2 = u1_old_1;

                // Number of iterations.
                c_picard = c_picard + 1;
            }
        }

        // CONSISTENCY CHECK.
        // Search for NaN values.
        // Count number of true positions in u1.isnan().
 
        if ( u1.isNaN().count() != 0 )
        {
            cout << "\n NaN found.";
            cout << "\n Saving variables in nc file. \n ";

            // Save previous iteration solution (before NaN encountered).
            f_write(c, u1_old_1, u2,  H, visc, S, tau_b, beta, tau_d, bed, \
                    C_bed, u2_dif_vec, u2_0_vec, L, t, u2_bc, u2_dif, \
                    error, dt, c_picard, mu, omega, theta, A, dL_dt);

            // Close nc file. 
            if ((retval = nc_close(ncid)))
            ERR(retval);
            printf("\n *** %s file has been successfully written \n", \
                    FILE_NAME);
                
            // Abort flowline.
            return 0;
        }
        
        // Update timestep from velocity field.
        // Courant-Friedrichs-Lewis condition (factor 1/2, 3/4).
        //dt_CFL = 0.01 * ds * L / u1.maxCoeff();  
        //dt     = min(dt_CFL, dt_max);
        dt = dt_max;

        // Update grounding line position with new velocity field.
        dL_dt = f_L(u1, H, S, H_c, dt, L, ds, n, bed, rho, rho_w, q);
        L_new = L + dL_dt * dt;

        // Save solution with desired output frequency.
        if (c == 0 || t > a(c))
        {
            cout << "\n t = " << t;

            // Write solution in nc.
            f_write(c, u1, u2,  H, visc, S, tau_b, beta, tau_d, bed, \
                    C_bed, u2_dif_vec, u2_0_vec, L, t, u2_bc, u2_dif, \
                    error, dt, c_picard, mu, omega, theta, A, dL_dt);

            c = c + 1;
        }  

        // Update ice viscosity with new u2 field.
        visc = f_visc(u2, B, n_exp, eps, L, n);
        //filt = gaussian_filter(visc, zeros, 4.0, L, ds, n);

        // Evaluate calving front thickness from tau_b field.
        //H_c = f_calv(tau_b, D, rho, rho_w, g, bed);

        // Integrate ice thickness forward in time.
        H = f_H_flux(u1, H, S, sigma, dt, ds, ds_inv, n, \
                     L_new, L, H_c, D, rho, rho_w, dL_dt, bed, q, 0);


        // Integrate Fourier heat equation.
        /*theta_now = f_theta(theta, u1, H, tau_b, theta_max, kappa, \
                            k, dt, G_k, ds, L, n, n_z);
        theta = theta_now;*/

        // Update multistep variables.
        L_old = L;
        L     = L_new;

        // Update time.
        t = t + dt;
    }
    

    // Running time (measures wall time).
    auto end     = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    
    // Print computational time.
    printf("\n Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    printf("\n Computational speed: %.3f kyr/hr.\n", \
            60 * 60 * (1.0e-3 * tf) /  (elapsed.count() * 1e-9) );

    // Close nc file. 
    if ((retval = nc_close(ncid)))
    ERR(retval);
 
    printf("\n *** %s file has been successfully written \n", FILE_NAME);

    return 0;
}