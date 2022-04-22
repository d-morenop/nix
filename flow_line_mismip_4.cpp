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

// Our flow line model uses netcdf and Eigen libraries. Make sure both are installed.
// Eigen: https://eigen.tuxfamily.org.
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

// Float can contain up to seven digits in total, not just following the decimal point.



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
float const sec_year = 3.154e7;              // Seconds in a year.
//float const gam      = 0.2;                // Robert-Asselin filter parameter. 0.3
float const u_min    = 0.0;
float const u_max    = 800.0 / sec_year;

// PHYSICAL CONSTANTS.
float const u_0   = 150.0 / sec_year;
float const m     = 1.0 / 3.0;              // 0.2, 1.0/3.0
float const g     = 9.8;                    // 9.81
float const rho   = 900.0;
float const rho_w = 1000.0;                 // 1028.8

// BEDROCK PARAMETRIZATION: f_C_bed.
float const C_thaw = 7.624e6;                  // 1.75e6 [Pa m^-1/3 s^1/3].
float const C_froz = 7.624e6;                  // 2.0e6 [Pa m^-1/3 s^1/3].

// ICE VISCOSITY: f_visc.
float const n_gln = 3.0;
float const n_exp = (1.0 - n_gln) / (2.0 * n_gln);    // Yelmo
//float const n_exp = (1.0 / n_gln) - 1.0;                // Schoof definitions.
//float const A     = 4.9e-25;                          // 4.9e-25 (-10ºC). [Pa³ / s] (Greve and Blatter, 2009)
//float const A     = 1.0e-30;                          // 1.0e-25
//float const B     = pow(A, ( -1 / n_gln ) );          // Pa³ / s  
//double const eps  = pow(1.0e-6, 2);                   // 1.0e-12

// eps is fundamental for GL, velocities, thickness, etc.
// We need to tune this paramter to obtain a similar profile to MISMIP exp1.
// A ~ 10^-24: 1.0e-14. A ~ 10^-25; 1.0e-16.
float const eps  = 1.0e-14;                            // Final: 1.0e-14. Current 1.0e-7. Yelmo: 1.0e-6. 2.5e-9 good GL but bad H.

// BASAL FRICTION.
float const tau_b_min = 30.0e3;

// REPEAT LONG SIMULATION TO SEE BEHAVIOUR!!

// DOMAIN DEFINITION.
float L     = 702.3e3;                    // Grounding line position [m] (1.2e6)
float L_old = 702.3e3;
float L_new;
float const t0    = 0.0;                // Starting time [s].
// SIMULATIONS STOP BEFORE REACHING ENDING TIME!!
float const tf    = 100.0e3 * sec_year;  // 75.0e3. 5.0e3. 1.0e5. Ending time [yr] * [s/yr]
//float tf    = 75.0e3 * sec_year; 
float t     = t0;                       // Time initialization [s].
float t_plot;
float dt;                               // Time step [s].
float dt_CFL;                           // Courant-Friedrichs-Lewis condition
float const dt_max = 5.0 * sec_year;    // Maximum time step = 10 years [s].
float const t_eq = 5.0 * sec_year;     // 20.0. Length of explicit scheme. try 50?
float const t_bc = 10.0 * sec_year;    // 1.0e3. Implicit scheme spin-up. End of u2_bc equilibration.
float dt_plot;

int const t_n = 30;                     // Number of output frames. 30.

ArrayXf a = ArrayXf::LinSpaced(t_n, t0, tf);      // Time steps in which the solution is saved. 


// COORDINATES.
int const n = 1000;                     // Number of horizontal points. 200, 500, 2000
float const ds = 1.0 / n;               // Normalized spatial resolution.
float const ds_inv = n;

int const   n_z = 10;                   // Number vertical layers. 10, 20.
float const dz = 1.0 / n_z;             // Normalized vertical resolution.
float const dz_inv = n_z;

ArrayXf sigma = ArrayXf::LinSpaced(n, 0.0, 1.0);    // Dimensionless x-coordinates. 

// Auxiliar definitions.
ArrayXf zeros = ArrayXf::Zero(n);


// THERMODYNAMICS.
float const k = 2.0;              // Thermal conductivity of ice [W / m · ºC].
float const G = 0.05;             // Geothermal heat flow [W / m^2] = [J / s · m^2].
float const G_k = G / k;          // [K / m] 
float const kappa = 1.4e-6;       // Thermal diffusivity of ice [m^2/s].
float const theta_max = 273.15;   // Max temperature of ice [K].

// CALVING
float D;                               // Depth below the sea level [m].
float u2_bc;                           // Boundary condition on u2 = du1/dx.
float u2_dif;                          // Difference between analytical and numerical.

// Runge-Kutta boundary conditions.
float u1_0 = 0.0;                      // Velocity in x0 (i.e., u(x0)).
float u2_0 = 0.0;                      // Velocity first derivative in x0 (i.e., du/dx(x0)).

// MISMIP EXPERIMENT CHOICE.
int const mismip = 1;
float A, B;

// PICARD ITERATION
float error;                           // Norm of the velocity difference between iterations.
float const picard_tol = 1.0e-4;       // 1.0e-5. Convergence tolerance within Picard iteration.
int const n_picard = 5;                // Max number iter. Good results: 1 is enough for convergence! (10, 15)
int c_picard;
float alpha, omega; //alpha_1, alpha_2, mu;


// PREPARE VARIABLES.
ArrayXf H(n);                        // Ice thickness [m].
ArrayXf H_now(n);                    // Current ice thickness [m].
ArrayXf H_old(n);                    // Previous ice thickness [m].
ArrayXf u1(n);                       // Velocity [m/s].
ArrayXf u2(n);                       // Velocity first derivative [1/s].
ArrayXf bed(n);                      // Bedrock elevation [m].
ArrayXf C_bed(n);                    // Friction coefficient [Pa m^-1/3 s^1/3].
ArrayXf visc(n);                     // Ice viscosity [Pa·s].
ArrayXf visc_new(n);                 // Current ice viscosity [Pa·s].
ArrayXf S(n);                        // Surface accumulation equivalent [mm/day].
ArrayXf H_c(n);                      // Maxium ice thickness permitted at calving front [m].
ArrayXf u1_plot(n);                  // Saved ice velocity [m/yr]
ArrayXf u2_plot(n);                  // Saved ice velocity derivative [1/yr]
ArrayXf tau_b(n);                    // Basal friction [Pa]
ArrayXf tau_d(n);                    // Driving stress [Pa]
ArrayXf filt(n);                     // 
ArrayXf u1_old(n);  
ArrayXf u2_old(n);  
ArrayXf u2_0_vec(n);                 // Ranged sampled of u2_0 for a certain iteration.
ArrayXf u2_dif_vec(n);               // Difference with analytical BC.
VectorXf dif_iter(n);                 // Velocity difference between two consequtive iterations [m/s].
VectorXf u2_vec(n); 
VectorXf c_s_1(n);                   // Correction vector Picard relaxed iteration.
VectorXf c_s_2(n);
//VectorXf u2_vec(n);


//ArrayXf smth(n);                     // Smooth field.
//MatrixXf smth(3,n);                // Smooth field.
MatrixXf u(7,n);                     // Matrix output.

ArrayXXf theta(n,n_z);               // Temperature field [K].
ArrayXXf theta_now(n,n_z);           // Current temperature field [K].
//MatrixXf u_old(1,n);                 // Store previous iteration for relaxation.




////////////////////////////////////////////////////
////////////////////////////////////////////////////
// TOOLS. USEFUL FUNCTIONS.

ArrayXf gaussian_filter(ArrayXf w, ArrayXf zeros, \
                        float sigma, float L, float ds, int n)
{
    ArrayXf smth(n), summ(n);
    //ArrayXf summ = ArrayXf::Zero(n);

    float x, y;
    float h_L = ds * L;
    //float h_L = ds;
    float A = 1.0 / (sqrt(2.0 * M_PI) * sigma);

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


// half_step(dhdx, visc, dH, d_visc, c1, c2, C_bed, H, n);
MatrixXf half_step(ArrayXf dhdx, ArrayXf visc, \
                   ArrayXf dH, ArrayXf d_visc, ArrayXf c1, \
                   ArrayXf c2, ArrayXf C_bed, ArrayXf H, int n)
{
    // Function that return a vector of length n-1 with the average of 
    // every contiguous elements.
    MatrixXf x(n+1,8), x_n2(n+1,8), half_n2(n+1,8), half(n-1,8); 
    
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


MatrixXf deriv_sigma(ArrayXf x, ArrayXf y, ArrayXf z,\
                     int n, float ds_inv, float L)
{   
    MatrixXf grad(n, 3), x_min(n+2, 3), \
             x_plu(n+2, 3), grad_n2(n+2, 3);
    
    //float dx = L * ds;           // ds = 1 / n.
    //float dx_inv = 1.0 / dx;
 
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


ArrayXf tridiagonal_solver(ArrayXf A, ArrayXf B, ArrayXf C, \
                           ArrayXf F, int n, float u2_bc, float t, float u2_RK)
{
    ArrayXf P(n), Q(n), u(n);
    float m;
    int j;
    
    // This allows us to perform O(n) iterations, rather than 0(n³).
    // A subdiagonal, B diagonal and C uppder-diagonal.
    
    // As the matrix is built.
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


    ////////////////////////////////////////
    // SOLVE THIS!!!!!!!!!!!!!
    // From notes: u(n-1) = Q(n-1).
    // Boundary condition at the GL on u2.
    u(n-1) = u2_bc;             


    // Smooth transition to u2_bc (since explicit scheme doesn't account for this).
    /*if (t <= t_bc)
    {
        u(n-1) = u2_RK + ( u2_bc - u2_RK ) * (t - t_eq) / (t_bc - t_eq);
    }
    else if (t > t_bc)
    {
        u(n-1) = u2_bc;
    }*/
    
    
    // Back substitution (n+1 is essential).
    // i = 2 --> j = n-2 --> u(n-2)
    // i = n --> j = 0   --> u(0)
    for (int i=2; i<n+1; i++)
    {
        j = n - i;
        u(j) = P(j) * u(j+1) + Q(j);
    }

    // Do we need this to ensure stability?
    //u(n-2) = u2_bc;

    return u;
}


////////////////////////////////////////////////////
////////////////////////////////////////////////////
// Flow line functions.

ArrayXf f_bed(ArrayXf sigma, float L, int n, int exp)
{
    ArrayXf bed(n), x_scal(n);

    // Same bedrock as Schoof (2007).
    // sigma = x / L. L in metres!
    x_scal = sigma * L / 750.0e3; 

    // MISMIP experiments bedrock.
    // Inverse sign to get a decreasing bedrock elevation.
    if (mismip == 1)
    {
        bed = 720.0 - 778.5 * x_scal;
    }
    else if (mismip == 3)
    {
        bed = ( 729.0 - 2148.8 * pow(x_scal, 2) + \
                        1031.72 * pow(x_scal, 4) + \
                        - 151.72 * pow(x_scal, 6) );
    }

    return bed;
}



float f_L(ArrayXf u1, ArrayXf H, ArrayXf S, ArrayXf H_c, \
          float dt, float L, float ds, int n, \
          ArrayXf bed, float rho, float rho_w)
{
    ArrayXf q(n);
    float num, den, dL_dt, L_new;
    
    q   = u1 * H;

    // Accumulation minus flux.
    //num = L * ds * S(n-1) - ( q(n-1) - q(n-2) );
    num = - L * ds * S(n-1) + ( q(n-1) - q(n-2) );

    // Yield strength ice. Following Bassis et al. (2017).
    //den = H_c(n-1) - H_c(n-2) - ( H(n-1) - H(n-2) );

    // Purely flotation condition (Following Schoof, 2007).
    // Sign before db/dx is correct. Otherwise, it migrates uphill.
    //den = H(n-1) - H(n-2) - ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );
    den = H(n-1) - H(n-2) + ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );

    // Integrate grounding line position forward in time.
    dL_dt = num / den;
    L_new = L + dL_dt * dt;

    return L_new;
} 

// First order scheme. New variable sigma.
ArrayXf f_H_flux(ArrayXf u, ArrayXf H, ArrayXf S, ArrayXf sigma, \
                    float dt, float ds_inv, int n, float L_new, \
                    float L, float L_old, ArrayXf H_c, float D, \
                    float rho, float rho_w)
{
    ArrayXf H_now(n), sigma_L(n), q(n);
    float L_inv, H_gl, H_float, delta_L, delta_L_tilde;

    q       = u * H;
    L_inv   = 1.0 / L;
    sigma_L = sigma * L_inv;

    // Grounding line position change.
    delta_L = L_new - L_old;
    //delta_L_tilde = L - L_old;

    // Advection equation. Centred dH in the sigma_L term.
    for (int i=1; i<n-1; i++)
    {
        H_now(i) = H(i) + sigma_L(i) * 0.5 * delta_L * \
                            0.5 * ( H(i+1) - H(i-1) ) * ds_inv + \
                            - dt * ( q(i) - q(i-1) ) * L_inv * ds_inv + S(i) * dt;
    }

    // Ice thickness BC at sigma = 0. dH = 0.
    H_now(0) = H_now(1); 

    // Flux in the last grid point. .
    H_gl = H(n-1) + sigma_L(n-1) * 0.5 * delta_L * \
                         ( H(n-1) - H(n-2) ) * ds_inv + \
                        - dt * ( q(n-1) - q(n-2) ) * L_inv * ds_inv + S(n-1) * dt;


    // Terminus thickness given by flotation condition.
    //H_float    = ( rho_w / rho ) * D;
    // New. This gives nearly zero velocities:
    //H_now(n-1) = H_float;
    
    // Flotation condition as the lower bound.
    // Original:
    //H_now(n-1) = max(H_gl, H_float);

    // According to Bassis et al. (2017?)
    //H_now(n-1) = min(H_gl, H_float);

    // Appendix A1 in Schoof (2007) remarks a possible drift \
    from flotation condition. Then:
    H_now(n-1) = H_gl;

	return H_now; 
}


ArrayXf f_dhdx(ArrayXf dH, ArrayXf b, int n)
{
    ArrayXf dhdx(n);

    // Ice surface elevation term plus bedrock contribution. \
    Ice surface elevation h = H + b.
    dhdx = dH + b;

	return dhdx;
}


ArrayXf f_visc(ArrayXf u2, float B, float n_exp, \
                double eps, float L, int n)
{
    ArrayXf u2_L(n), visc(n), u2_square(n); 
    float A;

    // u2 comes from integration where derivatives are respect
    // to sigma. It needs transformation to x-derivative.

    // MISMIP experiments viscosity.
    u2_L      = pow( (abs(u2) / L) + eps, 2);
    visc      = 0.5 * B * pow(u2_L, n_exp);

    // Constant viscosity experiment:
    // ArrayXf visc = ArrayXf::Constant(n, 0.5e17); // 1.0e15, 0.5e17
    
	return visc;
}	

ArrayXf f_calv(ArrayXf tau, float D, \
               float rho, float rho_w, float g, ArrayXf bed)
{
    ArrayXf H_c(n), tau_tilde(n), bed_tilde(n);

    // Following Bassis et al. (2011; 2017)
    // Maximum ice thickness permitted at the calving front H_c.
    tau_tilde = tau / (rho * g);

    // Bassis and Walker (2012) Eq. 2.10, Bassis et al. (2017). 
    H_c = tau_tilde + sqrt( pow(tau_tilde, 2) \
                            + (rho_w / rho) * pow(bed, 2) );

    return H_c;
} 


ArrayXXf f_theta(ArrayXXf theta, ArrayXf u, ArrayXf H, ArrayXf tau_b, \
                float theta_max, float kappa, float k, float dt, float G_k, \
                float ds, float L, int n, int n_z)
{
    MatrixXf theta_now(n,n_z);
    ArrayXf dz(n), dz_2_inv(n), Q_f_k(n);

    float dx_inv;
 
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

ArrayXf f_C_bed(ArrayXXf theta, float theta_max, \
                float C_thaw, float C_froz, int n)
{
    ArrayXf C_bed(n), theta_norm(n);

    // Normalized basal temperature with pressure melting point [0,1].
    theta_norm = theta.block(0,0,n,1) / theta_max;
    C_bed      = C_thaw * theta_norm + C_froz * ( 1.0 - theta_norm );

    return C_bed;
}
	

Array2f du2_ds(float u_1, float u_2, float dh_dx, float visc,\
               float c_1, float c_2, float u_0, float m, float L, \
               float C_bed, float dH, float d_visc, float H)
{
    Array2f out;
    float du2, tau_b; 

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


MatrixXf rungeKutta(float u1_0, float u2_0, float u_min, float u_max, float u_0, \
                    ArrayXf H, float ds, float ds_inv, int n, ArrayXf visc, \
                    ArrayXf bed, float rho, float g, float L, float m, \
                    ArrayXf C_bed, float t_eq)
{
    ArrayXf u1(n), u2(n), dhds(n), dvisc_H(n), visc_dot_H(n), \
            c1(n), c2(n), h(n), dH(n), d_visc(n), \
            dhds_h2(n-1), d_visc_h2(n-1), visc_h2(n), H_h2(n), dH_h2(n), \
            c1_h2(n-1), c2_h2(n-1), u1_h2(n-1), u2_h2(n-1), C_bed_h2(n-1);

    MatrixXf dff(n,3), out(7,n), half(n-1,8);

    Array2f k1, k2, k3, k4, u_sol;
    float pre, D, u2_bc, vareps, tol, u2_dif, u2_dif_now;


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
    D = abs( min(u_min, bed(n-1)) );      // u_min is just float32 0.0.

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

    ArrayXf u2_0_vec   = ArrayXf::Zero(n);
    ArrayXf u2_dif_vec = ArrayXf::Zero(n);

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
            
            u1(i+1) = max(u_min, u_sol(0));
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



MatrixXf vel_solver(ArrayXf H, float ds, float ds_inv, int n, ArrayXf visc, \
                    ArrayXf bed, float rho, float g, float L, ArrayXf C_bed, \
                    ArrayXf tau_b, float t, float u2_RK)
{
    ArrayXf u2(n), dhds(n), visc_dot_H(n), c1(n), c2(n), h(n), \
            A(n), B(n), C(n), F(n);

    MatrixXf dff(n,1), out(5,n);

    float D, u2_bc, d_vis_H;

    // Defined for convenience.
    visc_dot_H = 4.0 * visc * H;
    c1         = 1.0 / visc_dot_H;      
	c2         = rho * g * H;
    h          = bed + H;           // Ice surface elevation.

    ///////////////////////////////////////
    ///////////////////////////////////////
    // SOLVING FOR U2 INSTEAD OF U1.

    for (int i=1; i<n-1; i++)
    {
        // Surface elevation gradient. Centred stencil.
        dhds(i) = h(i+1) - h(i-1);

        // Diagonal, B; lower diagonal, A; upper diagonal, C.
        B(i) = 0.0;
        A(i) = visc_dot_H(i-1); 
        C(i) = visc_dot_H(i+1);
    }

    // Derivatives at the boundaries O(x).
    dhds(0)   = 2.0 * ( h(1) - h(0) );
    dhds(n-1) = 2.0 * ( h(n-1) - h(n-2) );
    
    // FINISH MISMIP EXPERIMENTS!!!
    // Boundary values vectors.
    // Original (is this correct?): \
    We need to tune eps as we change A?
    /*B(0)   = - 2.0 * visc_dot_H(0) * ds_inv;
    B(n-1) = 2.0 * visc_dot_H(n-1) * ds_inv;*/
    // New:
    B(0)   = - 2.0 * visc_dot_H(0);
    B(n-1) = 2.0 * visc_dot_H(n-1);

    // New.
    C(0)   = 2.0 * visc_dot_H(1);
    C(n-1) = 0.0;
    A(0)   = 0.0;
    A(n-1) = 2.0 * visc_dot_H(n-2); 

    // Vectors in tridiagonal matrix.
    A = - 0.5 * A * ds_inv; 
    C = 0.5 * C * ds_inv;
    B = 0.5 * B * ds_inv;

    dhds = 0.5 * dhds * ds_inv;

    // mi amor, no te preocupes. qué lo vas a hacer genial.

    // Stress balance: driving - friction.
    F = c2 * dhds + tau_b * L;

    //F = ArrayXf::Constant(n, -1.0e10); // Must be negative. -1.0e10

    // Grounding line sigma = 1 (x = L). u_min is just float32 0.0.
    D = abs( min(u_min, bed(n-1)) );   

    // HERE IS THE ISSUE WITH THE EQUILIBIRUM STATE.
    // DIFFERENT EQUILIBIRUM PROFILES ARE FOUND FOR EACH BC!!!
    // Equivalent (Greve and Blatter Eq. 6.64).
    //u2_bc = 0.125 * g * H(n-1) * L * rho * ( rho_w - rho ) / ( rho_w * visc(n-1) );
    // New:
    u2_bc = 0.5 * c1(n-1) * g * L * ( rho * pow(H(n-1),2) - rho_w * pow(D,2) );

    // TRIDIAGONAL SOLVER.
    u2 = tridiagonal_solver(A, B, C, F, n, u2_bc, t, u2_RK);

    // BOUNDARY CONDITIONS.
    // Ice divide in sigma = 0.
    u1(0) = 0.0;

    // Direct explicit integration to obtain u1 from u2. Forward integration.
    /*for (int i=0; i<n-1; i++)
    {
        // Modified Runge-Kutta order 2. \
        This way we consider BC imposed in u2.
        // Old (this is wrong):
        //u1(i+1) = u1(i) + ds * 0.5 * ( u2(i) + u2(i+1) );
        // New: no numerical oscillations?
        u1(i+1) = u1(i) + ds * u2(i);
        u1(i+1) = max(u_min, u1(i+1));
    }*/

    // New: centred scheme.
    for (int i=1; i<n-1; i++)
    {
        u1(i+1) = u1(i-1) + ds * 2.0 * u2(i);
        u1(i+1) = max(u_min, u1(i+1));
    }
    u1(1) = u1(0) + ds * u2(0);

    // Allocate solutions.
    out.row(0) = u1;
    out.row(1) = u2;
    out.row(2) = c2 * dhds;
    out.row(3) = tau_b;
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
    cout << " \n tf = " << tf / sec_year;

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
        S(i) = 0.3;              // Snow accumulation [m/s].
    }
    
    // Units consistency.
    S = S / sec_year; 

    // Temperature initial conditions (-25ºC).
    //ArrayXXf theta = ArrayXXf::Constant(n, n_z, 248.0);


    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    // MISMIP experiments.

    // Constant friction coeff.
    ArrayXf C_bed = ArrayXf::Constant(n, 7.624e6); //7.624e6, 4.624e6

    // Viscosity from constant A value. u2 = 0 initialization.
    // AS WE DECREASE A, THE ICE SHEET LOSES MASS EARLIER.
    A    = 4.6416e-24;               // 4.6416e-24, 2.1544e-24
    B    = pow(A, ( -1 / n_gln ) );
    u2   = pow(eps, 2);
    visc = 0.5 * B * pow(u2, n_exp);

    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////

   
    int c = 0;
    t  = 0.0;

    while (t < tf)
    {
        // Update bedrock with new domain extension L.
        bed = f_bed(sigma, L, n, 1);

        // Friction coefficient from temperaure.
        //C_bed = f_C_bed(theta, theta_max, C_thaw, C_froz, n);

        // First, explcit scheme. Then, implicit using explicit guess.
        if (t < t_eq)
        {
            // First iterations using an explicit scheme RK-4.
            //cout << " \n Runge-Kutta \n ";
            u = rungeKutta(u1_0, u2_0, u_min, u_max, u_0, H, ds, ds_inv, n, \
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
        }
        else
        {
            // Implicit velocity solver.
            // Solution from explicit scheme used as initial guess to 
            // ensure convergence. Picard iteration.
            // n_picard = 10 and error > 0.5e-6 seems reasonable with tau_b_min = 2.5e2.
            
            c_picard = 0;

            error = 1.0;
            while (error > picard_tol & c_picard < n_picard)
            {
                // Save previous iteration solution.
                u1_old = u1;
                u2_old = u2;

                // Call implicit solver.
                u = vel_solver(H, ds, ds_inv, n, visc, \
                               bed, rho, g, L, C_bed, tau_b, t, u2_bc);

                // Allocate variables.
                u1    = u.row(0);
                u2    = u.row(1);
                tau_d = u.row(2);
                //tau_b = u.row(3);
                D     = u(4,0);
                //u2_bc = u(4,1);

                // Relaxation within Picard iteration. This may cause instabilities?
                /*u2 = 0.3 * u2_old + 0.7 * u2;
                u1 = 0.3 * u1_old + 0.7 * u1;*/

                u2 = 0.5 * u2_old + 0.5 * u2;
                u1 = 0.5 * u1_old + 0.5 * u1;

                // Update viscosity.
                visc = f_visc(u2, B, n_exp, eps, L, n);

                // Current error (vector class required to compute norm). 
                // Eq. 12 De-Smedt et al. (2010).
                //dif_iter = u1 - u1_old;
                dif_iter = u2 - u2_old;
                //error    = dif_iter.norm();

                u2_vec = u2;
                error  = dif_iter.norm() / u2_vec.norm();
                

                // New relaxed Picard iteration. Pattyn  (2003). 
                /*if (c_picard > 0)
                {
                    c_s_1 = u1 - u1_old_1;
                    c_s_2 = u1_old_1 - u1_old_2;
                    c_s_dif = c_s_1 - c_s_2;

                    alpha = c_s_2.norm() / c_s_dif.norm();
                    
                    omega = acos( c_s_1.dot(c_s_2) / \
                                 ( c_s_1.norm() * c_s_2.norm() ) );

                    // De Smedt et al. (2010) Eq. 10.
                    alpha_1 = 0.125 * M_PI;
                    alpha_2 = (19.0 / 20.0) * M_PI;

                    if (alpha <= alpha_1)
                    {
                        mu = 2.5;
                    }
                    else if (alpha > alpha_1 & alpha < alpha_2)
                    {
                        mu = 1.0;
                    }
                    else
                    {
                        mu = 0.5;
                    }

                    // New guess based on updated alpha.
                    u2 = ( 1.0 - alpha ) * u2_old + alpha * u2;
                    u1 = ( 1.0 - alpha ) * u1_old + alpha * u1;
                }*/

                // Update basal friction with current step velocity u1.
                tau_b = C_bed * pow(u1, m);

                // Min tau_b to avoid losing ice at equilibirum?
                /*for (int i=0; i<n-1; i++)
                {
                    tau_b(i) = max(tau_b_min, tau_b(i));
                }*/

                // Update multistep variables.
                //u1_old_2 = u1_old;
                //u2_old_2 = u2_old;

                // Number of iterations.
                c_picard = c_picard + 1;
            }
        }
   
        //filt = gaussian_filter(u2, zeros, 3.0, L, ds, n);
        //u2 = filt;

        // Update BC for the explicit scheme.
        u1_0 = 0.0;
        u2_0 = 0.0;

        // Update timestep from velocity field.
        // Courant-Friedrichs-Lewis condition (factor 1/2, 3/4).
        dt_CFL = 0.5 * ds * L / u1.maxCoeff();  
        dt     = min(dt_CFL, dt_max);

        // Store solution in nc file.
        if (c == 0 || t > a(c))
        {
            cout << "\n t = " << t / sec_year;
            cout << "\n dt = " << dt / sec_year;

            u1_plot = sec_year * u1;
            u2_plot = sec_year * u2 / L;
            t_plot  = t / sec_year;
            dt_plot = dt / sec_year;

            start[0]   = c;
            start_0[0] = c;
            start_z[0] = c;
            if ((retval = nc_put_vara_float(ncid, x_varid, start, cnt, &u1_plot(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, u2_varid, start, cnt, &u2_plot(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, H_varid, start, cnt, &H(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, visc_varid, start, cnt, &visc(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, s_varid, start, cnt, &S(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, tau_varid, start, cnt, &tau_b(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, taud_varid, start, cnt, &tau_d(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, b_varid, start, cnt, &bed(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, C_bed_varid, start, cnt, &C_bed(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, u2_dif_vec_varid, start, cnt, &u2_dif_vec(0))))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, u2_0_vec_varid, start, cnt, &u2_0_vec(0))))
            ERR(retval);

            if ((retval = nc_put_vara_float(ncid, L_varid, start_0, cnt_0, &L)))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, t_varid, start_0, cnt_0, &t_plot)))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, u2_bc_varid, start_0, cnt_0, &u2_bc)))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, u2_dif_varid, start_0, cnt_0, &u2_dif)))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, picard_error_varid, start_0, cnt_0, &error)))
            ERR(retval);
            if ((retval = nc_put_vara_float(ncid, dt_varid, start_0, cnt_0, &dt_plot)))
            ERR(retval);
            if ((retval = nc_put_vara_int(ncid, c_pic_varid, start_0, cnt_0, &c_picard)))
            ERR(retval);

            if ((retval = nc_put_vara_float(ncid, theta_varid, start_z, cnt_z, &theta(0,0))))
            ERR(retval);

            c = c + 1;
        }  

        // Update ice viscosity with new u2 field.
        visc = f_visc(u2, B, n_exp, eps, L, n);
        /*filt = gaussian_filter(visc, zeros, 4.0, L, ds, n);
        visc = filt;*/

        // Evaluate calving front thickness from tau_b field.
        //H_c = f_calv(tau_b, D, rho, rho_w, g, bed);

        // Update grounding line position.
        L_new = f_L(u1, H, S, H_c, dt, L, ds, n, bed, rho, rho_w);

        // Integrate ice thickness forward in time.
        H_now = f_H_flux(u1, H, S, sigma, dt, ds_inv, n, \
                         L_new, L, L_old, H_c, D, rho, rho_w);
        H   = H_now; 

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
    printf("\n Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    printf("\n Computational speed: %.3f kyr/hr.\n", \
            60 * 60 * (1.0e-3 * tf / sec_year) /  (elapsed.count() * 1e-9) );

    // Close nc file. 
    if ((retval = nc_close(ncid)))
    ERR(retval);
 
    printf("\n *** %s file has been successfully written \n", FILE_NAME);

    return 0;
}