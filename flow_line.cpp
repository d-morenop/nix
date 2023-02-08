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

#include "read-write_nc.cpp"
//#include "read_nc.cpp"

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


////////////////////////////////////////////////////
////////////////////////////////////////////////////
// TOOLS. USEFUL FUNCTIONS.

/*
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
*/

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

Array2d f_dt(double error, double picard_tol, double dt_meth, \
            double t, double dt, double t_eq, double dt_min, \
            double dt_max, double dt_CFL, double rel)
{
    // Local variables.
    Array2d out;
    double dt_tilde;

    // Fixed time step.
    if ( dt_meth == 0 )
    {
        dt = dt_min;
    }
    // Idea: adaptative timestep from Picard error.
    else
    {
        // Mimum time step during equilibration.
        if ( t < t_eq )
        {
            dt = dt_min;
        }

        // Adaptative dt after equilibration.
        else
        {
            // Linear decrease of dt with Picard error.
            //dt_tilde = ( 1.0 - ( min(error, picard_tol) / picard_tol ) ) * \
                       ( dt_max - dt_min ) + dt_min;

            // Quadratic dependency.
            dt_tilde = ( 1.0 - pow( min(error, picard_tol) / picard_tol, 2) ) * \
                       ( dt_max - dt_min ) + dt_min;

            // Apply relaxation.
            dt = rel * dt + (1.0 - rel) * dt_tilde;

            // Ensure Courant-Friedrichs-Lewis condition is met.
            dt = min(dt, dt_CFL);
        }
    }

    // Update time.
    t = t + dt;

    // Output variables.
    out(0) = t;
    out(1) = dt;

    return out;
}



////////////////////////////////////////////////////
////////////////////////////////////////////////////
// Flow line functions.

ArrayXd f_bed(double L, int n, int experiment, \
              double y_0, double y_p, double x_1, double x_2)
{
    // Prepare variables.
    ArrayXd bed(n);
    ArrayXd x = ArrayXd::LinSpaced(n, 0.0, L); 


    // MISMIP experiments bedrock.
    // Same bedrock as Schoof (2007).
    // Inverse sign to get a decreasing bedrock elevation.
    if (experiment == 1)
    {
        x = x / 750.0e3; 
        bed = 720.0 - 778.5 * x;
    }
    else if (experiment == 3)
    {
        x = x / 750.0e3; 
        bed = 729.0 - 2148.8 * pow(x, 2) + \
                      1031.72 * pow(x, 4) + \
                    - 151.72 * pow(x, 6);
    }
    else if (experiment = 4)
    {
        // Variables.
        int c_x1 = 0;
        int c_x2 = 0;
        double y_1, y_2;
        double m_bed = y_p / ( x_2 - x_1 );

        // Piecewise function.
        for (int i=0; i<n; i++)
        {
            // First part.
		    if ( x(i) <= x_1 )
            {
                bed(i) = y_0 - 1.5e-3 * x(i);
            }
		
		    // Second.
            else if ( x(i) >= x_1 && x(i) <= x_2 )
            {
                // Save index of last point in the previous interval.
                if ( c_x1 == 0 )
                {
                    y_1  = bed(i-1);
                    c_x1 = c_x1 + 1	;
                }
                    
                // Bedrock function.
                bed(i) = y_1 + m_bed * ( x(i) - x_1 );
            }
                
            // Third.
            else if ( x(i) > x_2 )
            {
                // Save index of last point in the previous interval.
                if (c_x2 == 0)
                {
                    y_2  = bed(i-1);
                    c_x2 = c_x2 + 1;	
                }
                    
                // Bedrock function.
                bed(i) = y_2 - 5.0e-3 * ( x(i) - x_2 );
            } 
        }
    }

    return bed;
}


Array2d f_bc(ArrayXXd noise, ArrayXd t_vec, double dt_noise, \
             int N, double t, double dt, int n)
{
    Array2d noise_now;
    int idx;
    
    // This can be further improved by including interpolation.
    for (int i=0; i<N-1; i++) 
    {
        // Time frame closest to current flowline time since
        // dt_flowline != dt_stochastic.
        if ( abs(t_vec(i) - t) < abs(t_vec(i+1) - t) )
        {
            idx = i;
            break;
        }
        
        // Last index if condition is never met.
        else
        {
            idx = N - 1;
        }
    }

    // Load stochastic term given current time index.
    noise_now = noise.col(idx);

    return noise_now;
}


// SMB spatial distribution depends on the position of the ice.
// It does not follow the ice sheet, but rather there's a certain 
// value for each position x.

ArrayXd f_smb(ArrayXd sigma, double L, double S_0, \
              double x_mid, double x_sca, double x_varmid, \
              double x_varsca, double dlta_smb, double var_mult, \
              double smb_stoch, double t, double t_eq, int n, int stoch)
{
    // Variables
    ArrayXd x(n), S(n); 
    double stoch_pattern, smb_determ;

    // Horizontal dimension to define SBM.
    x = L * sigma;

    // Start with constant accumulation value for smoothness.
    if ( t < t_eq || stoch == 0)
    {
        S = ArrayXd::Constant(n, S_0);
    }
    
    // After equilibration, spatially dependent and potentially stochastic term.
    else
    {
        // Error function from Christian et al. (2022)
        for (int i=0; i<n; i++)
        {
            // No stochastic variability.
            //S(i) = S_0 + 0.5 * dlta_smb * ( 1.0 + erf((x(i) - x_mid) / x_sca) );

            // SMB stochastic variability sigmoid.
            stoch_pattern = var_mult + ( 1.0 - var_mult ) * 0.5 * \
                            ( 1.0 + erf((x(i) - x_varmid) / x_varsca) );

            // Deterministic SMB sigmoid.
            smb_determ = S_0 + 0.5 * dlta_smb * ( 1.0 + erf((x(i) - x_mid) / x_sca) );
            
            // Total SMB: stochastic sigmoid + deterministic sigmoid.
            S(i) = smb_stoch * stoch_pattern + smb_determ;
        }
    }
    
    return S;
}


ArrayXd f_q(ArrayXd u1, ArrayXd H, double H_f, double t, double t_eq, double D, \
               double rho, double rho_w, double m_dot, int calving_meth, int n)
{
    // Local variables.
    ArrayXd q(n);
    double H_mean;

    // Flux defined on velocity grid. Staggered grid.
    for (int i=0; i<n-1; i++)
    {
        // Vieli and Payne (2005) discretization.
        q(i) = u1(i) * 0.5 * ( H(i+1) + H(i) );

        // "One-before-last" approach.
        //q(i) = u1(i) * H(i+1);
    }   
        
    // GL flux definition (Vieli and Payne, 2005).
    if ( calving_meth == 0 )
    {
        q(n-1) = u1(n-1) * H(n-1);
    } 
    
    // Additional calving term (Christian et al., 2022).
    // ICE FLUX DISCRETIZATION SEEMS TO BE FUNDAMENTAL TO OBTAIN
    // THE SAME ICE SHEET ADVANCED AS CHRISTIAN. STAGGERED GRID.
    else if ( calving_meth == 1 )
    {
        // No calving during equilibration.
        if ( t < t_eq )
        {
            // Vieli and Payne (2005).
            q(n-1) = u1(n-1) * H(n-1);

            // Schoof (2007).
            //q(n-1) = H(n-1) * 0.5 * ( u1(n-1) + u1(n-2) );
        }
        
        // Calving after equilibration.
        else
        {
            // Prefactor to account for thickness difference in last grid point.
            // GL is defined in the last velocity grid point.
            // H(n-1) is not precisely H_f so we need a correction factor.
            
            // Successful MISMIP experiments but GL 
            //too retreated compared to Christian et al. (2022).
            //q(n-1) = ( u1(n-1) + ( H_f / H(n-1) ) * m_dot ) * H(n-1);

            // This seems to work as Christian et al. (2022).
            // Velocity evaluated at (n-2) instead of last point (n-1).
            //q(n-1) = ( u1(n-2) + ( H_f / H(n-1) ) * m_dot ) * H(n-1);

            // Ice flux at GL computed on the ice thickness grid. Schoof (2007).
            // Too retreated
            //q(n-1) = H(n-1) * 0.5 * ( u1(n-1) + u1(n-2) + ( H_f / H(n-1) ) * m_dot );

            // Ice flux at GL computed on the ice thickness grid. Schoof (2007).
            // Mean.
            // Works for stochastic! A more retreatesd ice sheet.
            //H_mean = 0.5 * ( H(n-1) + H_f ); 
            //q(n-1) = H_mean * 0.5 * ( u1(n-1) + u1(n-2) + ( H_f / H_mean ) * m_dot );
            //q(n-1) = H_mean * 0.5 * ( u1(n-1) + u1(n-2) + m_dot );

            // Previous grid points as the grid is staggered. Correct extension!!
            // Good results for stochastic with n = 350.
            q(n-1) = H(n-1) * 0.5 * ( u1(n-2) + u1(n-3) + ( H_f / H(n-1) ) * m_dot );

            // GL too advanced for n = 500.
            //q(n-1) = H_f * 0.5 * ( u1(n-1) + u1(n-2) + m_dot );

            // Slightly retreated.
            //q(n-1) = H_mean * ( u1(n-1) + m_dot );

            // GL too advanced for n = 500.
            //q(n-1) = H_f * ( u1(n-2) + m_dot );

            // H_n.
            // Too retreated.
            //q(n-1) = H(n-1) * ( u1(n-1) + m_dot );
        }
    }

    return q;
}


Array2d f_L(ArrayXd H, ArrayXd q, ArrayXd S, ArrayXd bed, \
          double dt, double L, double ds, int n, double rho, \
          double rho_w, int dL_dt_num_opt, int dL_dt_den_opt)
{
    //Local variables.
    Array2d out;
    double num, den, dL_dt;
    
    
    // DISCRETIZATION OPTIONS.
    // MIT finite difference calculator: https://web.media.mit.edu/~crtaylor/calculator.html
    // den_opt = 4 rises advance/retreat problems.
    
    
    // Accumulation minus flux (reverse sign). 

    // Two-point backward-difference.
    if ( dL_dt_num_opt == 1)
    {
        num = q(n-1) - q(n-2) - L * ds * S(n-1);
    }
    // Three-point asymmetric backward-diference for bedrock. Unstable!
    else if ( dL_dt_num_opt == 2 )
    {
        num = - L * ds * S(n-1) + 0.5 * ( 3.0 * q(n-1) - 4.0 * q(n-2) + q(n-3) );
    }


    // Purely flotation condition (Following Schoof, 2007).
    // Sign before db/dx is correct. Otherwise, it migrates uphill.

    // Two-point backward-diference for bedrock and ice thickness.
    if ( dL_dt_den_opt == 1 )
    {
        den = H(n-1) - H(n-2) + ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );
    }
    // Three-point asymmetric backward-diference for bedrock.
    else if ( dL_dt_den_opt == 2 )
    {
        den = H(n-1) - H(n-2) + ( rho_w / rho ) * \
              0.5 * ( 3.0 * bed(n-1) - 4.0 * bed(n-2) + bed(n-3) );
    }
    // / Three-point asymmetric backward-diference bedrock and thickness.
    else if ( dL_dt_den_opt == 3 ) 
    {
        den = 3.0 * H(n-1) - 4.0 * H(n-2) + H(n-3) + ( rho_w / rho ) * \
              0.5 * ( 3.0 * bed(n-1) - 4.0 * bed(n-2) + bed(n-3) );
    }
    // Five-point asymmetric backward-difference deffinition. Unstable!
    else if ( dL_dt_den_opt == 4 ) 
    {
        // Local vraiables.
        double D_5;

        D_5 = ( 137.0 * H(n-1) - 300.0 * H(n-2) + 300.0 * H(n-3) + \
                - 200.0 * H(n-4) + 75.0 * H(n-5) - 12.0 * H(n-6) ) / 60.0;
        
        // Simple bedrock slope.
        den = D_5 + ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );
    }

    // Grounding line time derivative.
    dL_dt = num / den;

    // Update grounding line position.
    L = L + dL_dt * dt;

    // Output vairables.
    out(0) = L;
    out(1) = dL_dt;

    return out;
} 


// First order scheme. New variable sigma.
ArrayXd f_H(ArrayXd u1, ArrayXd H, ArrayXd S, ArrayXd sigma, \
            double dt, double ds, double ds_inv, int n, \
            double L, double D, double rho, double rho_w, \
            double dL_dt, ArrayXd bed, ArrayXd q, int H_meth)
{
    // Local variables.
    ArrayXd H_now(n);

    double L_inv  = 1.0 / L;

    // Solution to the modified advection equation considering a streched coordinate
    // system sigma. Two schemes are available, explicit and implicit, noted as
    // meth = 0, 1 respectively. 
    //  Right now, explicit seems more stable since the 
    // implicit crasher earlier. 
    
    // Explicit scheme. Centred dH in the sigma_L term.
    if ( H_meth == 0 )
    {
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
        
        // Lateral boundary: sigma(n-1) = 1.
        H_now(n-1) = H(n-1) + dt * ( ds_inv * L_inv * \
                                        (  dL_dt * ( H(n-1) - H(n-2) ) + \
                                            - ( q(n-1) - q(n-2) ) ) + S(n-1) );
        
    }
    
    // Implicit scheme.
    else if ( H_meth == 1 )
    {
        // Local variables.
        ArrayXd A(n), B(n), C(n), F(n);
        double gamma = dt / ( 2.0 * ds * L );

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
        H_now(0) = H_now(2);

        // The first option appears to be the best.
        //H_now(n-1) = ( F(n-1) - A(n-1) * H_now(n-2) ) / B(n-1);
        
        //H_now(n-1) = H(n-1) + S(n-1) * dt - gamma * ( 2.0 * dL_dt * H_now(n-2) \
                    - u1(n-2) * H_now(n-2) ) / ( 1.0 + gamma * ( - 2.0 * dL_dt \
                                                + u1(n-1) - u1(n-2) ) ); 
        
        //H_now(n-1) = H(n-1) + S(n-1) * dt - gamma * ( 2.0 * dL_dt * H_now(n-2) \
                    - u1(n-2) * H_now(n-2) ) / ( 1.0 + gamma * ( - 2.0 * dL_dt \
                                                - u1(n-2) ) ); 
        
        H_now(n-1) = min( D * ( rho_w / rho ), H_now(n-1) );
        
        
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


ArrayXd f_visc(ArrayXd u2, ArrayXXd theta, double theta_act, \
               Array2d Q_act, Array2d A_0, double n_gln, double R, double B, double n_exp, \
               double eps, double t, double t_eq, int n, int n_z, int visc_therm)
{
    ArrayXd u2_eps(n), visc(n); 
    

    if (visc_therm == 1 && t > t_eq)
    {
        ArrayXXd A(n,n_z), B(n,n_z);
        ArrayXd B_bar(n);

        // Calculate temperature-dependent rate factor if thermodynamics is switched on.
        // Arrhenius law A(T,p) equivalent to A(T').
        // Eq. 4.15 and 6.54 (Greve and Blatter, 2009).
        for (int i=1; i<n; i++)
        {
            for (int j=1; j<n_z; j++)
            {
                // Rate factor. We consider two temperature regimes (Greve and Blatter, 2009).
                if ( theta(i,j) < theta_act )
                {
                    A(i,j) = A_0(0) * exp(- Q_act(0) / (R * theta(i,j)) );
                }
                else
                {
                    A(i,j) = A_0(1) * exp(- Q_act(1) / (R * theta(i,j)) );
                }

            }
        }

        // Associated rate factor.
        B = pow(A, ( -1 / n_gln ) );

        // Vertical average from base to H.
        B_bar = B.rowwise().mean();

        // Regularitazion term to avoid division by 0. 
        u2_eps = u2 + eps;

        // Exponent n_exp = (1-n)/n
        visc = 0.5 * B_bar * pow(u2_eps, n_exp);

        // Avoid singularity???????
        visc(1) = visc(0);

    }
    
    else
    {
        // Regularitazion term to avoid division by 0. 
        u2_eps = u2 + eps;

        // n_exp = (1-n)/n
        visc = 0.5 * B * pow(u2_eps, n_exp);
        //cout << "\n visc = " << visc * sec_year;

        // Constant viscosity experiment:
        //visc = ArrayXd::Constant(n, 0.5e17); // 1.0e15, 0.5e17
    }
    
	return visc;
}	

ArrayXd f_C_bed(ArrayXd C_ref, ArrayXXd theta, double t, double t_eq, double theta_max, \
                double theta_frz, double C_frz, double C_thw, int fric_therm, int n)
{
    ArrayXd C_bed(n), theta_norm(n);

   
    // Basal friction coupled with thermal state of the base.
    if (fric_therm == 1 && t > t_eq)
    {
        
        // Binary friction.
        for (int i=1; i<n; i++)
        {
            // Reference value for frozen bed.
            if ( theta(i,0) < theta_frz )
            {
                C_bed(i) = C_ref(i);
            }

            // Reduction by a factor of 2 if we approach melting.
            else
            {
                C_bed(i) = 0.1 * C_ref(i);
            }
        }
        

        // Normalized basal temperature with pressure melting point [0,1].
        //theta_norm = abs( theta.block(0,0,n,1) - theta_frz ) / abs(theta_max - theta_frz);
        //C_bed      = C_thw * theta_norm + C_frz * ( 1.0 - theta_norm );

    }

    // Friction given by reference value.
    else 
    {
        C_bed = C_ref;
    }

    return C_bed;
}


ArrayXXd f_theta(ArrayXXd theta, ArrayXd u1, ArrayXd H, ArrayXd tau_b, ArrayXd Q_fric, \
                 ArrayXd sigma, double theta_max, double T_air, double kappa, double k, \
                 double dt, double G_k, double ds, double L, \
                 double dL_dt, double t, double t_eq, int n, int n_z)
{
    MatrixXd theta_now(n,n_z);
    ArrayXd dz(n), dz_2_inv(n), Q_f_k(n);

    double dx_inv;
 
    // Evenly-spaced vertical grid.
    dz       = H / n_z;
    dz_2_inv = 1.0 / pow(dz, 2);
    dx_inv   = 1.0 / (ds * L);

    // Frictional heat units. // [Pa m yr^-1] = [J yr^-1 m^-2] = [W m^-2] => [K / m]
    Q_f_k = Q_fric / k;      // [Pa m s^-1] = [J s^-1 m^-2] = [W m^-2] => [K / m]
    
    // Zero frictional heat test.
    //Q_f_k = ArrayXd::Zero(n);

    if ( t > t_eq )
    {
        for (int i=1; i<n; i++)
        {
            for (int j=1; j<n_z-1; j++)
            {
                // Old formulation.
                //theta_now(i,j) = theta(i,j) + dt * ( kappa * dz_2_inv(i) * \
                                ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                                - u1(i) * ( theta(i,j) - theta(i-1,j) ) * dx_inv );

                
                // Just vertical difussion.
                //theta_now(i,j) = theta(i,j) + dt * kappa * dz_2_inv(i) * \
                                ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) );

                // Horizontal advection. Correction for sigma coord.
                // We have not considered the stagerred grid yet.
                theta_now(i,j) = theta(i,j) + dt * ( kappa * dz_2_inv(i) * \
                                ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                                ( sigma(i) * dL_dt - u1(i) ) * \
                                ( theta(i,j) - theta(i-1,j) ) * dx_inv );
            }
            
            // Boundary conditions. Geothermal heat flow at the base \
            and prescribed T_air at the surface.
            // We add friciton heat contribution Q_f_k.
            theta_now(i,0)     = theta_now(i,1) + dz(i) * ( G_k + Q_f_k(i) ); 
            //theta_now(i,0)     = 268.0;
            theta_now(i,n_z-1) = T_air;

            // Pressure melting point as the upper bound.
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
        //theta_now(0,0)     = 268.0; 
        theta_now(0,n_z-1) = T_air;
        theta_now(0,0)     = min(theta_now(0,0), theta_max);

        // LAST POINT IN SUPPOSED TO BE ATTACHED TO THE ICE SHELF!
    }

    // No integration while equilibration.
    else
    {
        theta_now = theta;
    }

    

    return theta_now;
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
                    ArrayXd bed, double rho, double rho_w, double g, double L, double m, \
                    ArrayXd C_bed, double t_eq, ArrayXd tau_b)
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

        ++c;
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



MatrixXd vel_solver(ArrayXd u1, ArrayXd H, double ds, double ds_inv, int n, ArrayXd visc, \
                    ArrayXd bed, double rho, double rho_w, double g, double L, ArrayXd C_bed, \
                    double t, double u2_RK, ArrayXd beta, double A_ice, double n_gln)
{
    ArrayXd u2(n), dhds(n), visc_H(n), c1(n), c2(n), h(n), \
            A(n), B(n), C(n), F(n);

    MatrixXd out(3,n);

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
    dhds(n-1) = H(n-1) * ( h(n-1) - h(n-2) ); 
    //dhds(n-1) = H(n-1) * 0.5 * ( 3.0 * h(n-1) - 4.0 * h(n-2) + h(n-3) );

    // Tridiagonal boundary values. 
    A(0) = 0.0;
    B(0) = - gamma * ( visc_H(0) + visc_H(1) ) - beta(0);
    C(0) = visc_H(1);

    A(n-1) = visc_H(n-1);
    B(n-1) = - gamma * visc_H(n-1) - beta(n-1);
    C(n-1) = 0.0;
    
    // Inhomogeneous term.
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
            // Default successful.
            //u1(i) = u1(i+1); 

            // Experimental.
            u1(i) = u1(i-1);
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
    out(2,0)   = D;
    out(2,1)   = u2_bc;
    
    return out;
}


// RUN FLOW LINE MODEL.
// Driver method
int main()
{
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////
    // Initialize flowline.

    // GENERAL PARAMETERS.
    double const sec_year = 3.154e7;                // Seconds in a year.

    // PHYSICAL CONSTANTS.
    double const u_0   = 150.0 / sec_year;
    double const g     = 9.8;                      // Gravitational acceleration [m/s²].
    double const rho   = 900.0;                     // Ice density [kg/m³]. 917.0
    double const rho_w = 1000.0;                    // Water denisity [kg/m³]. 1028.0

   
    // GROUNDING LINE.
    double L = 694.5e3;                              // Grounding line position [m] (479.1e3), 50.0e3
    double dL_dt;                                   // GL migration rate [m/yr]. 
    int const dL_dt_num_opt = 1;                    // GL migration numerator discretization opt.
    int const dL_dt_den_opt = 1;                    // GL migration denominator discretization opt.

    // ICE VISCOSITY: visc.
    double const n_gln = 3.0;
    double const n_exp = (1.0 - n_gln) / n_gln;      // Pattyn.

    // VISCOSITY REGULARIZATION TERM.
    // eps is fundamental for GL, velocities, thickness, etc.
    double const eps = 1.0e-10;                            

    // BASAL FRICTION.
    double const m = 1.0 / 3.0;                      // Friction exponent.


    // SIMULATION PARAMETERS.
    int const n   = 250;                             // 600. Number of horizontal points 350, 500, 1000, 1500
    int const n_z = 10;                              // Number vertical layers. 10, 20.
    
    double const ds     = 1.0 / n;                   // Normalized spatial resolution.
    double const dz     = 1.0 / n_z;                 // Normalized vertical resolution.
    double const ds_inv = n;
    double const dz_inv = n_z;

    double const t0   = 0.0;                         // Starting time [yr].
    double const tf   = 20.0e3;                       // 2.0e4, Ending time [yr]. 1.0e4.
    double t;                                        // Time variable [yr].

    // TIME STEPING. Quite sensitive (use fixed dt in case of doubt).
    // For stochastic perturbations. dt = 0.1 and n = 250.
    int const dt_meth = 0;                           // Time-stepping method. Fixed, 0; adapt, 1.
    double dt;                                       // Time step [yr].
    double dt_CFL;                                   // Courant-Friedrichs-Lewis condition [yr].
    double dt_tilde;                                 // New timestep. 
    double const t_eq = 0.2 * tf;                    // Length of equilibration time [yr] .
    double const dt_min = 0.1;                       // Minimum time step [yr]. 0.1
    double const dt_max = 2.0;                       // Maximum time step [yr]. 
    double const rel = 0.7;                          // Relaxation between interations [0,1]. 0.5
    
    // INPUT DEFINITIONS.   
    int const N = 20000;                             // Number of time points in BC (input from noise.nc in glacier_ews).
    double const dt_noise = 1.0;                     // Assumed time step in stochastic_anom.py 

    // OUTPUT DEFINITIONS.
    int const t_n = 100;                             // Number of output frames. 30.

    // BEDROCK
    // Glacier ews option.
    double const x_1 = 346.0e3;                      // Peak beginning [m].
    double const x_2 = 350.0e3;                      // Peak end [m].
    double const y_p = 88.0;                         // Peak height [m].
    double const y_0 = 70.0;                         // Initial bedrock elevation (x=0) [m].
    
    // SURFACE MASS BALANCE.
    int const stoch = 0;                             // Stochastic SBM.
    double const S_0      = 0.7;                     // SMB at x = 0 (for equilibration) [m/yr].
    double const dlta_smb = -4.0;                    // Difference between interior and terminus SMB [m/yr]. 
    double const x_acc    = 300.0e3;                 // Position at which accumulation starts decreasing [m]. 300.0, 355.0
    double const x_mid    = 3.5e5;                   // Position of middle of SMB sigmoid  [m]. 365.0, 375.0
    double const x_sca    = 4.0e4;                   // Length scale of area where SMB changing. [m]
    double const x_varmid = 2.0e5;                   // Position of middle of SMB variability sigmoid [m].
    double const x_varsca = 8.0e4;                   // Length scale of area where SMB varaibility changing. [m]
    double const var_mult = 0.25;                    // Factor by which inland variability is less than max.
    double m_stoch;
    double smb_stoch;


    // THERMODYNAMICS.
    int const thermodynamics = 1;                    // Apply thermodynamic solver at each time step.
    double const k = 2.0;                            // Thermal conductivity of ice [W / m · ºC].
    double const G = 0.05;                           // Geothermal heat flow [W / m^2] = [J / s · m^2].
    double const G_k = G / k;                        // [K / m] 
    double const kappa = 1.4e-6 * sec_year;          // Thermal diffusivity of ice [m^2/s] --> [m^2/yr].
    double const theta_max = 273.15;                 // Max temperature of ice [K].
    double const theta_act = 263.15;                 // Threshold temperature for the two regimes in activation energy [K]
    double const R = 8.314;                          // Universal gas constant [J / K mol]
    double const T_air = 253.15;                     // BC: prescribed air temperature.
    

     // BEDROCK PARAMETRIZATION: f_C_bed.
    int const fric_therm = 0;                       // Coupled friction + thermodynamics.
    double const theta_frz = 268.15;
    double const C_frz = 7.624e6 / pow(sec_year, m); // Frozen friction coeff. [Pa m^-1/3 yr^1/3]
    double const C_thw = 0.5 * C_frz;                // Thawed friction coeff. [Pa m^-1/3 yr^1/3]

    // REVISE UNITS HERE!!!
    int const visc_therm = 1;                        // Temperature dependent viscosity.
    Array2d Q_act, A_0; 
    Q_act << 60.0, 139.0;                            // Activation energies [kJ/mol].
    Q_act = 1.0e3 * Q_act;                           // [kJ/mol] --> [J/mol] 
    A_0 << 3.985e-13, 1.916e3;                       // Pre-exponential constants [Pa^-3 s^-1]
    A_0 = A_0 * sec_year;                            // [Pa^-3 s^-1] --> [Pa^-3 yr^-1]

    // AVECTION EQUATION.
    int const H_meth = 0;                              // Solver scheme: 0, explicit; 1, implicit.

    // LATERAL BOUNDARY CONDITION.
    double D;                                        // Depth below the sea level [m].
    double u2_bc;                                    // Boundary condition on u2 = du1/dx.
    double u2_dif;                                   // Difference between analytical and numerical.

    // CALVING.
    int const calving_meth = 0;                      // 0, no calving; 1, Christian et al. (2022).
    double const m_dot = 30.0;                       // Mean frontal ablation [m/yr]. 30.0
    double H_f;

    // MISMIP EXPERIMENT CHOICE.
    // Following Pattyn et al. (2012) the overdeepening hysterisis uses n = 250.
    // exp = "mismip_1", "mismip_3", "galcier_ews"
    //int const mismip = 1;
    int experiment = 1;
    double A, B;

    // PICARD ITERATION
    double error;                           // Norm of the velocity difference between iterations.
    double omega;                           // Angle between two consecutive velocities [rad]. 
    double mu;                              // Relaxation method within Picard iteration. 
    
    int c_picard;                           // Number of Picard iterations.
    int const n_picard = 10;                // Max number iter. Good results: 10, 20.
    
    double const picard_tol = 1.0e-4;              // Convergence tolerance within Picard iteration. 1.0e-5
    double const omega_1 = 0.125 * M_PI;           // De Smedt et al. (2010) Eq. 10.
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
    ArrayXd C_ref(n);                    // Reference riction coefficient [Pa m^-1/3 s^1/3].
    ArrayXd C_bed(n);                    // Friction coefficient [Pa m^-1/3 s^1/3].
    ArrayXd Q_fric(n);                   // Frictional heat [W / m^2].
    ArrayXd visc(n);                     // Ice viscosity [Pa·s].
    ArrayXd S(n);                        // Surface accumulation equivalent [mm/day].
    ArrayXd u1_plot(n);                  // Saved ice velocity [m/yr]
    ArrayXd u2_plot(n);                  // Saved ice velocity derivative [1/yr]
    ArrayXd tau_b(n);                    // Basal friction [Pa]
    ArrayXd beta(n);                     // Basal friction [Pa m^-1 yr]
    ArrayXd tau_d(n);                    // Driving stress [Pa]
    ArrayXd u1_old_1(n);  
    ArrayXd u1_old_2(n);  
    ArrayXd u2_0_vec(n);                 // Ranged sampled of u2_0 for a certain iteration.
    ArrayXd u2_dif_vec(n);               // Difference with analytical BC.
    
    // Stochasticmatrices and vectors.
    ArrayXXd noise(2,N);                            // Matrix to allocate stochastic BC.
    Array2d noise_now;                              // Two-entry vector with current stochastic noise.
    ArrayXd t_vec = ArrayXd::LinSpaced(N, 0.0, N);  // Time vector with dt_noise time step.
    
    // Vectors to compute norm.
    VectorXd u1_vec(n); 
    VectorXd u2_vec(n); 
    VectorXd c_u1_1(n);                   // Correction vector Picard relaxed iteration.
    VectorXd c_u1_2(n);
    VectorXd c_u1_dif(n);

    // MISMIP FORCING.
    //ArrayXd A_s(n_s);                    // Rarte factor values for MISMIP exp.
    //ArrayXd t_s(n_s);                    // Time length for each step of A.

    // MATRICES.
    MatrixXd u(3,n);                     // Matrix output.

    ArrayXXd theta(n,n_z);                 // Temperature field [K].

    // Function outputs.
    Array2d L_out;                    // Grounding line function output.
    Array2d dt_out;                   // Time step function output.

    // Normalised horizontal dimension.
    ArrayXd sigma = ArrayXd::LinSpaced(n, 0.0, 1.0);      // Dimensionless x-coordinates. 
    ArrayXd a     = ArrayXd::LinSpaced(t_n, t0, tf);      // Time steps in which the solution is saved. 


    // EXPERIMENT. Christian et al (2022).
    // Constant friction coeff. 7.624e6 [Pa m^-1/3 s^1/3]
    C_ref = ArrayXd::Constant(n, 7.624e6/ pow(sec_year, m) );    // [Pa m^-1/3 yr^1/3] 7.0e6

    // We assume a constant viscosity in the first iteration. 1.0e13 Pa s.
    visc = ArrayXd::Constant(n, 1.0e12 / sec_year);            // [Pa yr]

    // Implicit initialization.
    beta = ArrayXd::Constant(n, 5.0e3);             // [Pa yr / m]
    u1   = ArrayXd::Constant(n, 1.0);               // [m / yr]
    
    // Viscosity from constant A value. u2 = 0 initialization.
    // 4.6416e-24, 2.1544e-24. 4.227e-25 [Pa^-3 s^-1] ==> [Pa^-3 yr^-1]
    A = 4.6416e-24 * sec_year;               // 4.23e-25
    B = pow(A, ( -1 / n_gln ) );

    // Temperature initial conditions (-25ºC).
    theta = ArrayXXd::Constant(n, n_z, 248.0);

    // Intialize ice thickness and SMB.
    H = ArrayXd::Constant(n, 10.0);
    S = ArrayXd::Constant(n, 0.3);

    
    
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////


    // Print spatial and time dimensions.
    cout << " \n h = " << ds;
    cout << " \n n = " << n;
    cout << " \n tf = " << tf;

    // Call nc read function.
    noise = f_nc_read(N);
    //cout << "\n noise_ocn = " << noise;

    // Call nc write function.
    f_nc(n, n_z);


    // Wall time for computational speed.
    auto begin = std::chrono::high_resolution_clock::now();

    // Initilize ice thickness and accumulation.
    /*
    for (int i=0; i<n; i++)
    {
        // Initial ice thickness H0.
        //H(i) = 1.8e3 * (1.0 - pow(sigma(i), 2.5)) + 1.2e3;

        // Initial ice thickness H0 [m].
        H(i) = 100.0;         
        //S(i) = S_0; 

    }
    */
   
    // Counter to write solution.
    int c = 0;

    // Counter for MISMIP ice factor A forcing.
    int c_s = 0;

    // Initialize time.
    t  = t0;
    dt = dt_min;

    // Time integration.
    while (t < tf)
    {
        // Update bedrock with new domain extension L.
        bed = f_bed(L, n, experiment, y_0, y_p, x_1, x_2);

        // Friction coefficient.
        C_bed = f_C_bed(C_ref, theta, t, t_eq, theta_max, \
                        theta_frz, C_frz, C_thw, fric_therm, n);

        // Stochastic configuration. 
        // Update time-dependent boundary conditions after equilibration.
        if ( stoch == 1 )
        {
            // Christian's spin-up also considers stochastic anomalies?
            // Lower bound of zero in m to avoid numerical issues.
            noise_now = noise.col(floor(t));
            m_stoch   = max(0.0, noise_now(0)); 
            smb_stoch = noise_now(1);

            // Update SMB considering new domain extension and current stochastic term.
            S = f_smb(sigma, L, S_0, x_mid, x_sca, x_varmid, \
                      x_varsca, dlta_smb, var_mult, smb_stoch, t, t_eq , n, stoch);
        }

            
        // Update basal friction with previous step velocity. 
        // Before Picard iteration???
        tau_b    = beta * u1;
        tau_b(0) = tau_b(1);    // Symmetry ice divide. tau_b(0) = tau_b(1); 

        // Frictional heat. [Pa · m / yr] --> [W / m^2].
        Q_fric = tau_b * u1 / sec_year;

        // Picard initialization.
        error    = 1.0;
        c_picard = 0;

        // Implicit velocity solver. Picard iteration for non-linear viscosity and beta.
        while (error > picard_tol & c_picard < n_picard)
        {
            // Save previous iteration solution.
            u1_old_1 = u1;

            // Implicit solver.
            u = vel_solver(u1, H, ds, ds_inv, n, visc, bed, rho, rho_w, g, L, \
                           C_bed, t, u2_bc, beta, A, n_gln);

            // Allocate variables.
            u1    = u.row(0);
            u2    = u.row(1);
            D     = u(2,0);
            u2_bc = u(2,1);

            // Beta definition: tau_b = beta * u.
            beta    = C_bed * pow(u1, m - 1.0);
            beta(0) = beta(1);            // beta(0) = beta(1);

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
                
                // Angle defined between two consecutive vel solutions.
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
                beta(0) = beta(1); // Symmetry beta(0) = beta(1);

                // Update viscosity with new u2 field.
                for (int i=1; i<n-1; i++)
                {
                    // Centred stencil.
                    u2(i) = 0.5 * ( u1(i+1) - u1(i-1) );
                }
                u2(0)   = u1(1) - u1(0);
                u2(n-1) = u1(n-1) - u1(n-2);

                u2 = abs(u2) / (ds * L);
                visc = f_visc(u2, theta, theta_act, Q_act, A_0, n_gln, R, B, \
                              n_exp, eps, t, t_eq, n, n_z, visc_therm);
                
            }

            // Update multistep variables.
            u1_old_2 = u1_old_1;

            // Update number of iterations.
            ++c_picard;
        }


        // CONSISTENCY CHECK. Search for NaN values.
        // Count number of true positions in u1.isnan().
        if ( u1.isNaN().count() != 0 )
        {
            cout << "\n NaN found.";
            cout << "\n Saving variables in nc file. \n ";

            // Save previous iteration solution (before NaN encountered).
            f_write(c, u1_old_1, u2,  H, visc, S, tau_b, beta, tau_d, bed, \
                    C_bed, Q_fric, u2_dif_vec, u2_0_vec, L, t, u2_bc, u2_dif, \
                    error, dt, c_picard, mu, omega, theta, A, dL_dt, \
                    m_stoch, smb_stoch);

            // Close nc file. 
            if ((retval = nc_close(ncid)))
            ERR(retval);
            printf("\n *** %s file has been successfully written \n", \
                    FILE_NAME);
                
            // Abort flowline.
            return 0;
        }

        // Ice flux calculation. Flotation thickness H_f.
        H_f = D * ( rho_w / rho );
        q   = f_q(u1, H, H_f, t, t_eq, D, rho, rho_w, m_stoch, calving_meth, n);

        // Update grounding line position with new velocity field.
        L_out = f_L(H, q, S, bed, dt, L, ds, n, rho, rho_w, \
                    dL_dt_num_opt, dL_dt_den_opt);
        L     = L_out(0);
        dL_dt = L_out(1);

        // Save solution with desired output frequency.
        if (c == 0 || t > a(c))
        {
            cout << "\n t = " << t;

            // Write solution in nc.
            f_write(c, u1, u2,  H, visc, S, tau_b, beta, tau_d, bed, \
                    C_bed, Q_fric, u2_dif_vec, u2_0_vec, L, t, u2_bc, u2_dif, \
                    error, dt, c_picard, mu, omega, theta, A, dL_dt, \
                    m_stoch, smb_stoch);

            ++c;
        }  

        // Update ice viscosity with new u2 field.
        visc = f_visc(u2, theta, theta_act, Q_act, A_0, n_gln, R, B, \
                      n_exp, eps, t, t_eq, n, n_z, visc_therm);

        // Integrate ice thickness forward in time.
        H = f_H(u1, H, S, sigma, dt, ds, ds_inv, n, \
                L, D, rho, rho_w, dL_dt, bed, q, H_meth);


        // Apply thermodynamic solver if desired.
        if ( thermodynamics == 1 )
        {
            // Integrate Fourier heat equation.
            theta = f_theta(theta, u1, H, tau_b, Q_fric, sigma, theta_max, T_air, kappa, \
                            k, dt, G_k, ds, L, dL_dt, t, t_eq, n, n_z);
        }

        // Courant-Friedrichs-Lewis condition.
        // Factor 0.5 is faster since it yields fewer Picard's iterations.
        dt_CFL = 0.5 * ds * L / u1.maxCoeff();  
        
        // Update timestep and current time.
        dt_out = f_dt(error, picard_tol, dt_meth, t, dt, \
                      t_eq, dt_min, dt_max, dt_CFL, rel);
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

    // Close nc file. 
    if ((retval = nc_close(ncid)))
    ERR(retval);
 
    printf("\n *** %s file has been successfully written \n", FILE_NAME);

    return 0;
}