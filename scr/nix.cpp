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

using namespace Eigen;
using namespace std;

#include "read-write_nc.cpp"


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
                   Notation: ub = u (velocity) and u2 = dub/ds (s denotes sigma).
                   Vectorial magnitude: f_du_dz(0) = dub/ds and f_du_ds(1) = du2/ds.
rungeKutta   --->  4th-order Runge-Kutta integration scheme for the SSA stress 
                   balance. Spatial resolution is ds to ensure consistency with 
                   derivatives. 
*/


////////////////////////////////////////////////////
////////////////////////////////////////////////////
// TOOLS. USEFUL FUNCTIONS.


ArrayXd gaussian_filter(ArrayXd w, double sigma, int p, int n)
{
    ArrayXd smth(n);
    ArrayXd summ = ArrayXd::Zero(n);

    double x, y;
    //double dx = 1.0e-3 * ds * L; // [m] --> [km]. L in metres but bed is in km.
    
    // Test forcing dx = 1.0. It must be 1.0 for y-amplitude consistency.
    double dx = 1.0;
    //sigma = 2.0 * dx;
    
    // Handy definition. Standard deviation is a multiple of dx (real distance among gridpoints).
    double sigma_inv = 1.0 / sigma;
    double A = sigma_inv / sqrt(2.0 * M_PI);
    
    // Weierstrass transform.
    for (int i=p; i<n-p; i++)
    {
        x = i * dx;

        for (int j=0; j<n; j++)
        {
            y = j * dx;
            summ(i) += w(j) * exp(- 0.5 * pow((x - y) * sigma_inv, 2) );
        }
    }

    // Normalizing Kernel.
    smth = A * summ;

    // The edges are identical to the original array.
    // (p-1) since the very element (n-1-p) must be also filled.
    smth.block(0,0,p,1)       = w.block(0,0,p,1);
    smth.block(n-1-p,0,p+1,1) = w.block(n-1-p,0,p+1,1); 

    return smth;
}



ArrayXd running_mean(ArrayXd x, int p, int n)
{
    ArrayXd y(n);
    double sum, k;

    // Assign values at the borders f(p).
    y = x;

    // Average.
    k = 1.0 / ( 2.0 * p + 1.0 );
 
    // Loop.
    for (int i=p; i<n-p; i++) 
    {
        sum = 0.0;

        for (int j=i-p; j<i+p+1; j++) 
        {
            //sum = sum + x(j);
            sum += x(j);
        }
        y(i) = k * sum;
    }

    // It cannot be out of the loop cause points at the boundaries arent averaged.
    //y = k * sum;
 
    return y;
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
    //cout << "\n ub(n-1) = " << ub(n-1);
    
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
    if ( dt_meth == 0 || t < t_eq )
    {
        dt = dt_min;
    }
    
    // Idea: adaptative timestep from Picard error.
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

    // Update time.
    //t = t + dt;
    t += dt;

    // Output variables.
    out(0) = t;
    out(1) = dt;

    return out;
}



////////////////////////////////////////////////////
////////////////////////////////////////////////////
// Flow line functions.

ArrayXd f_bed(double L, int n, int bed_exp, \
              double y_0, double y_p, double x_1, double x_2, \
              int smooth_bed, double sigma_gauss, ArrayXd sigma)
{
    // Prepare variables.
    ArrayXd bed(n);
    //ArrayXd x = ArrayXd::LinSpaced(n, 0.0, L); 
    ArrayXd x = sigma * L; 

    // Number of points at the edges of the array that are not smoothed out.
    int p = 3;

    // MISMIP experiments bedrock.
    // Same bedrock as Schoof (2007).
    // Inverse sign to get a decreasing bedrock elevation.
    if (bed_exp == 1)
    {
        x = x / 750.0e3; 
        bed = 720.0 - 778.5 * x;
    }
    else if (bed_exp == 3)
    {
        x = x / 750.0e3; 
        // Schoof: 2184.8. Daniel: 2148.8.
        bed = 729.0 - 2148.8 * pow(x, 2) \
                    + 1031.72 * pow(x, 4) + \
                    - 151.72 * pow(x, 6);
    }
    else if (bed_exp == 4)
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
                    //c_x1 = c_x1 + 1;
                    c_x1 += 1;
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
                    //c_x2 = c_x2 + 1;
                    c_x2 += 1;	
                }
                    
                // Bedrock function.
                bed(i) = y_2 - 5.0e-3 * ( x(i) - x_2 );
            } 
        }
    }

    // Potential smooth bed.
    if ( smooth_bed == 1 )
    {
        // Gaussian smooth. Quite sensitive to p value (p=5 for n=250).
        //bed = gaussian_filter(bed, sigma_gauss, p, n);

        // Running mean.
        bed = running_mean(bed, 3, n);
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

double f_melt(double T_oce, double T_0, double rho, double rho_w, \
              double c_po, double L_i, double gamma_T, int meth)
{
    double M;

    // Linear.
    if ( meth == 0 )
    {
        M = ( T_oce - T_0 ) * gamma_T * ( rho_w * c_po ) / ( rho * L_i );
    }

    // Quadratic.
    else if ( meth == 1 )
    {
        M = pow((T_oce - T_0), 2) * gamma_T * pow( ( rho_w * c_po ) / ( rho * L_i ), 2);
    }
    
    return M;
}


// SMB spatial distribution depends on the position of the ice.
// It does not follow the ice sheet, but rather there's a certain 
// value for each position x.

ArrayXd f_smb(ArrayXd sigma, double L, double S_0, \
              double x_mid, double x_sca, double x_varmid, \
              double x_varsca, double dlta_smb, double var_mult, \
              double smb_stoch, double t, double t_eq, int n, bool stoch)
{
    // Variables
    ArrayXd x(n), S(n); 
    double stoch_pattern, smb_determ;

    // Horizontal dimension to define SBM.
    x = L * sigma;

    // Start with constant accumulation value for smoothness.
    if ( t < t_eq || stoch == false)
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


ArrayXd f_q(ArrayXd u_bar, ArrayXd H, double H_f, double t, double t_eq, \
               double rho, double rho_w, double m_dot, double M, int calving_meth, int n)
{
    // Local variables.
    ArrayXd q(n);
    double H_mean;

    // Flux defined on velocity grid. Staggered grid.
    for (int i=0; i<n-1; i++)
    {
        // Vieli and Payne (2005) discretization.
        q(i) = u_bar(i) * 0.5 * ( H(i+1) + H(i) );

        // "One-before-last" approach.
        //q(i) = ub(i) * H(i+1);
    }   
        
    // GL flux definition (Vieli and Payne, 2005).
    if ( calving_meth == 0 )
    {
        //q(n-1) = u_bar(n-1) * H(n-1);

        // We impose sythetic reduction potentially cause by ocean cooling/warming.
        q(n-1) = u_bar(n-1) * H(n-1);
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
            q(n-1) = u_bar(n-1) * H(n-1);

            // Schoof (2007).
            //q(n-1) = H(n-1) * 0.5 * ( ub(n-1) + ub(n-2) );
        }
        
        // Calving after equilibration.
        else
        {
            // Prefactor to account for thickness difference in last grid point.
            // GL is defined in the last velocity grid point.
            // H(n-1) is not precisely H_f so we need a correction factor.
            
            // Successful MISMIP experiments but GL 
            //too retreated compared to Christian et al. (2022).
            //q(n-1) = ( u_bar(n-1) + ( H_f / H(n-1) ) * m_dot ) * H(n-1);

            // This seems to work as Christian et al. (2022).
            // Velocity evaluated at (n-2) instead of last point (n-1).
            //q(n-1) = ( u_bar(n-2) + ( H_f / H(n-1) ) * m_dot ) * H(n-1);

            // Ice flux at GL computed on the ice thickness grid. Schoof (2007).
            // Too retreated
            // LAST ATTEMPT.
            // ALMOST GOOD, A BIT TOO RETREATED FOR N=350 POINTS.
            q(n-1) = H(n-1) * 0.5 * ( u_bar(n-1) + u_bar(n-2) + ( H_f / H(n-1) ) * m_dot );

            // Ice flux at GL computed on the ice thickness grid. Schoof (2007).
            // Mean.
            // Works for stochastic! A more retreatesd ice sheet.
            //H_mean = 0.5 * ( H(n-1) + H_f ); 
            //q(n-1) = H_mean * 0.5 * ( ub(n-1) + ub(n-2) + ( H_f / H_mean ) * m_dot );
            //q(n-1) = H_mean * 0.5 * ( ub(n-1) + ub(n-2) + m_dot );

            // Previous grid points as the grid is staggered. Correct extension!!
            // GOOD RESULTS for stochastic with n = 350.
            //q(n-1) = H(n-1) * 0.5 * ( u_bar(n-2) + u_bar(n-3) + ( H_f / H(n-1) ) * m_dot );

            // GL too advanced for n = 500.
            //q(n-1) = H_f * 0.5 * ( ub(n-1) + ub(n-2) + m_dot );

            // Slightly retreated.
            //q(n-1) = H_mean * ( ub(n-1) + m_dot );

            // GL too advanced for n = 500.
            //q(n-1) = H_f * ( ub(n-2) + m_dot );

            // H_n.
            // Too retreated.
            //q(n-1) = H(n-1) * ( ub(n-1) + m_dot );
        }
    }

    // Deterministic calving from sub-shelf melting (e.g., Favier et al., 2019).
    else if ( calving_meth == 2 )
    {
        // No calving during equilibration.
        if ( t < t_eq )
        {
            // Vieli and Payne (2005).
            q(n-1) = u_bar(n-1) * H(n-1);
        }
        
        // Calving after equilibration.
        else
        {
            q(n-1) = H(n-1) * ( u_bar(n-1) + M );
            //q(n-1) = ( H(n-1) + M ) * u_bar(n-1);
        }
    }

    return q;
}


Array2d f_L(ArrayXd H, ArrayXd q, ArrayXd S, ArrayXd bed, \
          double dt, double L, ArrayXd ds, int n, double rho, \
          double rho_w, double M, int dL_dt_num_opt, int dL_dt_den_opt)
{
    //Local variables.
    Array2d out;
    double num, den, dL_dt;
    
    
    // DISCRETIZATION OPTIONS.
    // MIT finite difference calculator: https://web.media.mit.edu/~crtaylor/calculator.html
    
    // Accumulation minus flux (reverse sign). 

    num = q(n-1) - q(n-2) - L * ds(n-2) * S(n-1);
    den = H(n-1) - H(n-2) + ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );

    /*
    // Two-point backward-difference.
    if ( dL_dt_num_opt == 1)
    {
        // Standard.
        num = q(n-1) - q(n-2) - L * ds * S(n-1);

        // Sub-shelf melting at the grounding line M.
        //num = q(n-1) - q(n-2) + L * ds * ( M - S(n-1) );
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
    // Grounding line does not advance/retreate.
    else if ( dL_dt_den_opt == 3 ) 
    {
        den = 3.0 * H(n-1) - 4.0 * H(n-2) + H(n-3) + ( rho_w / rho ) * \
             ( bed(n-1) - bed(n-2) );
    }
    else if ( dL_dt_den_opt == 4 ) 
    {
        den = 3.0 * H(n-1) - 4.0 * H(n-2) + H(n-3) + ( rho_w / rho ) * \
              0.5 * ( 3.0 * bed(n-1) - 4.0 * bed(n-2) + bed(n-3) );
    }
    // Five-point asymmetric backward-difference deffinition. Unstable!
    else if ( dL_dt_den_opt == 5 ) 
    {
        // Local vraiables.
        double D_5;

        D_5 = ( 137.0 * H(n-1) - 300.0 * H(n-2) + 300.0 * H(n-3) + \
                - 200.0 * H(n-4) + 75.0 * H(n-5) - 12.0 * H(n-6) ) / 60.0;
        
        // Simple bedrock slope.
        den = D_5 + ( rho_w / rho ) * ( bed(n-1) - bed(n-2) );
    }
    */

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
ArrayXd f_H(ArrayXd u_bar, ArrayXd H, ArrayXd S, ArrayXd sigma, \
            double dt, ArrayXd ds, ArrayXd ds_inv, ArrayXd ds_sym, int n, \
            double L, double D, double rho, double rho_w, \
            double dL_dt, ArrayXd bed, ArrayXd q, double M, int H_meth, double t, double t_eq)
{
    // Local variables.
    ArrayXd H_now(n), dx_inv(n-1), dx_sym_inv(n-1);

    double L_inv  = 1.0 / L;
    
    dx_inv     = L_inv * ds_inv;
    dx_sym_inv = L_inv * ds_sym;

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
            //H_now(i) = H(i) + dt * ( dx_inv(i) * ( sigma(i) * dL_dt * 0.5 * ( H(i+1) - H(i-1) ) + \
            //                                - ( q(i) - q(i-1) ) ) + S(i) );
            

            // Centred in sigma, upwind in flux. Unevenly-spaced horizontal grid.
            H_now(i) = H(i) + dt * ( sigma(i) * dL_dt * ( H(i+1) - H(i-1) ) * dx_sym_inv(i) + \
                                            - dx_inv(i) * ( q(i) - q(i-1) ) + S(i) );
        }
        
        // Symmetry at the ice divide (i = 1).
        H_now(0) = H_now(2);
        
        // Lateral boundary: sigma(n-1) = 1.
        // Sub-shelf melt directly on the flux.
        // Note that ds has only (n-1) points.
        H_now(n-1) = H(n-1) + dt * ( ds_inv(n-2) * L_inv * \
                                        ( dL_dt * ( H(n-1) - H(n-2) ) + \
                                            - ( q(n-1) - q(n-2) ) ) + S(n-1) );

        
        // Make sure that grounding line thickness is above minimum?
        //H_now(n-1) = max( (rho_w/rho)*D, H_now(n-1));
    }
    
    // Implicit scheme.
    /*
    else if ( H_meth == 1 )
    {
        // Local variables.
        ArrayXd A(n), B(n), C(n), F(n);
        ArrayXd gamma = dt / ( 2.0 * ds * L );

        // Implicit scheme. REVISE TRIDIAGONAL MATRIX.
        for (int i=1; i<n-1; i++)
        {
            // Tridiagonal vectors.
            A(i) = - u_bar(i-1) + sigma(i) * dL_dt;
            B(i) = 1.0 + gamma(i) * ( u_bar(i) - u_bar(i-1) );
            C(i) = u_bar(i) - sigma(i) * dL_dt;

            // Inhomogeneous term.
            F(i) = H(i) + S(i) * dt;
        }

        // Vectors at the boundaries.
        A(0) = 0.0;
        B(0) = 1.0 + gamma(0) * u_bar(0);
        //B(0) = 1.0 + gamma * ( u_bar(1) - u_bar(0) );
        C(0) = u_bar(0);                  

        A(n-1) = - u_bar(n-2) + sigma(n-1) * dL_dt;
        B(n-1) = 1.0 + gamma(n-2) * ( u_bar(n-1) - u_bar(n-2) );
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
                    - u_bar(n-2) * H_now(n-2) ) / ( 1.0 + gamma * ( - 2.0 * dL_dt \
                                                + u_bar(n-1) - u_bar(n-2) ) ); 
        
        //H_now(n-1) = H(n-1) + S(n-1) * dt - gamma * ( 2.0 * dL_dt * H_now(n-2) \
                    - u_bar(n-2) * H_now(n-2) ) / ( 1.0 + gamma * ( - 2.0 * dL_dt \
                                                - u_bar(n-2) ) ); 
        
        H_now(n-1) = min( D * ( rho_w / rho ), H_now(n-1) );
        
        
        //H_now(n-1) = D * ( rho_w / rho );
        //H_now(n-1) = min( D * ( rho_w / rho ), H(n-2) );

        //cout << "\n H_now = " << H_now;   
    }
    */

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


ArrayXXd f_visc(ArrayXXd theta, ArrayXXd u, ArrayXXd visc, ArrayXd H, ArrayXd tau_b, \
                ArrayXd u_bar, ArrayXd dz, \
                double theta_act, ArrayXd ds, ArrayXd ds_inv, ArrayXd ds_sym, double L, \
                Array2d Q_act, Array2d A_0, double n_gln, double R, double B, double n_exp, \
                double eps, double t, double t_eq, double sec_year, \
                int n, int n_z, int vel_meth, double A, bool visc_therm, \
                double t_eq_A_theta, double visc_min, \
                double visc_max, double visc_0)
{
    ArrayXXd out(n,5*n_z+2), u_x(n,n_z), u_z(n,n_z), \
             strain_2d(n,n_z), A_theta(n,n_z), B_theta(n,n_z);
    ArrayXd u_bar_x(n), strain_1d(n), visc_bar(n), B_theta_bar(n);
    
    ArrayXd dx_inv = 1.0 / ( ds * L );
    ArrayXd dz_inv = 1.0 / dz;
    ArrayXd dx_sym_inv = 1.0 / ( ds_sym * L );

    
    // Equilibration and constant viscosity experiments.
    if ( t < t_eq )
    {
        // Full viscosity.
        visc = ArrayXXd::Constant(n, n_z, visc_0); // 1.0e15, 0.5e17
        
        // Vertically-averaged viscosity.
        visc_bar = ArrayXd::Constant(n, visc_0);

        // We assume a constant ice rate factor during equilibration.
        A_theta = ArrayXXd::Constant(n, n_z, A);
    }
    
    // After equilibration.
    else 
    {
        // Potential viscosity dependency on temperature.
        // We calculate B from A(T,p).
        if ( visc_therm == false )
        {
            // We assume a constant ice rate factor during equilibration.
            A_theta = ArrayXXd::Constant(n, n_z, A);
        }
        
        else if ( visc_therm == true )
        {
            // Calculate temperature-dependent rate factor if thermodynamics is switched on.
            // Arrhenius law A(T,p) equivalent to A(T').
            // Eq. 4.15 and 6.54 (Greve and Blatter, 2009).
            for (int i=0; i<n; i++)
            {
                for (int j=0; j<n_z; j++)
                {
                    // Rate factor. We consider two temperature regimes (Greve and Blatter, 2009).
                    if ( theta(i,j) < theta_act )
                    {
                        A_theta(i,j) = A_0(0) * exp(- Q_act(0) / (R * theta(i,j)) );
                    }
                    else
                    {
                        A_theta(i,j) = A_0(1) * exp(- Q_act(1) / (R * theta(i,j)) );
                    }
                }
            }

            // Avoid unnecessary loops.
            //A_theta = (theta < theta_act).select(A_0(0) * exp(- Q_act(0) / (R * theta) ), A_theta);
            //A_theta = (theta >= theta_act).select(A_0(1) * exp(- Q_act(1) / (R * theta) ), A_theta);

        }

        // Associated rate factor. A: [Pa^-3 yr^-1]
        B_theta = pow(A_theta, (-1 / n_gln) );

        // Vertically averaged B for the SSA.
        B_theta_bar = B_theta.rowwise().mean();

        // We use the median rather than the mean?
        //B_theta_bar = f_median(B_theta, n, n_z);

        // SSA solver.
        if ( vel_meth == 1 )
        {
            // Horizontal derivatives.
            for (int i=1; i<n-1; i++)
            {
                //u_bar_x(i) = 0.5 * ( u_bar(i+1) - u_bar(i-1) ) * dx_inv(i);
                u_bar_x(i) = ( u_bar(i+1) - u_bar(i) ) * dx_inv(i);
                //u_bar_x(i) =  ( u_bar(i+1) - u_bar(i-1) ) * dx_sym_inv(i);
            }

            // Boundary derivatives.
            u_bar_x(0)   = ( u_bar(1) - u_bar(0) ) * dx_inv(0);
            u_bar_x(n-1) = ( u_bar(n-1) - u_bar(n-2) ) * dx_inv(n-2);
            
            // Regularization term to avoid division by 0. 
            strain_1d = pow(u_bar_x,2) + eps;

            // Viscosity potentially dependending on visc_term. 
            visc_bar = 0.5 * B_theta_bar * pow(strain_1d, n_exp);
        }

        // DIVA solver.
        else if ( vel_meth == 2 )
        {     
            // Horizontal derivative du/dx as defined in Eq. 21 (Lipscomb et al., 2019).
            for (int i=1; i<n-1; i++)
            {
                // Centred.
                //u_bar_x(i) = 0.5 * ( u_bar(i+1) - u_bar(i-1) ) * dx_inv(i);

                // Centred with unevenly-spaced grid.
                u_bar_x(i) =  ( u_bar(i+1) - u_bar(i-1) ) * dx_sym_inv(i);

                // Forward.
                //u_bar_x(i) = u_bar(i+1) - u_bar(i);

                // Backwards.
                //u_bar_x(i) = u_bar(i) - u_bar(i-1);
            }
            u_bar_x(0)   = ( u_bar(1) - u_bar(0) ) * dx_inv(0);
            u_bar_x(n-1) = ( u_bar(n-1) - u_bar(n-2) ) * dx_inv(n-2);

            // Vertical shear stress du/dz from Eq. 36 Lipscomb et al.
            for (int j=0; j<n_z; j++)
            {
                // Fill matrix since horizontal derivates have no vertical depedency since
                // take the derizative of the vertically-averaged velocity.
                u_x.col(j) = u_bar_x;

                for (int i=0; i<n; i++)
                {
                    // Velocity vertical derivative du/dz. Eq. 36 Lipscomb et al. (2019).
                    u_z(i,j) = tau_b(i) * ( H(i) - j * dz(i) ) / ( visc(i,j) * H(i) );
                }
            }

            // Strain rate and regularization term to avoid division by 0. 
            strain_2d = pow(u_x,2) + 0.25 * pow(u_z,2) + eps;

            // Viscosity option dependending on visc_term via B_theta.
            visc = 0.5 * B_theta * pow(strain_2d, n_exp);

            // Vertically averaged viscosity.
            visc_bar = visc.rowwise().mean();

        }

        // Blatter-Pattyn.
        else if ( vel_meth == 3 )
        {     
            // Spatial derivatives du/dx, du/dz.
            for (int j=1; j<n_z-1; j++)
            {
                // Centred (evenly-spaced grid in the z axis).
                u_z.col(j) = 0.5 * ( u.col(j+1) - u.col(j-1) ) * dz_inv(j);
                
                // Forwards.
                //u_z.col(j) = ( u.col(j+1) - u.col(j) ) * dz_inv(j);

                // Forwards. It works.
                //u_z.col(j) = u.col(j) - u.col(j-1);
            }

            for (int i=1; i<n-1; i++)
            {
                // Centred.
                //u_x.row(i) = 0.5 * ( u.row(i+1) - u.row(i-1) ) * dx_inv(i);

                // Centred with unevenly-spaced grid.
                u_x.row(i) = ( u.row(i+1) - u.row(i-1) ) * dx_sym_inv(i);
                
                // Forwards.
                //u_x.row(i) = ( u.row(i+1) - u.row(i) ) * dx_inv(i);

                // Try backwards instead. It works.
                //u_x.row(i) = u.row(i) - u.row(i-1);
            }

            // Boundaries.
            // Two-point derivative.
            /*
            u_x.row(0) = ( u.row(1) - u.row(0) ) * dx_inv(0);
            u_z.col(0) = ( u.col(1) - u.col(0) ) * dz_inv(0);

            u_x.row(n-1)   = ( u.row(n-1) - u.row(n-2) ) * dx_inv(n-2);
            u_z.col(n_z-1) = ( u.col(n_z-1) - u.col(n_z-2) ) * dz_inv(n_z-1);
            */

            // Three-point derivative.
            // MIT.
            // du/dz = 0.5 * ( 3.0 * u(n_z-1) - 4.0 * u(n_z-2) + 1.0 * u(n_z-3) ) 
            
            u_x.row(0) = 0.5 * ( u.row(0) - 4.0 * u.row(1) + 3.0 * u.row(2) ) * dx_inv(0);
            u_z.col(0) = 0.5 * ( u.col(0) - 4.0 * u.col(1) + 3.0 * u.col(2) ) * dz_inv(0);

            //u_x.row(n-1)   = 0.5 * ( u.row(n-1) - 4.0 * u.row(n-2) + 3.0 * u.row(n-3) ) * dx_inv(n-2);
            //u_z.col(n_z-1) = 0.5 * ( u.col(n_z-1) - 4.0 * u.col(n_z-2) + 3.0 * u.col(n_z-3) ) * dz_inv(n-1);
            u_x.row(n-1)   = ( u.row(n-1) - u.row(n-2) ) * dx_inv(n-2);
            u_z.col(n_z-1) = ( u.col(n_z-1) - u.col(n_z-2) ) * dz_inv(n_z-1);
            
            // Try Vieli and Payne assymetric differences.
            // Vieli 1 (Eq. B12).
            // du/dz = ( 4.0 * u(n_z-1) - 3.0 * u(n_z-2) - 1.0 * u(n_z-3) ) / 3.0
            /*
            u_x.row(0) = ( 4.0 * u.row(0) - 3.0 * u.row(1) - u.row(2) ) / 3.0;
            u_z.col(0) = ( 4.0 * u.col(0) - 3.0 * u.col(1) - u.col(2) ) / 3.0;

            u_x.row(n-1)   = ( 4.0 * u.row(n-1) - 3.0 * u.row(n-2) - u.row(n-3) ) / 3.0;
            u_z.col(n_z-1) = ( 4.0 * u.col(n_z-1) - 3.0 * u.col(n_z-2) - u.col(n_z-3) ) / 3.0;
            */
            

            // Strain rate and regularization term to avoid division by 0. 
            strain_2d = pow(u_x,2) + 0.25 * pow(u_z,2) + eps;

            // Viscosity option dependending on visc_term.
            visc = 0.5 * B_theta * pow(strain_2d, n_exp);

            // Vertically-averaged viscosity.
            visc_bar = visc.rowwise().mean();

        }

    }

    // Allocate output variables.
    out << visc, visc_bar, u_bar_x, u_x, u_z, strain_2d, A_theta;

    return out;
}	


ArrayXd f_C_bed(ArrayXd C_ref, ArrayXXd theta, ArrayXd H, double t, double t_eq, double theta_max, \
                double theta_frz, double C_frz, double C_thw, double rho, double g, int fric_therm, int n)
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

            // Reduction by a given factor as we approach melting.
            else
            {
                C_bed(i) = 0.25 * C_ref(i);
            }
        }
        
        // Normalized basal temperature with pressure melting point [0,1].
        //theta_norm = abs( theta.block(0,0,n,1) - theta_frz ) / abs(theta_max - theta_frz);
        //C_bed      = C_thw * theta_norm + C_frz * ( 1.0 - theta_norm );

    }

    // Overburden pressure of ice. C_bed = N.
    // Seems to be too high, arbitrary factor 0.001 !!!!!!!???
    else if (fric_therm == 2 && t > t_eq)
    {
        C_bed = 0.001 * rho * g * H;

        // g in m/s² --> m/yr^2 SOMETHING MIGHT BE WRONG HERE!
        //C_bed = 0.001 * rho * g * H * pow(sec_year, 2);
    }

    // Friction coefficient given by reference value.
    else 
    {
        C_bed = C_ref;
    }

    return C_bed;
}


ArrayXXd f_theta(ArrayXXd theta, ArrayXd ub, ArrayXd H, ArrayXd tau_b, ArrayXd Q_fric, \
                 ArrayXd sigma, ArrayXd dz, double theta_max, double T_air, double kappa, double k, \
                 double dt, double G_k, ArrayXd ds, double L, \
                 double dL_dt, double t, double t_eq, ArrayXd w, int n, int n_z, int vel_meth, \
                 ArrayXXd strain_2d)
{
    ArrayXXd theta_now(n,n_z);
    ArrayXd dx_inv(n-1), dz_inv(n), dz_2_inv(n), Q_f_k(n);

    //ArrayXXd theta_now(n,n_z), dz_inv(n,n_z), dz_2_inv(n,n_z);
    //ArrayXd Q_f_k(n);
    
    //double dx_inv;
 
    // Evenly-spaced vertical grid, though x-dependency via ice thickness.
    dz_inv   = 1.0 / dz;
    dz_2_inv = pow(dz_inv, 2);
    dx_inv   = 1.0 / (ds * L);

    // Frictional heat units. // [Pa m yr^-1] = [J yr^-1 m^-2] = [W m^-2] => [K / m]
    Q_f_k = Q_fric / k;     
    
    // Zero frictional heat test.
    //Q_f_k = ArrayXd::Zero(n);

    // Temperature integration.
    for (int i=1; i<n; i++)
    {
        for (int j=1; j<n_z-1; j++)
        {
            // Vertical advection.
            // Since w < 0 we need an opposite discretization scheme in theta.
            // Vertical velocity con z-dependency.
            theta_now(i,j) = theta(i,j) + dt * ( kappa * dz_2_inv(i) * \
                            ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,j) - theta(i-1,j) ) * dx_inv(i) + \
                            ( theta(i,j) - theta(i,j-1) ) * ( - w(j) ) * dz_inv(i) );

            // Unevenly-spaced vertical grid.
            // It needs correction terms in the derivative to avoid numerical instabilities.
            /*
            theta_now(i,j) = theta(i,j) + dt * ( kappa * dz_2_inv(i,j) * \
                            ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,j) - theta(i-1,j) ) * dx_inv + \
                            ( theta(i,j+1) - theta(i,j) ) * ( - w(j) ) * dz_inv(i,j) );
            */
        }
        
        // Boundary conditions. Geothermal heat flow at the base.
        // We add friciton heat contribution Q_f_k.
        theta_now(i,0) = theta_now(i,1) + dz(i) * ( G_k + Q_f_k(i) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,0) - theta(i-1,0) ) * dx_inv(i); 

        // w(z=0) = 0 right??

        // Surface.
        theta_now(i,n_z-1) = T_air;
    }

    // Vertical loop for x = 0.
    /*
    for (int j=1; j<n_z-1; j++)
    {
        theta_now(0,j) = theta(0,j) + dt * kappa * dz_2_inv(0) * \
                        ( theta(0,j+1) - 2.0 * theta(0,j) + theta(0,j-1) ) ;
    }
    */

    // Due to symmetry theta_now(0,j) = theta_now(2,j). Ice divide.
    theta_now.row(0) = theta_now.row(2);
    
    // Boundary conditions at x = 0.
    //theta_now(0,0)     = theta_now(0,1) + dz(0) * G_k; 
    //theta_now(0,n_z-1) = T_air;

    // For the DIVA solver, consider the strain heat contribution.
    if ( vel_meth == 2 )
    {
        theta_now = theta_now + ( kappa / k ) * strain_2d * dt;
    }

    // Pressure melting point as the upper bound.
    // theta = (theta.array() > 273.15).select(273.15, theta);
    theta_now = (theta_now.array() > theta_max).select(theta_max, theta_now);


    // Test for the grounding line column.
    theta_now.row(n-1) = theta_now.row(n-2);

    return theta_now;
}
	

ArrayXXd F_int_all(ArrayXXd visc, ArrayXd H, ArrayXd dz, int n_z, int n) {
    
    ArrayXXd F_all(n,n_z+1), F_1(n,n_z);
    ArrayXd F_2(n);

    double z, sum_1, sum_2, value_1, value_2, H_minus_z;
    int n_2 = 2;
    
    // Horizontal loop.
    for (int i = 0; i < n; ++i) 
    {
        z = 0;
        sum_1 = 0;
        sum_2 = 0;

        
        // Vertical integration.
        for (int j = 0; j < n_z; ++j) 
        {
            // Current vertical height.
            H_minus_z = ( H(i) - z ) / H(i);
            
            // F_2 integral is a 1D array.
            value_2 = pow(H_minus_z, n_2) / visc(i, j);
            sum_2  += value_2;

            // F_1 integral is a 2D array.
            value_1  = H_minus_z / visc(i, j);
            sum_1   += value_1;
            F_1(i,j) = sum_1;

            // Update vertical height.
            z += dz(i);
        }
        
        // Integral value.
        F_1.row(i) = dz(i) * F_1.row(i);
        F_2(i)     = dz(i) * sum_2;
        
    }
    
    // Allocate solutions.
    //F_all.block(0,0,n,1)   = F_2;
    //F_all.block(0,1,n,n_z) = F_1;

    F_all << F_2, F_1;
    
    return F_all;
}




ArrayXXd f_u(ArrayXXd u, ArrayXd u_bar, ArrayXd beta, ArrayXd C_bed, ArrayXXd visc, \
             ArrayXd H, ArrayXd dz, double sec_year, double t, double t_eq, \
             double m, int vel_meth, int n_z, int n)
{
    ArrayXXd out(n,4), F_1(n,n_z), F_all(n,n_z+1); //out(n,n_z+3)
    ArrayXd beta_eff(n), ub(n), F_2(n), tau_b(n), Q_fric(n);

    //ArrayXXd out(n,n_z+6),

    // For constant visc or SSA, ub = u_bar.
    if ( vel_meth == 0 || vel_meth == 1 )
    {
        // Vel definition. New beta from ub.
        ub = u_bar;

        // beta = beta_eff, ub = u_bar for SSA.
        beta_eff    = C_bed * pow(ub, m - 1.0);
        beta_eff(0) = beta_eff(1);
    }

    // For DIVA solver, Eq. 32 Lipscomb et al.
    else if ( vel_meth == 2 )
    {
        // Useful integral. n_z-1 as we count from 0.
        F_all = F_int_all(visc, H, dz, n_z, n);
        
        F_2 = F_all.block(0,0,n,1);
        F_1 = F_all.block(0,1,n,n_z);
        
        
        // REVISE HOW BETA IS CALCULATED
        // Obtain ub from new u_bar and previous beta (Eq. 32 Lipscomb et al.).
        ub = u_bar / ( 1.0 + beta * F_2 );

        // Impose BC here?
        ub(0) = - ub(1);

        // Beta definition. New beta from ub.
        beta = C_bed * pow(ub, m - 1.0);

        // DIVA solver requires an effective beta.
        beta_eff = beta / ( 1.0 + beta * F_2 );

        // Impose ice divide boundary condition.
        beta(0)     = beta(1);
        beta_eff(0) = beta_eff(1);

        
        // Now we can compute u(x,z) from beta and visc.
        for (int j=0; j<n_z; j++)
        {
            // Integral only up to current vertical level j.
            // 2D velocity field (Eq. 29 Lipscomb et al., 2019).
            // With F_int_all.
            u.col(j) = ub * ( 1.0 + beta * F_1.col(j) );
        }   
        
        // Impose BC here?
        u.row(0) = - u.row(1);
        
    }

    // Blatter-Pattyn velocity solver.
    else if ( vel_meth == 3 )
    {
        // Basal velocity from velocity matrix.
        ub = u.col(0);
        
        // beta = beta_eff, ub = u_bar for SSA.
        beta_eff    = C_bed * pow(ub, m - 1.0);
    }
    

    // Calculate basal shear stress with ub (not updated in Picard iteration). 
    tau_b    = beta_eff * ub;
    tau_b(0) = tau_b(1);    // Symmetry ice divide. Avoid negative tau as u_bar(0) < 0. 

    // Frictional heat. [Pa · m / yr] --> [W / m^2].
    Q_fric    = tau_b * ub / sec_year;
    Q_fric(0) = Q_fric(1);


    // Allocate output variables.
    out << beta_eff, tau_b, Q_fric, ub;

    
    return out;
}


ArrayXXd solver_2D(int n, int n_z, ArrayXd dx, ArrayXd dz, ArrayXXd visc, ArrayXd F, ArrayXXd u_0) 
{

    ArrayXd dz_2_inv = 1.0 / pow(dz,2);
    ArrayXd gamma    = 4.0 / pow(dx, 2); // 4.0 / pow(dx, 2)

    // Inhomogeneous term: A*x = b.
    VectorXd b = VectorXd::Zero(n * n_z);

    // Build initial guess from previous velocity solution u_0.
    // Border should be zero as the solver does not include boundary conditions.
    /*
    u_0.col(0)     = ArrayXd::Zero(n);
    u_0.col(n_z-1) = ArrayXd::Zero(n);
    u_0.row(0)     = ArrayXd::Zero(n_z);
    u_0.row(n-1)   = ArrayXd::Zero(n_z);

    Map<VectorXd> x_0(u_0.data(), n*n_z);
    */
    
    // Initialize a triplet list to store non-zero entries.
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // Reserve memory for triplets.
    // 5 unkowns in a n*n_z array.
    tripletList.reserve(5 * (n-2) * (n_z-2));  

    
    // Loop through grid points
    for (int i=1; i<n-1; i++) 
    {
        for (int j=1; j<n_z-1; j++) 
        {
            // New index.
            int idx = i*n_z + j;

            // Compute coefficients. This works!!
            /*
            double c_x1 = gamma * visc(i+1,j);
            double c_x  = gamma * visc(i,j);

            double c_z1 = dz_2_inv(i) * visc(i,j+1);
            double c_z  = dz_2_inv(i) * visc(i,j);
            */
            
            double c_x1 = gamma(i) * visc(i+1,j);
            double c_x  = gamma(i-1) * visc(i,j);

            double c_z1 = dz_2_inv(i) * visc(i,j+1);
            double c_z  = dz_2_inv(i) * visc(i,j);
            
           
           // From the staggered grid definition, we should not take points at n_z-1 (j+1)
           // to calculate the velocities at j = n_z-2. 
           if ( j == n_z-2 )
           {
            c_z1 = 0.0;
           }
            

            // Add non-zero entries to the triplet list
            tripletList.push_back(T(idx, idx, - ( c_x1 + c_x + c_z1 + c_z )));
            tripletList.push_back(T(idx, idx+n_z, c_x1));
            tripletList.push_back(T(idx, idx-n_z, c_x));
            tripletList.push_back(T(idx, idx+1, c_z1));
            tripletList.push_back(T(idx, idx-1, c_z));
            

            // Fill vector b.
            b(idx) = F(i);
        }
    }

    // Set the triplets in the sparse matrix
    // declares a column-major sparse matrix type of double.
    SparseMatrix<double> A_sparse(n*n_z, n*n_z); 
    
    // Define your sparse matrix A_spare from triplets.
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    // Solver.
    BiCGSTAB<SparseMatrix<double> > solver;
    //solver.compute(A_sparse);

    // Preconditioner. It works as fast as using the previous vel sol as guess x_0.
    // If we use an initial guess x_0 from previous iter, do not use a preconditioner.
    IncompleteLUT<double> preconditioner;
    preconditioner.setDroptol(1.0e-4); // 1.0e-4. Set ILU preconditioner parameters
    solver.preconditioner().compute(A_sparse);
    solver.compute(A_sparse);
    

    // Set tolerance and maximum number of iterations.
    int maxIter = 100;                   // 100, 50
    double tol = 1.0e-3;                // 1.0e-3, 1.0e-2
    solver.setMaxIterations(maxIter);
    solver.setTolerance(tol);

    // Solve without guess (assumes x = 0).
    VectorXd x = solver.solve(b);

    // Solve with first guess x_0.
    //solver.compute(A_sparse);
    //VectorXd x = solver.solveWithGuess(b, x_0);
    
    cout << "\n #iterations:     " << solver.iterations();
    cout << "\n Estimated error: " << solver.error();
    
    /* ... update b ... */
    // THINK ABOUT THIS!!!!!!!!!!!
    //x = solver.solve(b); // solve again??? No need to update b in our case.
        

    /* Copy values from x into u (no memory reallocation).
    Unlike u = x.reshaped<RowMajor>(n,n_z), where there is reallocation.

    We create a mapped view u using Eigen::Map. 
    This mapped view u will have the desired shape without memory reallocation.
    
    When you add elements to a dynamically allocated data structure and it reaches its 
    current capacity, the data structure may need to allocate a larger chunk of memory, 
    copy the existing data to the new memory location, and then release the old memory.
    
    In row-major order, the elements of a matrix are stored in memory row by row. 
    This means that consecutive elements in the same row are stored next to each other in memory.*/
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> u(x.data(), n, n_z);

    return u;
}


ArrayXXd vel_solver(ArrayXd H, ArrayXd ds, ArrayXd ds_inv, ArrayXd ds_sym, ArrayXd dz, int n, int n_z, ArrayXd visc_bar, \
                    ArrayXd bed, double rho, double rho_w, double g, double L, ArrayXd C_bed, \
                    double t, ArrayXd beta, double A_ice, ArrayXXd A_theta,
                    double n_gln, ArrayXXd visc, ArrayXXd u, ArrayXXd u_z, bool visc_therm, \
                    int vel_meth, double t_eq, double tf)
{
    ArrayXXd out(n,n_z+1), u_sol(n,n_z); 

    ArrayXd u_bar(n), dhds(n), visc_H(n), h(n), A_bar(n), \
            A(n), B(n), C(n), F(n), dz_inv_2(n), ds_inv_2(n-1), gamma(n-1);

    double D, u_x_bc, L_inv;
    
    L_inv    = 1.0 / L;
    ds_inv_2 = pow(ds_inv, 2);
    dz_inv_2 = pow(1.0/dz, 2);

    // Note that ds has only (n-1) point rather than (n).
    gamma    = 4.0 * ds_inv_2 * pow(L_inv, 2); // Factor 2 difference from Vieli and Payne (2005).
    
    // Handy definitions.
    h = bed + H;           // Ice surface elevation.

    ///////////////////////////////////////
    ///////////////////////////////////////
    // Staggered grid (Vieli and Payne solutions, appendix).

    // SSA and DIVA solvers.
    if ( vel_meth == 1 || vel_meth == 2 )
    {
        // Handy definitions.
        visc_H = visc_bar * H;

        // Staggered grid (Vieli and Payne solutions, appendix).
        for (int i=1; i<n-1; i++)
        {
            // Surface elevation gradient. Centred stencil.
            dhds(i) = 0.5 * ( H(i) + H(i+1) ) * ( h(i+1) - h(i) ) * ds_inv(i);

            // Diagonal, B; lower diagonal, A; upper diagonal, C.
            A(i) = gamma(i-1) * visc_H(i);
            B(i) = - gamma(i) * ( visc_H(i) + visc_H(i+1) ) - beta(i);
            C(i) = gamma(i) * visc_H(i+1);
        }

        // Derivatives at the boundaries O(x).
        dhds(0)   = 0.5 * ( H(0) + H(1) ) * ( h(1) - h(0) ) * ds_inv(0);
        dhds(n-1) = H(n-1) * ( h(n-1) - h(n-2) ) * ds_inv(n-2); 
        //dhds(n-1) = H(n-1) * 0.5 * ( 3.0 * h(n-1) - 4.0 * h(n-2) + h(n-3) );

        // Tridiagonal boundary values. 
        A(0) = 0.0;
        B(0) = - gamma(0) * ( visc_H(0) + visc_H(1) ) - beta(0);
        C(0) = gamma(1) * visc_H(1); // gamma(0)

        A(n-1) = gamma(n-2) * visc_H(n-1);
        B(n-1) = - gamma(n-2) * visc_H(n-1) - beta(n-1);
        C(n-1) = 0.0;
        
        // Inhomogeneous term.
        F = rho * g * dhds * L_inv;

        //A = gamma * A;
        //C = gamma * C;

        // Grounding line sigma = 1 (x = L). 
        D = abs( min(0.0, bed(n-1)) );   

        // Imposed rate factor.
        if (visc_therm == 0)
        {
            u_x_bc = L * A_ice * pow( 0.25 * ( rho * g * H(n-1) * (1.0 - rho / rho_w) ), n_gln);
        }

        // Temperature-dependent ice rate factor.
        else if (visc_therm == 1)
        {
            // Vertically averaged ice rate factor.
            A_bar = A_theta.rowwise().mean();

            // Boundary condition.
            u_x_bc = L * A_bar(n-1) * pow( 0.25 * ( rho * g * H(n-1) * (1.0 - rho / rho_w) ), n_gln);
        }
           
        // TRIDIAGONAL SOLVER.
        u_bar = tridiagonal_solver(A, B, C, F, n); 

        // Replace potential negative values with given value. (result.array() < 0).select(0, result);
        // Sometimes, u_bar(1) < 0 so that u_bar(0) > 0 after the BC and the model crashes.
        // Necessary for SSA solver, not DIVA.
        //u_bar = (u_bar < 0.0).select(0.0, u_bar);
        u_bar = (u_bar < 0.0).select(0.25 * u_bar(2), u_bar);

        // This works but yields extremely thick ice at the divde.
        //u_bar = (u_bar < 0.0).select(0.0, u_bar);

        // Boundary conditions.
        // Ice divide: symmetry x = 0.
        u_bar(0)   = - u_bar(1);
        u_bar(n-1) = u_bar(n-2) + ds(n-2) * u_x_bc;
    }
    
    else if ( vel_meth ==  3 )
    {
        /*
        double dx = ds*L;
        double dx_inv = 1.0 / dx;
        double dx_2_inv = pow(dx_inv,2); 
        */
        ArrayXd dx(n-1), dx_inv(n-1), dx_2_inv(n-1);
        dx       = ds * L;
        dx_inv   = 1.0 / dx;
        dx_2_inv = pow(dx_inv,2);    

        //cout << "\n dx = " << dx;

        for (int i=0; i<n-1; i++)
        {
            // Surface elevation gradient.
            dhds(i) = ( h(i+1) - h(i) ) * dx_inv(i);

            // Unstable.
            //dhds(i) = 0.5 * ( h(i+1) - h(i-1) );
        }
        
        // Boundaries.
        dhds(n-1) = ( h(n-1) - h(n-2) ) * dx_inv(n-2); 

        // Inhomogeneous term.
        F = rho * g * dhds;

        // Blatter-Pattyn solution.
        u_sol = solver_2D(n, n_z, dx, dz, visc, F, u); 

        
        // VELOCITY BOUNDARY CONDITIONS.
        // Eq. 25, Pattyn (2003).
        for (int i=1; i<n-1; i++)
        {
            // Derivative from current velocity sol.
            // Centred differences and changed sign.
            
            //cout << "\n i              = " << i;
            //cout << "\n u_sol(i,1)     = " << u(i,1);

            // Use previous iteration solution to calculate basal friction.
            double alpha = 4.0 * ( u_sol(i,1) - u_sol(i-1,1) ) * abs( bed(i+1) - bed(i) ) * dx_2_inv(i) + \
                             0.5 * beta(i) * u(i,0) / visc(i,0);

            u_sol(i,0) = u_sol(i,1) - dz(i) * alpha;
            u_sol(i,0) = max(1.0, u_sol(i,0));
            
            //cout << "\n i          = " << i;
            //cout << "\n visc(i,1)    = " <<  visc(i,1);
            //cout << "\n u_sol(i,0) = " << u_sol(i,0);

            /*
            double alpha_0 = dz(i) * 4.0 * ( 0.5 * beta(i) * u_sol(i,0) / visc(i,0) + \
                                        0.5 * ( u_sol(i+1,0) - u_sol(i-1,0) ) *  \
                                            abs( bed(i+1) - bed(i) ) * dx_2_inv );
            
            u_sol(i,0) = ( 4.0 * u_sol(i,1) - u_sol(i,2) - alpha_0 ) / 3.0;
            */
            
            // Free surface. Pattyn (2003).
            //cout << "\n dx_2_inv     = " << dx_2_inv;
            //cout << "\n abs(du/dx)      = " << abs(u_sol(i+1,n_z-2) - u_sol(i,n_z-2)) * dx_inv;
            //cout << "\n abs(dh/dx) = " << abs(dhds(i)) * dx_inv;
            
            // Three-point vertical derivative. As Pattyn (2003).
            double alpha_h = dz(i) * 4.0 * ( u_sol(i+1,n_z-2) - u_sol(i,n_z-2) ) *  \
                                         abs(dhds(i)) * dx_2_inv(i);
            
            // MIT calculator.
            // du/dz = 0.5 * ( 3.0 * u(n_z-1) - 4.0 * u(n_z-2) + 1.0 * u(n_z-3) ) 
            u_sol(i,n_z-1) = ( 4.0 * u_sol(i,n_z-2) - u_sol(i,n_z-3) + 2.0 * alpha_h ) / 3.0;

            // Vieli 1 (Eq. B13). BEST ONE!
            // du/dz = ( - 4.0 * u(n_z-1) + 3.0 * u(n_z-2) + 1.0 * u(n_z-3) ) / 3.0
            //u_sol(i,n_z-1) = ( 4.0 * u_sol(i,n_z-2) + 3.0 * u_sol(i,n_z-3) - 3.0 * alpha_h ) / 4.0;

            // Test to avoid velocity differences between u_sol.col(n_z-1) and u_sol.col(n_z-2).
            // It works, but too mild effect.
            //u_sol(i,n_z-2) = 0.25 * ( 4.0 * u_sol(i,n_z-1) - 3.0 * u_sol(i,n_z-3) + 3.0 * alpha_h );
        }

        // Test to avoid velocity differences between u_sol.col(n_z-1) and u_sol.col(n_z-2).
        //u_sol.col(n_z-2) = u_sol.col(n_z-1);

        

        // Hydrostatic equilibrium with the ocean.
        for (int j=0; j<n_z; j++)
        {
            // Stable.
            //u_x_bc = dx * A_ice * pow( 0.25 * ( rho * g * ( H(n-1) - j * dz(n-1) ) * \
            //                          (1.0 - rho / rho_w) ), n_gln);

            //u_x_bc = dx(n-2) * A_ice * pow( 0.25 * ( rho * g * ( H(n-1) - j * dz(n-1) ) * \
                                      (1.0 - rho / rho_w) ), n_gln);
            
            // Same BC regardless of particular vertical layer depth?
            u_x_bc = dx(n-2) * A_ice * pow( 0.25 * ( rho * g * H(n-1) * \
                                        (1.0 - rho / rho_w) ), n_gln);
            u_sol(n-1,j) = u_sol(n-2,j) + u_x_bc;
        }

        
        // Ensure positive velocities.
        //u_sol = (u_sol < 0.0).select(0.0, u_sol);
        u_sol = (u_sol < 0.1).select(0.1, u_sol);

        // Symmetry at the ice divide.
        u_sol.row(0) = - u_sol.row(1);

        // Vertically averaged velocity from full Blatter-Pattyn solution.
        u_bar = u_sol.rowwise().mean();
        
    }
    

    // Allocate solutions.
    out << u_bar, u_sol;

    
    return out;
}



// RUN FLOW LINE MODEL.
// Driver method.
int main()
{
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
    int const dL_dt_num_opt = 1;                    // GL migration numerator discretization opt.
    int const dL_dt_den_opt = 1;                    // 1. GL migration denominator discretization opt.

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
    double const tf   = 57.0e4;                       // 54.0e3, 57.0, 32.0e4, MISMIP-therm: 3.0e4, 90.0e4, Ending time [yr]. EWR: 5.0e3
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
                                    u_z, visc_therm, vel_meth, t_eq, tf);
            
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
                                        visc_therm, t_eq_A_theta, visc_min, visc_max, visc_0);
            
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
        L_out = f_L(H, q, S, bed, dt, L, ds, n, rho, rho_w, M, dL_dt_num_opt, dL_dt_den_opt);
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