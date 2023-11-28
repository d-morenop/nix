

// NIX ICE FLUX MODULE.
// Computes grounding line positions, fluxes and calving at GL.

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


ArrayXd f_q(ArrayXd u_bar, ArrayXd H, double H_f, double t, double t_eq, \
               double rho, double rho_w, double m_dot, double M, string calving_meth, int n)
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
    if ( calving_meth == "none" )
    {
        //q(n-1) = u_bar(n-1) * H(n-1);

        // We impose sythetic reduction potentially cause by ocean cooling/warming.
        q(n-1) = u_bar(n-1) * H(n-1);
    } 
    
    // Additional calving term (Christian et al., 2022).
    // ICE FLUX DISCRETIZATION SEEMS TO BE FUNDAMENTAL TO OBTAIN
    // THE SAME ICE SHEET ADVANCED AS CHRISTIAN. STAGGERED GRID.
    else if ( calving_meth == "stochastic" )
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
    else if ( calving_meth == "deterministic" )
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
          double rho_w, double M)
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

