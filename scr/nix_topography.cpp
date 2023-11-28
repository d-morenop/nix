

// TOPOGRAPHY MODULE.
// Bed geomtry, surface elevation and surface mass balance.
// SMB spatial distribution depends on the position of the ice.
// It does not follow the ice sheet, but rather there's a certain 
// value for each position x.

//ArrayXd f_bed(double L, int n, string exp, \
//              double y_0, double y_p, double x_1, double x_2, \
//              int smooth_bed, double sigma_gauss, ArrayXd sigma)

ArrayXd f_bed(double L, ArrayXd sigma, DomainParams& params)
{
    
    /*int n = params.n;
    string exp = params.experiment;
    double y_0 = params.bedrock_ews.y_0;
    double y_p = params.bedrock_ews.y_p;
    double x_1 = params.bedrock_ews.x_1;
    double x_2 = params.bedrock_ews.x_2;
    bool smooth_bed = params.bedrock_ews.smooth_bed;
    double sigma_gauss = params.bedrock_ews.sigma_gauss;*/

    // Using structured bindings for DomainParams to extract the desried values.
    auto [exp, n, n_z, grid, grid_exp, bedrock_ews] = params;
    auto [smooth_bed, sigma_gauss, t0_gauss, x_1, x_2, y_p, y_0] = params.bedrock_ews;

    // Prepare variables.
    ArrayXd bed(n);
    //ArrayXd x = ArrayXd::LinSpaced(n, 0.0, L); 
    ArrayXd x = sigma * L; 

    // Number of points at the edges of the array that are not smoothed out.
    int p = 3;

    // MISMIP experiments bedrock.
    // Same bedrock as Schoof (2007).
    // Inverse sign to get a decreasing bedrock elevation.
    if ( exp == "mismip_1" || exp == "mismip_1_therm" )
    {
        x = x / 750.0e3; 
        bed = 720.0 - 778.5 * x;
    }
    else if ( exp == "mismip_3" || exp == "mismip_3_therm" )
    {
        x = x / 750.0e3; 
        // Schoof: 2184.8. Daniel: 2148.8.
        bed = 729.0 - 2148.8 * pow(x, 2) \
                    + 1031.72 * pow(x, 4) + \
                    - 151.72 * pow(x, 6);
    }
    else if ( exp == "ews" )
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
    if ( smooth_bed == true )
    {
        // Gaussian smooth. Quite sensitive to p value (p=5 for n=250).
        //bed = gaussian_filter(bed, sigma_gauss, p, n);

        // Running mean.
        bed = running_mean(bed, 3, n);
    }

    return bed;
}



/*ArrayXd f_smb(ArrayXd sigma, double L, double S_0, \
              double x_mid, double x_sca, double x_varmid, \
              double x_varsca, double dlta_smb, double var_mult, \
              double smb_stoch, double t, double t_eq, int n, bool stoch)*/

ArrayXd f_smb(ArrayXd sigma, double L, double t, double smb_stoch, \
              BoundaryConditionsParams& bc, DomainParams& domain, \
              TimeParams& time)
{
    
    int n       = domain.n;
    double t_eq = time.t_eq;
    auto [stoch, t0_stoch, S_0, dlta_smb, x_acc, x_mid, x_sca, x_varmid, x_varsca, var_mult] = bc.smb;
    
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



// First order scheme. New variable sigma.
ArrayXd f_H(ArrayXd u_bar, ArrayXd H, ArrayXd S, ArrayXd sigma, \
            double dt, ArrayXd ds, ArrayXd ds_inv, ArrayXd ds_sym, int n, \
            double L, double D, double rho, double rho_w, \
            double dL_dt, ArrayXd bed, ArrayXd q, double M, string H_meth, \
            double t, double t_eq)
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
    if ( H_meth == "explicit" )
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
    else if ( H_meth == "implicit" )
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


