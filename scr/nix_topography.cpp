

// TOPOGRAPHY MODULE.
// Bed geomtry, surface elevation and surface mass balance.
// SMB spatial distribution depends on the position of the ice.
// It does not follow the ice sheet, but rather there's a certain 
// value for each position x.


ArrayXd f_bed(double L, ArrayXd sigma, ArrayXd ds, double t, DomainParams dom)
{
    
    // Prepare variables.
    ArrayXd bed(dom.n);
    ArrayXd x = sigma * L; 

    // Number of points at the edges of the array that are not smoothed out.
    //int p = 3;

    // MISMIP experiments bedrock.
    // Same bedrock as Schoof (2007).
    // Inverse sign to get a decreasing bedrock elevation.
    //if ( dom.bed_exp == "mismip_1" || dom.bed_exp == "mismip_1_therm" )

    if ( dom.bed_exp == "mismip_1" )
    {
        x = x / 750.0e3; 
        bed = 720.0 - 778.5 * x;
    }
    else if ( dom.bed_exp == "mismip_3" )
    {
        x = x / 750.0e3; 
        // Schoof: 2184.8. Daniel: 2148.8.
        bed = 729.0 - 2148.8 * pow(x, 2) \
                    + 1031.72 * pow(x, 4) + \
                    - 151.72 * pow(x, 6);
    }
    else if ( dom.bed_exp == "ews" )
    {
        // Variables.
        int c_x1 = 0;
        int c_x2 = 0;
        double y_1, y_2;
        double m_bed = dom.ews.y_p / ( dom.ews.x_2 - dom.ews.x_1 );

        // Piecewise function.
        for (int i=0; i<dom.n; i++)
        {
            // First part.
		    if ( x(i) <= dom.ews.x_1 )
            {
                bed(i) = dom.ews.y_0 - 1.5e-3 * x(i);
            }
		
		    // Second.
            else if ( x(i) >= dom.ews.x_1 && x(i) <= dom.ews.x_2 )
            {
                // Save index of last point in the previous interval.
                if ( c_x1 == 0 )
                {
                    y_1  = bed(i-1);
                    c_x1 += 1;
                }
                    
                // Bedrock function.
                bed(i) = y_1 + m_bed * ( x(i) - dom.ews.x_1 );
            }
                
            // Third.
            else if ( x(i) > dom.ews.x_2 )
            {
                // Save index of last point in the previous interval.
                if (c_x2 == 0)
                {
                    y_2  = bed(i-1);
                    c_x2 += 1;	
                }
                    
                // Bedrock function.
                bed(i) = y_2 - 5.0e-3 * ( x(i) - dom.ews.x_2 );
            } 
        }
    }
    else
    {
        cout << "\n Bed geometry not recognised. Please, select: mismip_1... ";
        abort();
    }

    // Potential smooth bed.
    // Note that the smoothing is computed at every time step and thus changes over time
    // as the edges that the ice sheet "sees" move.
    if ( t >= dom.ews.t0_smth )
    {
        if ( dom.ews.smooth == "gauss" )
        {
            // Gaussian smooth.
            bed = gaussian_filter(bed, sigma, ds, dom.ews.sigma_gauss, dom.ews.p, dom.n);
        }
        
        else if ( dom.ews.smooth == "running_mean" )
        {
            // Running mean.
            bed = running_mean(bed, dom.ews.p, dom.n);
        }
    }
    

    return bed;
}



ArrayXd f_smb(ArrayXd sigma, double L, double t, double smb_stoch, \
              BoundaryConditionsParams& bc, DomainParams& dom, \
              TimeParams& tm)
{
   
    // Variables.
    ArrayXd x(dom.n), S(dom.n); 
    double stoch_pattern, smb_determ;

    // Horizontal dimension to define SBM.
    x = L * sigma;

    // Start with constant accumulation value for smoothness.
    if ( t < tm.t_eq )
    {
        // Ensure healthy equilibiration before desired surface accumulation: 0.4 m/yr.
        S = ArrayXd::Constant(dom.n, 0.4);
    }
    else
    {
        // Deterministic SMB.
        if ( bc.smb.stoch == false )
        {
            S = ArrayXd::Constant(dom.n, bc.smb.S_0);
        }

        // Stochastic SMB.
        else
        {
            // Error function from Christian et al. (2022)
            for (int i=0; i<dom.n; i++)
            {
                // No stochastic variability.
                //S(i) = S_0 + 0.5 * dlta_smb * ( 1.0 + erf((x(i) - x_mid) / x_sca) );

                // SMB stochastic variability sigmoid.
                stoch_pattern = bc.smb.var_mult + ( 1.0 - bc.smb.var_mult ) * 0.5 * \
                                ( 1.0 + erf((x(i) - bc.smb.x_varmid) / bc.smb.x_varsca) );

                // Deterministic SMB sigmoid.
                smb_determ = bc.smb.S_0 + 0.5 * bc.smb.dlta_smb * \
                                        ( 1.0 + erf((x(i) - bc.smb.x_mid) / bc.smb.x_sca) );
                
                // Total SMB: stochastic sigmoid + deterministic sigmoid.
                S(i) = smb_stoch * stoch_pattern + smb_determ;
            }
        }
    }
 
    return S;
}



// First order scheme. New variable sigma.
ArrayXd f_H(ArrayXd u_bar, ArrayXd H, ArrayXd S, ArrayXd sigma, \
            double dt, ArrayXd ds, ArrayXd ds_inv, ArrayXd ds_sym, ArrayXd ds_u_inv, \
            double L, double D, double dL_dt, ArrayXd bed, ArrayXd q, \
            double M, double t, DomainParams& dom, TimeParams& tm, AdvectionParams& adv)
{
    // Local variables.
    ArrayXd H_now(dom.n), dx_inv(dom.n-1), dx_sym_inv(dom.n-1);

    double L_inv  = 1.0 / L;
    
    dx_inv     = L_inv * ds_inv;
    dx_sym_inv = L_inv * ds_sym;

    ArrayXd dx_u_inv = L_inv * ds_u_inv;

    // Solution to the modified advection equation considering a streched coordinate
    // system sigma. Two schemes are available, explicit and implicit, noted as
    // meth = 0, 1 respectively. 
    //  Right now, explicit seems more stable since the 
    // implicit crasher earlier. 
    
    // Explicit scheme. Centred dH in the sigma_L term.
    if ( adv.meth == "explicit" )
    {
        /*for (int i=1; i<dom.n-1; i++)
        {
            // Centred in sigma, upwind in flux. Unevenly-spaced horizontal grid.
            //H_now(i) = H(i) + dt * ( sigma(i) * dL_dt * ( H(i+1) - H(i-1) ) * dx_sym_inv(i) + \
                                            - dx_inv(i) * ( q(i) - q(i-1) ) + S(i) );

            // Test uneven velocty grid.
            H_now(i) = H(i) + dt * ( sigma(i) * dL_dt * ( H(i+1) - H(i-1) ) * dx_sym_inv(i) + \
                                            - dx_u_inv(i) * ( q(i) - q(i-1) ) + S(i) );

        }*/

        // Vectorized version.
        H_now = H + dt * ( sigma * dL_dt * ( shift(H,-1,dom.n) - shift(H,1,dom.n) ) * dx_sym_inv + \
                                            - dx_u_inv * ( q - shift(q,1,dom.n) ) + S );
        
        // Symmetry at the ice divide (i = 1).
        H_now(0) = H_now(2);
        
        // Lateral boundary: sigma(n-1) = 1.
        // Sub-shelf melt directly on the flux.
        // Note that ds has only (n-1) points.
        //H_now(dom.n-1) = H(dom.n-1) + dt * ( ds_inv(dom.n-2) * L_inv * \
        //                                ( dL_dt * ( H(dom.n-1) - H(dom.n-2) ) + \
        //                                    - ( q(dom.n-1) - q(dom.n-2) ) ) + S(dom.n-1) );

        H_now(dom.n-1) = H(dom.n-1) + dt * ( dx_inv(dom.n-2) * \
                                        ( dL_dt * ( H(dom.n-1) - H(dom.n-2) ) + \
                                            - ( q(dom.n-1) - q(dom.n-2) ) ) + S(dom.n-1) );

        
        // Make sure that grounding line thickness is above minimum?
        //H_now(n-1) = max( (rho_w/rho)*D, H_now(n-1));

        // Try relaxation during equilibration to avoid early crashing?
        if ( t < tm.t_eq )
        {
            double rel = 0.7;
            H_now = H * rel  + ( 1.0 - rel ) * H_now;
        }
    }
    
    // Implicit scheme.
    /*
    else if ( adv.meth == "implicit" )
    {
        // Local variables.
        ArrayXd A(dom.n), B(dom.n), C(dom.n), F(dom.n);
        ArrayXd gamma = dt / ( 2.0 * ds * L );

        // Implicit scheme. REVISE TRIDIAGONAL MATRIX.
        for (int i=1; i<dom.n-1; i++)
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


