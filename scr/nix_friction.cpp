
// NIX FRICTION MODULE.


ArrayXd f_C_bed(ArrayXd C_ref, ArrayXXd theta, ArrayXd H, double t, \
                DomainParams dom, ConstantsParams cnst, TimeParams tm, \
                FrictionParams fric)
{
    
    ArrayXd C_bed(dom.n), theta_norm(dom.n);

    // Basal friction coupled with thermal state of the base.
    if (fric.therm == "two-valued" && t > tm.t_eq)
    {
        
        // Binary friction.
        for (int i=1; i<dom.n; i++)
        {
            // Reference value for frozen bed.
            if ( theta(i,0) < fric.theta_frz )
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
    else if (fric.therm == "N_eff" && t > tm.t_eq)
    {
        C_bed = 0.001 * cnst.rho * cnst.g * H;

        // g in m/s² --> m/yr^2 SOMETHING MIGHT BE WRONG HERE!
        //C_bed = 0.001 * rho * g * H * pow(sec_year, 2);
    }

    // Friction coefficient given by reference value.
    else if ( fric.therm == "none" )
    {
        C_bed = C_ref;
    }

    return C_bed;
}


ArrayXXd f_u(ArrayXXd u, ArrayXd u_bar, ArrayXd beta, ArrayXd C_bed, ArrayXXd visc, \
             ArrayXd H, ArrayXd dz, double t, DomainParams& dom, \
             DynamicsParams& dyn, FrictionParams& fric, ConstantsParams& cnst)
{

    // Prepare varibales.
    ArrayXXd out(dom.n,4), F_1(dom.n,dom.n_z), F_all(dom.n,dom.n_z+1); 
    ArrayXd beta_eff(dom.n), ub(dom.n), F_2(dom.n), tau_b(dom.n), Q_fric(dom.n);

    //ArrayXXd out(n,n_z+6),

    // For constant visc or SSA, ub = u_bar.
    if ( dyn.vel_meth == "const" || dyn.vel_meth == "SSA" )
    {
        // Vel definition. New beta from ub.
        ub = u_bar;

        // beta = beta_eff, ub = u_bar for SSA.
        beta_eff    = C_bed * pow(ub, fric.m - 1.0);
        beta_eff(0) = beta_eff(1);
    }

    // For DIVA solver, Eq. 32 Lipscomb et al.
    else if ( dyn.vel_meth == "DIVA" )
    {
        // Useful integral. n_z-1 as we count from 0.
        F_all = F_int_all(visc, H, dz, dom.n_z, dom.n);
        
        F_2 = F_all.block(0, 0, dom.n, 1);
        F_1 = F_all.block(0, 1, dom.n, dom.n_z);
        
        
        // REVISE HOW BETA IS CALCULATED
        // Obtain ub from new u_bar and previous beta (Eq. 32 Lipscomb et al.).
        ub = u_bar / ( 1.0 + beta * F_2 );

        // Impose BC here?
        ub(0) = - ub(1);

        // Beta definition. New beta from ub.
        beta = C_bed * pow(ub, fric.m - 1.0);

        // DIVA solver requires an effective beta.
        beta_eff = beta / ( 1.0 + beta * F_2 );

        // Impose ice divide boundary condition.
        beta(0)     = beta(1);
        beta_eff(0) = beta_eff(1);

        
        // Now we can compute u(x,z) from beta and visc.
        for (int j=0; j<dom.n_z; j++)
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
    else if ( dyn.vel_meth == "Blatter-Pattyn" )
    {
        // Basal velocity from velocity matrix.
        ub = u.col(0);
        
        // beta = beta_eff, ub = u_bar for SSA.
        beta_eff    = C_bed * pow(ub, fric.m - 1.0);
    }
    

    // Calculate basal shear stress with ub (not updated in Picard iteration). 
    tau_b    = beta_eff * ub;
    tau_b(0) = tau_b(1);    // Symmetry ice divide. Avoid negative tau as u_bar(0) < 0. 

    // Frictional heat. [Pa · m / yr] --> [W / m^2].
    Q_fric    = tau_b * ub / cnst.sec_year;
    Q_fric(0) = Q_fric(1);


    // Allocate output variables.
    out << beta_eff, tau_b, Q_fric, ub;

    
    return out;
}


