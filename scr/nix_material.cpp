

// NIX MATERIAL MODULE.
// Viscosity calculations.


ArrayXXd f_visc(ArrayXXd theta, ArrayXXd u, ArrayXXd visc, ArrayXd H, ArrayXd tau_b, \
                ArrayXd u_bar, ArrayXd dz, ArrayXd ds, ArrayXd ds_inv, ArrayXd ds_sym, \
                double L, double t, double A, DomainParams& dom, ThermodynamicsParams& thrm, \
                ViscosityParams& vis, TimeParams& tm, DynamicsParams& dyn, InitParams& init)
{


    // Prepare variables.
    ArrayXXd out(dom.n,5*dom.n_z+2), u_x(dom.n,dom.n_z), u_z(dom.n,dom.n_z), \
             strain_2d(dom.n,dom.n_z), A_theta(dom.n,dom.n_z), B_theta(dom.n,dom.n_z);
    ArrayXd u_bar_x(dom.n), strain_1d(dom.n), visc_bar(dom.n), B_theta_bar(dom.n);
    
    // Handy definitions.
    ArrayXd dx_inv = 1.0 / ( ds * L );
    ArrayXd dz_inv = 1.0 / dz;
    ArrayXd dx_sym_inv = 1.0 / ( ds_sym * L );

    
    // Equilibration and constant viscosity experiments.
    if ( t < tm.t_eq )
    {
        // Full viscosity.
        visc = ArrayXXd::Constant(dom.n, dom.n_z, init.visc_0); // 1.0e15, 0.5e17
        
        // Vertically-averaged viscosity.
        visc_bar = ArrayXd::Constant(dom.n, init.visc_0);

        // We assume a constant ice rate factor during equilibration.
        A_theta = ArrayXXd::Constant(dom.n, dom.n_z, A);
    }
    
    // After equilibration.
    else 
    {
        // Potential viscosity dependency on temperature.
        // We calculate B from A(T,p).
        if ( vis.therm == false )
        {
            // We assume a constant ice rate factor if thermodynamics is off.
            A_theta = ArrayXXd::Constant(dom.n, dom.n_z, A);
        }
        
        else if ( vis.therm == true )
        {
            // Calculate temperature-dependent rate factor if thermodynamics is switched on.
            // Arrhenius law A(T,p) equivalent to A(T').
            // Eq. 4.15 and 6.54 (Greve and Blatter, 2009).
            /*for (int i=0; i<dom.n; i++)
            {
                for (int j=0; j<dom.n_z; j++)
                {
                    // Rate factor. We consider two temperature regimes (Greve and Blatter, 2009).
                    if ( theta(i,j) < thrm.theta_act )
                    {
                        A_theta(i,j) = vis.A_0_1 * exp(- vis.Q_act_1 / (vis.R * theta(i,j)) );
                    }
                    else
                    {
                        A_theta(i,j) = vis.A_0_2 * exp(- vis.Q_act_2 / (vis.R * theta(i,j)) );
                    }
                }
            }*/

            // Avoid unnecessary loops.
            A_theta = (theta < thrm.theta_act).select(vis.A_0_1 * exp(- vis.Q_act_1 / (vis.R * theta) ), A_theta);
            A_theta = (theta >= thrm.theta_act).select(vis.A_0_2 * exp(- vis.Q_act_2 / (vis.R * theta) ), A_theta);

        }

        // Associated rate factor. A: [Pa^-3 yr^-1]
        B_theta = pow(A_theta, (-1 / vis.n_gln) );

        // Vertically averaged B for the SSA.
        B_theta_bar = B_theta.rowwise().mean();

        // We use the median rather than the mean?
        //B_theta_bar = f_median(B_theta, n, n_z);

        // SSA solver.
        if ( dyn.vel_meth == "SSA" )
        {
            // Horizontal derivatives.
            for (int i=1; i<dom.n-1; i++)
            {
                // Try this for stability of the vertical advection.
                u_bar_x(i) = ( u_bar(i+1) - u_bar(i-1) ) * dx_sym_inv(i);
                
                // Currently working.
                //u_bar_x(i) = ( u_bar(i+1) - u_bar(i) ) * dx_inv(i);
                
                // Not working.
                //u_bar_x(i) =  ( u_bar(i+1) - u_bar(i-1) ) * dx_sym_inv(i);
            }

            // Boundary derivatives.
            u_bar_x(0)       = ( u_bar(1) - u_bar(0) ) * dx_inv(0);
            u_bar_x(dom.n-1) = ( u_bar(dom.n-1) - u_bar(dom.n-2) ) * dx_inv(dom.n-2);
            
            // Regularization term to avoid division by 0. 
            strain_1d = pow(u_bar_x,2) + vis.eps;

            // Viscosity potentially dependendent on visc_term. 
            visc_bar = 0.5 * B_theta_bar * pow(strain_1d, vis.n_exp);
        }

        // DIVA solver.
        else if ( dyn.vel_meth == "DIVA" )
        {     
            // Horizontal derivative du/dx as defined in Eq. 21 (Lipscomb et al., 2019).
            for (int i=1; i<dom.n-1; i++)
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
            u_bar_x(0)       = ( u_bar(1) - u_bar(0) ) * dx_inv(0);
            u_bar_x(dom.n-1) = ( u_bar(dom.n-1) - u_bar(dom.n-2) ) * dx_inv(dom.n-2);

            // Vertical shear stress du/dz from Eq. 36 Lipscomb et al.
            for (int j=0; j<dom.n_z; j++)
            {
                // Fill matrix since horizontal derivates have no vertical depedency since
                // take the derizative of the vertically-averaged velocity.
                u_x.col(j) = u_bar_x;

                for (int i=0; i<dom.n; i++)
                {
                    // Velocity vertical derivative du/dz. Eq. 36 Lipscomb et al. (2019).
                    u_z(i,j) = tau_b(i) * ( H(i) - j * dz(i) ) / ( visc(i,j) * H(i) );
                }
            }

            // Strain rate and regularization term to avoid division by 0. 
            strain_2d = pow(u_x,2) + 0.25 * pow(u_z,2) + vis.eps;

            // Viscosity option dependending on visc_term via B_theta.
            visc = 0.5 * B_theta * pow(strain_2d, vis.n_exp);

            // Vertically averaged viscosity.
            visc_bar = visc.rowwise().mean();

        }

        // Blatter-Pattyn.
        else if ( dyn.vel_meth == "Blatter-Pattyn" )
        {     
            // Spatial derivatives du/dx, du/dz.
            for (int j=1; j<dom.n_z-1; j++)
            {
                // Centred (evenly-spaced grid in the z axis).
                u_z.col(j) = 0.5 * ( u.col(j+1) - u.col(j-1) ) * dz_inv(j);
                
                // Forwards.
                //u_z.col(j) = ( u.col(j+1) - u.col(j) ) * dz_inv(j);

                // Forwards. It works.
                //u_z.col(j) = u.col(j) - u.col(j-1);
            }

            for (int i=1; i<dom.n-1; i++)
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

            u_x.row(dom.n-1)   = ( u.row(dom.n-1) - u.row(dom.n-2) ) * dx_inv(dom.n-2);
            u_z.col(dom.n_z-1) = ( u.col(dom.n_z-1) - u.col(dom.n_z-2) ) * dz_inv(dom.n_z-1);
            */

            // Three-point derivative.
            // MIT.
            // du/dz = 0.5 * ( 3.0 * u(dom.n_z-1) - 4.0 * u(dom.n_z-2) + 1.0 * u(dom.n_z-3) ) 
            
            u_x.row(0) = 0.5 * ( u.row(0) - 4.0 * u.row(1) + 3.0 * u.row(2) ) * dx_inv(0);
            u_z.col(0) = 0.5 * ( u.col(0) - 4.0 * u.col(1) + 3.0 * u.col(2) ) * dz_inv(0);

            //u_x.row(dom.n-1)   = 0.5 * ( u.row(dom.n-1) - 4.0 * u.row(dom.n-2) + 3.0 * u.row(dom.n-3) ) * dx_inv(dom.n-2);
            //u_z.col(dom.n_z-1) = 0.5 * ( u.col(dom.n_z-1) - 4.0 * u.col(dom.n_z-2) + 3.0 * u.col(dom.n_z-3) ) * dz_inv(dom.n-1);
            u_x.row(dom.n-1)   = ( u.row(dom.n-1) - u.row(dom.n-2) ) * dx_inv(dom.n-2);
            u_z.col(dom.n_z-1) = ( u.col(dom.n_z-1) - u.col(dom.n_z-2) ) * dz_inv(dom.n_z-1);
            
            // Try Vieli and Payne assymetric differences.
            // Vieli 1 (Eq. B12).
            // du/dz = ( 4.0 * u(dom.n_z-1) - 3.0 * u(dom.n_z-2) - 1.0 * u(dom.n_z-3) ) / 3.0
            /*
            u_x.row(0) = ( 4.0 * u.row(0) - 3.0 * u.row(1) - u.row(2) ) / 3.0;
            u_z.col(0) = ( 4.0 * u.col(0) - 3.0 * u.col(1) - u.col(2) ) / 3.0;

            u_x.row(dom.n-1)   = ( 4.0 * u.row(dom.n-1) - 3.0 * u.row(dom.n-2) - u.row(dom.n-3) ) / 3.0;
            u_z.col(dom.n_z-1) = ( 4.0 * u.col(dom.n_z-1) - 3.0 * u.col(dom.n_z-2) - u.col(dom.n_z-3) ) / 3.0;
            */
            

            // Strain rate and regularization term to avoid division by 0. 
            strain_2d = pow(u_x,2) + 0.25 * pow(u_z,2) + vis.eps;

            // Viscosity option dependending on visc_term.
            visc = 0.5 * B_theta * pow(strain_2d, vis.n_exp);

            // Vertically-averaged viscosity.
            visc_bar = visc.rowwise().mean();

        }

        else
        {
            cout << " \n Velocity solver not defined. Please, select it: SSA, DIVA... ";
            abort();
        }
    }

    // Allocate output variables.
    out << visc, visc_bar, u_bar_x, u_x, u_z, strain_2d, A_theta;

    return out;
}	
