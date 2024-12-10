

// NIX MATERIAL MODULE.
// Viscosity calculations.


ArrayXXd f_visc(ArrayXXd theta, ArrayXXd u, ArrayXXd visc, ArrayXd H, ArrayXd tau_b, \
                ArrayXd u_bar, ArrayXd dz, ArrayXd ds, ArrayXd ds_u, ArrayXd ds_sym, \
                double L, double t, double A, DomainParams& dom, ThermodynamicsParams& thrm, \
                ViscosityParams& vis, TimeParams& tm, DynamicsParams& dyn, InitParams& init)
{


    // Prepare variables.
    ArrayXXd out(dom.n,5*dom.n_z+2), u_x(dom.n,dom.n_z), u_z(dom.n,dom.n_z), \
             strain_2d(dom.n,dom.n_z), A_theta(dom.n,dom.n_z), B_theta(dom.n,dom.n_z);
    ArrayXd u_bar_x(dom.n), strain_1d(dom.n), visc_bar(dom.n), B_theta_bar(dom.n);
    
    // Handy definitions.
    //ArrayXd dx_inv = 1.0 / ( ds * L );
    ArrayXd dx_inv = 1.0 / ( ds_u * L );
    ArrayXd dz_inv = 1.0 / dz;
    ArrayXd dx_sym_inv = 1.0 / ( ds_sym * L );


    //ArrayXd dx_u_sym_inv = 1.0 / ( ds_u_sym * L );

    
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
            // We assume a constant ice rate factor during equilibration.
            A_theta = ArrayXXd::Constant(dom.n, dom.n_z, A);
        }
        
        else if ( vis.therm == true )
        {
            // Calculate temperature-dependent rate factor if thermodynamics is switched on.
            // Arrhenius law A(T,p) equivalent to A(T').
            // Eq. 4.15 and 6.54 (Greve and Blatter, 2009).
            for (int i=0; i<dom.n; i++)
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
            }

            // Avoid unnecessary loops.
            //A_theta = (theta < thrm.theta_act).select(A_0(0) * exp(- Q_act(0) / (R * theta) ), A_theta);
            //A_theta = (theta >= thrm.theta_act).select(A_0(1) * exp(- Q_act(1) / (R * theta) ), A_theta);

        }

        // Associated rate factor. A: [Pa^-3 yr^-1]
        B_theta = pow(A_theta, (-1 / vis.n_gln) );

        // Vertically averaged B for the SSA.
        B_theta_bar = B_theta.rowwise().mean();

        // SSA solver.
        if ( dyn.vel_meth == "SSA" )
        {
            // Horizontal derivatives.
            /*for (int i=1; i<dom.n-1; i++)
            {
                // Try this for stability of the vertical advection.
                //u_bar_x(i) = ( u_bar(i+1) - u_bar(i-1) ) * dx_sym_inv(i);
                
                // Currently working.
                //u_bar_x(i) = ( u_bar(i+1) - u_bar(i) ) * dx_inv(i);
                
                // Working.
                u_bar_x(i) = ( u_bar(i) - u_bar(i-1) ) * dx_inv(i-1);
            }*/

            u_bar_x = ( shift(u_bar,-1,dom.n) - u_bar ) * dx_inv;

            // Boundary derivatives.
            u_bar_x(0)   = ( u_bar(1) - u_bar(0) ) * dx_inv(0);
            u_bar_x(dom.n-1) = ( u_bar(dom.n-1) - u_bar(dom.n-2) ) * dx_inv(dom.n-2);
            
            // Regularization term to avoid division by 0. 
            strain_1d = pow(u_bar_x,2) + vis.eps;

            // Viscosity potentially dependending on visc_term.
            visc_bar = 0.5 * B_theta_bar * pow(strain_1d, vis.n_exp);
        }

        // DIVA solver.
        else if ( dyn.vel_meth == "DIVA" )
        {     
            // Horizontal derivative du/dx as defined in Eq. 21 (Lipscomb et al., 2019).
            /*for (int i=1; i<dom.n-1; i++)
            {
                // Centred with unevenly-spaced grid.
                //u_bar_x(i) =  ( u_bar(i+1) - u_bar(i-1) ) * dx_sym_inv(i);

                // Working.
                u_bar_x(i) = ( u_bar(i) - u_bar(i-1) ) * dx_inv(i-1);

            }*/
            u_bar_x = ( shift(u_bar,-1,dom.n) - u_bar ) * dx_inv;

            u_bar_x(0)       = ( u_bar(1) - u_bar(0) ) * dx_inv(0);
            u_bar_x(dom.n-1) = ( u_bar(dom.n-1) - u_bar(dom.n-2) ) * dx_inv(dom.n-2);

            // Vertical shear stress du/dz from Eq. 36 Lipscomb et al.
            /*for (int j=0; j<dom.n_z; j++)
            {
                // Fill matrix since horizontal derivates have no vertical depedency since
                // take the derizative of the vertically-averaged velocity.
                u_x.col(j) = u_bar_x;

                for (int i=0; i<dom.n; i++)
                {
                    // Velocity vertical derivative du/dz. Eq. 36 Lipscomb et al. (2019).
                    u_z(i,j) = tau_b(i) * ( H(i) - j * dz(i) ) / ( visc(i,j) * H(i) );
                }
            }*/


            // Assign horizontally averaged velocity to all columns of u_x
            u_x = u_bar_x.replicate(1, dom.n_z);

            // Create a row vector for the vertical indices
            ArrayXd j_indices = ArrayXd::LinSpaced(dom.n_z, 0, dom.n_z - 1);

            // Broadcast H, tau_b, dz for matrix computation
            ArrayXXd H_mat(dom.n,dom.n_z), dz_mat(dom.n,dom.n_z), \
                    j_mat(dom.n_z,dom.n), j_trans(dom.n_z,dom.n), tau_mat(dom.n,dom.n_z);


            H_mat   = H.replicate(1, dom.n_z);        // Ice thickness replicated
            dz_mat  = dz.replicate(1, dom.n_z);      // Layer thickness replicated
            j_mat   = j_indices.replicate(1, dom.n); // Vertical indices
            j_trans = j_mat.transpose();
            tau_mat = tau_b.replicate(1, dom.n_z);

            // Compute u_z using the given formula (element-wise operations)
            /*cout << "\n Col j_mat = " << j_mat.cols();
            cout << "\n Row j_mat = " << j_mat.rows();

            cout << "\n j_mat = " << j_trans;

            cout << "\n Col dz_mat = " << dz_mat.cols();
            cout << "\n Row dz_mat = " << dz_mat.rows();*/

            //u_z = dz_mat;
            u_z = tau_mat * ( H_mat - j_trans * dz_mat ) / (visc * H_mat);




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
            /*for (int j=1; j<dom.n_z-1; j++)
            {
                // Centred (evenly-spaced grid in the z axis).
                u_z.col(j) = 0.5 * ( u.col(j+1) - u.col(j-1) ) * dz_inv(j);
                
                // Forwards.
                //u_z.col(j) = ( u.col(j+1) - u.col(j) ) * dz_inv(j);

                // Backwards. It works.
                //u_z.col(j) = ( u.col(j) - u.col(j-1) ) * dz_inv(j);
            }*/

            ArrayXXd dz_inv_mat(dom.n,dom.n_z), dx_inv_mat(dom.n,dom.n_z);

            dz_inv_mat = dz_inv.replicate(1, dom.n_z);
            dx_inv_mat = dx_sym_inv.replicate(1, dom.n_z);


            // Centred (evenly-spaced grid in the z axis).
            // There is no uneven grid in the vertical dimension for now.
            u_z = 0.5 * ( shift_2D(u,0,-1) - shift_2D(u,0,1) ) * dz_inv_mat;


            /*for (int i=1; i<dom.n-1; i++)
            {
                // Centred with unevenly-spaced grid.
                u_x.row(i) = ( u.row(i+1) - u.row(i-1) ) * dx_sym_inv(i);
                
                // Forwards.
                //u_x.row(i) = ( u.row(i+1) - u.row(i) ) * dx_inv(i);

                // Try backwards instead. It works.
                //u_x.row(i) = ( u.row(i) - u.row(i-1) ) * dx_inv(i-1);
            }*/


            // No need for factor 0.5 as it is contained in dx_sym_inv spacing.
            u_x = ( shift_2D(u,-1,0) - shift_2D(u,1,0) ) * dx_inv_mat;

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
            //strain_2d = pow(u_x,2) + 0.25 * pow(u_z,2) + eps;

            // Viscosity option dependending on visc_term.
            visc = 0.5 * B_theta * pow(strain_2d, vis.n_exp);

            // Viscosity in divide????????????
            visc.row(0) = visc.row(2);

            // Vertically-averaged viscosity.
            visc_bar = visc.rowwise().mean();

        }

    }

    // Allocate output variables.
    out << visc, visc_bar, u_bar_x, u_x, u_z, strain_2d, A_theta;

    return out;
}	
