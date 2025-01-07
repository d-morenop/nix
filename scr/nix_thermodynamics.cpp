

// NIX THERMODYNAMICS MODULE.

ArrayXXd f_theta(ArrayXXd theta, ArrayXd u_bar, ArrayXd H, ArrayXd tau_b, ArrayXd Q_fric, ArrayXd bed, \
                 ArrayXd sigma, ArrayXd dz, double dt, ArrayXd ds, double L, double T_air, \
                 double dL_dt, double t, ArrayXXd w, ArrayXXd strain_2d, ArrayXXd u, \
                 DomainParams& dom, ThermodynamicsParams& thrm, DynamicsParams& dyn, \
                 BoundaryConditionsParams& bc, ConstantsParams& cnst, CalvingParams& calv)
{
    
    ArrayXXd out(dom.n,dom.n_z+1), theta_now(dom.n,dom.n_z);
    ArrayXd b_dot(dom.n), h(dom.n);
    ArrayXd b_melt = ArrayXd::Zero(dom.n);
 
    // Evenly-spaced vertical grid, though x-dependency via ice thickness.
    ArrayXd dz_inv   = 1.0 / dz;
    ArrayXd dz_2_inv = pow(dz_inv, 2);
    ArrayXd dx_inv   = 1.0 / (ds * L);

    // Frictional heat units. // [Pa m yr^-1] = [J yr^-1 m^-2] = [W m^-2] => [K / m]
    ArrayXd Q_f_k = Q_fric / thrm.k;     

    // Lapse rate test.// Adiabatic lapse rate 9.8e-3 [ºC / m]. Moist adiabatic: 5.0e-3 [ºC / m]
    // Greenland mean: 7.1e-3 [ºC/m]: Steffen and Box (2001).
    double gamma = 5.0e-3; // 9.8e-3
    h = H + bed;
    
    // Zero frictional heat test.
    //Q_f_k = ArrayXd::Zero(n);

    // Temperature integration.
    /*for (int i=1; i<dom.n; i++)
    {
        for (int j=1; j<dom.n_z-1; j++)
        {
            // Vertical advection.
            // Since w < 0 we need an opposite discretization scheme in theta.
            // Vertical velocity con z-dependency.
            // Unevenly-spaced vertical grid.
            // w(i,j) now also depends on x!
            // Try symmetric in vertical advection (factor 0.5 since it is evenly-spaced in z direction)
            theta_now(i,j) = theta(i,j) + dt * ( thrm.kappa * dz_2_inv(i) * \
                            ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,j) - theta(i-1,j) ) * dx_inv(i-1) + \
                            ( theta(i,j) - theta(i,j+1) ) * w(i,j) * dz_inv(i) ); //( theta(i,j) - theta(i,j+1) ) * w(i,j) * dz_inv(i) );

            // BP.
            //theta_now(i,j) = theta(i,j) + dt * ( thrm.kappa * dz_2_inv(i) * \
                            ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                            ( sigma(i) * dL_dt - u(i,j) ) * \
                            ( theta(i,j) - theta(i-1,j) ) * dx_inv(i-1) + \
                            ( theta(i,j) - theta(i,j+1) ) * w(i,j) * dz_inv(i) ); 

        }

        // Boundary conditions. Geothermal heat flow at the base.
        // We add friciton heat contribution Q_f_k. w(z=0) = 0 
        // thrm.G_k was alread transformed in readparams to [K/m].
        theta_now(i,0) = theta_now(i,1) + dz(i) * ( thrm.G_k + Q_f_k(i) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,0) - theta(i-1,0) ) * dx_inv(i-1) + \
                            ( theta(i,0) - theta(i,1) ) * w(i,0) * dz_inv(i);

        // BP.
        //theta_now(i,0) = theta_now(i,1) + dz(i) * ( thrm.G_k + Q_f_k(i) ) + \
                            ( sigma(i) * dL_dt - u(i,0) ) * \
                            ( theta(i,0) - theta(i-1,0) ) * dx_inv(i-1) + \
                            ( theta(i,0) - theta(i,1) ) * w(i,0) * dz_inv(i);

        // Surface. Not from param file since BC may change in time!
        //theta_now(i,dom.n_z-1) = T_air;

        // Test vertical dependency on adiabatic lapse rate.
        theta_now(i,dom.n_z-1) = T_air - gamma * max(h(i), 0.0);

        
    }*/

    
    // Create matrix in SSA and DIVA case.
    if ( dyn.vel_meth == "SSA" || dyn.vel_meth == "DIVA" )
    {
        u = u_bar.replicate(1, dom.n_z);
    }
    
    // Prepare 2D arrays for element-wise multiplication.
    ArrayXXd dz_2_inv_mat = dz_2_inv.replicate(1, dom.n_z);
    ArrayXXd dx_inv_mat   = dx_inv.replicate(1, dom.n_z);
    ArrayXXd dz_inv_mat   = dz_inv.replicate(1, dom.n_z);
    ArrayXXd sigma_mat    = sigma.replicate(1, dom.n_z);


    // Vectorial form working perfectly for SSA and DIVA. BP has some problems.
    // It seems to be caused by the vertical velocities calculation.
    // Noise in temperatures yields noisy viscosity and it does not converge.
    //u = 0.5 * ( u + shift_2D(u,-1,0) );
    //w = ArrayXXd::Zero(dom.n, dom.n_z);

    // Vectorial solution.
    theta_now = theta + dt * ( thrm.kappa * dz_2_inv_mat * \
                        ( shift_2D(theta,0,-1) - 2.0 * theta + shift_2D(theta,0,1) ) + \
                        ( sigma_mat * dL_dt - u ) * \
                        ( theta - shift_2D(theta,1,0) ) * dx_inv_mat + \
                        ( theta - shift_2D(theta,0,-1) ) * w * dz_inv_mat ); 


    // Basal boundary condition.
    theta_now.col(0) = theta_now.col(1) + dz * ( thrm.G_k + Q_f_k ) + \
                            ( sigma * dL_dt - u.col(0) ) * \
                            ( theta.col(0) - shift(theta.col(0),1,dom.n) ) * dx_inv + \
                            ( theta.col(0) - theta.col(1) ) * w.col(0) * dz_inv;
    
    // Top boundary consition. Ensure positive values of surface elevation.
    h = (h < 0.0).select(0.0, h);
    theta_now.col(dom.n_z-1) = T_air - gamma * h;
    //theta_now.col(dom.n_z-1) = T_air;


    // Due to symmetry theta_now(0,j) = theta_now(2,j). Ice divide.
    theta_now.row(0) = theta_now.row(2);

    // Boundary condition at x = L??
    theta_now.row(dom.n-1) = theta_now.row(dom.n-2);

    // For the DIVA solver, consider the strain heat contribution.
    if ( dyn.vel_meth == "DIVA" || dyn.vel_meth == "Blatter-Pattyn" )
    {
        theta_now = theta_now + ( thrm.kappa / thrm.k ) * strain_2d * dt;
    }


    // Apply relaxation at all times. It will crash without this!
    //double rel = 0.0; // 0.8, 0.7. 0.5 also works
    //theta_now = ( 1.0 - rel ) * theta_now + rel * theta;
    
    // Compute total basal melting. [m/s] --> [m/yr].
    b_dot  = cnst.sec_year * ( thrm.k / (cnst.rho * calv.sub_shelf_melt.L_i ) ) * ( thrm.G_k + Q_f_k );

    // Only apply for those points that reach the pressure melting point.
    b_melt = (theta_now.col(0) >= thrm.theta_max).select(b_dot, b_melt);

    // Pressure melting point as the upper bound.
    theta_now = (theta_now > thrm.theta_max).select(thrm.theta_max, theta_now);

    // Test to avoid instabilities.
    //theta_now = (theta_now < theta_now(0,dom.n_z-1)).select(theta, theta_now);

    // Output.
    out << theta_now, b_melt;

    return out;
}