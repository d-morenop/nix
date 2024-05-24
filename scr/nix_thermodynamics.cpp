

// NIX THERMODYNAMICS MODULE.

ArrayXXd f_theta(ArrayXXd theta, ArrayXd ub, ArrayXd H, ArrayXd tau_b, ArrayXd Q_fric, ArrayXd bed, \
                 ArrayXd sigma, ArrayXd dz, double dt, ArrayXd ds, double L, double T_air, \
                 double dL_dt, double t, ArrayXXd w, ArrayXXd strain_2d, \
                 DomainParams& dom, ThermodynamicsParams& thrm, DynamicsParams& dyn, \
                 BoundaryConditionsParams& bc, ConstantsParams& cnst, CalvingParams& calv)
{
    
    ArrayXXd out(dom.n,dom.n_z+1), theta_now(dom.n,dom.n_z);
    ArrayXd dx_inv(dom.n-1), dz_inv(dom.n), dz_2_inv(dom.n), Q_f_k(dom.n), b_dot(dom.n), h(dom.n);
    ArrayXd b_melt = ArrayXd::Zero(dom.n);
 
    // Evenly-spaced vertical grid, though x-dependency via ice thickness.
    dz_inv   = 1.0 / dz;
    dz_2_inv = pow(dz_inv, 2);
    dx_inv   = 1.0 / (ds * L);

    // Frictional heat units. // [Pa m yr^-1] = [J yr^-1 m^-2] = [W m^-2] => [K / m]
    Q_f_k = Q_fric / thrm.k;     

    // Lapse rate test.// Adiabatic lapse rate 9.8e-3 [ºC / m]. Moist adiabatic: 5.0e-3 [ºC / m]
    // Greenland mean: 7.1e-3 [ºC/m]: Steffen and Box (2001).
    double gamma = 9.8e-3; 
    h = H + bed;
    
    // Zero frictional heat test.
    //Q_f_k = ArrayXd::Zero(n);

    // Temperature integration.
    for (int i=1; i<dom.n; i++)
    {
        for (int j=1; j<dom.n_z-1; j++)
        {
            // Vertical advection.
            // Since w < 0 we need an opposite discretization scheme in theta.
            // Vertical velocity con z-dependency.
            // Unevenly-spaced vertical grid.
            // w(i,j) now also depends on x!
            // Try symmetric in vertical advection (factor 0.5 since it is evenly-spaced in z direction)
            // 0.5 * ( theta(i,j+1) - theta(i,j-1) ) * ( - w(i,j) )
            /*theta_now(i,j) = theta(i,j) + dt * ( thrm.kappa * dz_2_inv(i) * \
                            ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,j) - theta(i-1,j) ) * dx_inv(i-1) + \
                            ( theta(i,j) - theta(i,j-1) ) * ( - w(i,j) ) * dz_inv(i) );*/

            theta_now(i,j) = theta(i,j) + dt * ( thrm.kappa * dz_2_inv(i) * \
                            ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,j) - theta(i-1,j) ) * dx_inv(i-1) + \
                            ( theta(i,j) - theta(i,j+1) ) * w(i,j) * dz_inv(i) ); //( theta(i,j) - theta(i,j+1) ) * w(i,j) * dz_inv(i) );

        }
        
        // Boundary conditions. Geothermal heat flow at the base.
        // We add friciton heat contribution Q_f_k. w(z=0) = 0 
        // thrm.G_k was alread transformed in readparams to [K/m].
        /*theta_now(i,0) = theta_now(i,1) + dz(i) * ( thrm.G_k + Q_f_k(i) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,0) - theta(i-1,0) ) * dx_inv(i-1) + \
                            ( theta(i,1) - theta(i,0) ) * ( - w(i,0) ) * dz_inv(i);*/


        theta_now(i,0) = theta_now(i,1) + dz(i) * ( thrm.G_k + Q_f_k(i) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,0) - theta(i-1,0) ) * dx_inv(i-1) + \
                            ( theta(i,0) - theta(i,1) ) * w(i,0) * dz_inv(i);

        // Surface. Not from param file since BC may change in time!
        //theta_now(i,dom.n_z-1) = T_air;

        // Test vertical dependency on adiabatic lapse rate.
        theta_now(i,dom.n_z-1) = T_air - gamma * max(h(i), 0.0);
    }

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
    double rel = 0.8; // 0.7
    theta_now = ( 1.0 - rel ) * theta_now + rel * theta;
    
    // Compute total basal melting. [m/s] --> [m/yr].
    b_dot  = cnst.sec_year * ( thrm.k / (cnst.rho * calv.sub_shelf_melt.L_i ) ) * ( thrm.G_k + Q_f_k );

    // Only apply for those points that reach the pressure melting point.
    b_melt = (theta_now.col(0) >= thrm.theta_max).select(b_dot, b_melt);

    // Pressure melting point as the upper bound.
    theta_now = (theta_now > thrm.theta_max).select(thrm.theta_max, theta_now);

    // Test to avoid instabilities.
    //theta_now = (theta_now < theta_now(0,dom.n_z-1)).select(theta, theta_now);

    // Test for the grounding line column.
    //theta_now.row(dom.n-1) = theta_now.row(dom.n-2);

    // Output.
    out << theta_now, b_melt;

    return out;
}