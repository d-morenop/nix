

// NIX THERMODYNAMICS MODULE.

ArrayXXd f_theta(ArrayXXd theta, ArrayXd ub, ArrayXd H, ArrayXd tau_b, ArrayXd Q_fric, \
                 ArrayXd sigma, ArrayXd dz, double dt, ArrayXd ds, double L, \
                 double dL_dt, double t, ArrayXd w, ArrayXXd strain_2d, \
                 DomainParams& dom, ThermodynamicsParams& thrm, DynamicsParams& dyn, BoundaryConditionsParams& bc)
{
    
    ArrayXXd theta_now(dom.n,dom.n_z);
    ArrayXd dx_inv(dom.n-1), dz_inv(dom.n), dz_2_inv(dom.n), Q_f_k(dom.n);
 
    // Evenly-spaced vertical grid, though x-dependency via ice thickness.
    dz_inv   = 1.0 / dz;
    dz_2_inv = pow(dz_inv, 2);
    dx_inv   = 1.0 / (ds * L);

    // Frictional heat units. // [Pa m yr^-1] = [J yr^-1 m^-2] = [W m^-2] => [K / m]
    Q_f_k = Q_fric / thrm.k;     
    
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
            theta_now(i,j) = theta(i,j) + dt * ( thrm.kappa * dz_2_inv(i) * \
                            ( theta(i,j+1) - 2.0 * theta(i,j) + theta(i,j-1) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,j) - theta(i-1,j) ) * dx_inv(i-1) + \
                            ( theta(i,j) - theta(i,j-1) ) * ( - w(j) ) * dz_inv(i) );
        }
        
        // Boundary conditions. Geothermal heat flow at the base.
        // We add friciton heat contribution Q_f_k. w(z=0) = 0 
        theta_now(i,0) = theta_now(i,1) + dz(i) * ( thrm.G_k + Q_f_k(i) ) + \
                            ( sigma(i) * dL_dt - ub(i) ) * \
                            ( theta(i,0) - theta(i-1,0) ) * dx_inv(i-1); 

        // Surface.
        theta_now(i,dom.n_z-1) = bc.therm.T_air;
    }

    // Due to symmetry theta_now(0,j) = theta_now(2,j). Ice divide.
    theta_now.row(0) = theta_now.row(2);
    
    // Boundary conditions at x = 0.
    //theta_now(0,0)     = theta_now(0,1) + dz(0) * G_k; 
    //theta_now(0,n_z-1) = T_air;

    // For the DIVA solver, consider the strain heat contribution.
    if ( dyn.vel_meth == "DIVA" || dyn.vel_meth == "Blatter-Pattyn" )
    {
        theta_now = theta_now + ( thrm.kappa / thrm.k ) * strain_2d * dt;
    }

    // TRY RELAXATION TO AVOID SPURIOUS RESULTS DURING SPIN-UP.
    double rel = 0.75;
    theta_now = ( 1.0 - rel ) * theta_now + rel * theta;

    // Pressure melting point as the upper bound.
    // theta = (theta.array() > 273.15).select(273.15, theta);
    theta_now = (theta_now.array() > thrm.theta_max).select(thrm.theta_max, theta_now);


    // Test for the grounding line column.
    //theta_now.row(dom.n-1) = theta_now.row(dom.n-2);

    return theta_now;
}