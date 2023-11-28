

// NIX THERMODYNAMICS MODULE.

ArrayXXd f_theta(ArrayXXd theta, ArrayXd ub, ArrayXd H, ArrayXd tau_b, ArrayXd Q_fric, \
                 ArrayXd sigma, ArrayXd dz, double theta_max, double T_air, double kappa, double k, \
                 double dt, double G_k, ArrayXd ds, double L, \
                 double dL_dt, double t, double t_eq, ArrayXd w, int n, int n_z, string vel_meth, \
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
    if ( vel_meth == "DIVA" || vel_meth == "Blatter-Pattyn" )
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