

// NIX TIMESTEP MODULE.

Array2d f_dt(double error, double picard_tol, double dt_meth, \
            double t, double dt, double t_eq, double dt_min, \
            double dt_max, double dt_CFL, double rel)
{
    // Local variables.
    Array2d out;
    double dt_tilde;

    // Fixed time step.
    if ( dt_meth == 0 || t < t_eq )
    {
        dt = dt_min;
    }
    
    // Idea: adaptative timestep from Picard error.
    else
    {
        // Linear decrease of dt with Picard error.
        //dt_tilde = ( 1.0 - ( min(error, picard_tol) / picard_tol ) ) * \
                    ( dt_max - dt_min ) + dt_min;

        // Quadratic dependency.
        dt_tilde = ( 1.0 - pow( min(error, picard_tol) / picard_tol, 2) ) * \
                    ( dt_max - dt_min ) + dt_min;

        // Apply relaxation.
        dt = rel * dt + (1.0 - rel) * dt_tilde;

        // Ensure Courant-Friedrichs-Lewis condition is met.
        dt = min(dt, dt_CFL);
    }

    // Update time.
    //t = t + dt;
    t += dt;

    // Output variables.
    out(0) = t;
    out(1) = dt;

    return out;
}


