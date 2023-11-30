

// NIX TIMESTEP MODULE.

/*Array2d f_dt(double error, double picard_tol, string dt_meth, \
            double t, double dt, double t_eq, double dt_min, \
            double dt_max, double dt_CFL, double rel)*/

Array2d f_dt(double L, double t, double dt, double u_bar_max, \
            double ds_min, double error, TimestepParams& tmstep, \
            TimeParams& tm, PicardParams& picard)
{
    // Local variables.
    Array2d out;
    double dt_tilde, dt_CFL;

    // Factor 0.5 is faster since it yields fewer Picard's iterations.
    dt_CFL = 0.5 * ds_min * L / u_bar_max;


    // Fixed time step.
    if ( tmstep.dt_meth == "fixed" || t < tm.t_eq )
    {
        dt = tmstep.dt_min;
    }
    
    // Idea: adaptative timestep from Picard error.
    else
    {
        // Linear decrease of dt with Picard error.
        //dt_tilde = ( 1.0 - ( min(error, picard_tol) / picard_tol ) ) * \
                    ( dt_max - dt_min ) + dt_min;

        // Quadratic dependency.
        dt_tilde = ( 1.0 - pow( min(error, picard.tol) / picard.tol, 2) ) * \
                    ( tmstep.dt_max - tmstep.dt_min ) + tmstep.dt_min;

        // Apply relaxation.
        dt = tmstep.rel * dt + (1.0 - tmstep.rel) * dt_tilde;

        // Ensure Courant-Friedrichs-Lewis condition is met.
        dt = min(dt, dt_CFL);
    }

    // Update time.
    t += dt;

    // Output variables.
    out(0) = t;
    out(1) = dt;

    return out;
}


