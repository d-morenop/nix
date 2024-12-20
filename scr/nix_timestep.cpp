

// NIX TIMESTEP MODULE.

Array2d f_dt(double L, double t, double dt, double u_bar_max, \
            double ds_min,double error, TimestepParams& tmstep, \
            TimeParams& tm, PicardParams& picard)
{
    // Local variables.
    Array2d out;
    double dt_tilde, dt_CFL;

    // Factor 0.5 is faster since it yields fewer Picard's iterations.
    dt_CFL   = 0.5 * ds_min * L / u_bar_max;
    //dt_CFL_w = 0.5 * dz_min / abs(w_max);


    // Smallest timestep for equilibration.
    if ( t < tmstep.t_eq_dt )
    {
        dt = tmstep.dt_min;
    }
    
    // Options after equilibration: fixed/adapted.
    else
    {
        // Fixed timestep.
        if ( tmstep.dt_meth == "fixed" )
        {
            //dt = tmstep.dt_min;
            dt = 1.0;
        }
        
        // Idea: adaptative timestep from Picard error.
        else if ( tmstep.dt_meth == "adapt" )
        {
            // Linear decrease of dt with Picard error.
            //dt_tilde = ( 1.0 - ( min(error, picard_tol) / picard_tol ) ) * \
                        ( dt_max - dt_min ) + dt_min;

            // Quadratic dependency.
            dt_tilde = ( 1.0 - pow( min(error, picard.tol) / picard.tol, 2) ) * \
                        ( tmstep.dt_max - tmstep.dt_min ) + tmstep.dt_min;

            // Apply relaxation.
            dt = tmstep.rel * dt + (1.0 - tmstep.rel) * dt_tilde;

            // If thermodynamic solver is applied, consider stability in explicit temperature solution.
            /*if ( thrm.therm == true )
            {tm
                dt_CFL_min = min(dt_CFL, dt_CFL_w);
            }
            else
            {
                dt_CFL_min = dt_CFL;
            }*/
            
            // Ensure Courant-Friedrichs-Lewis condition is met (in both directions if necessary).
            //dt = min(dt, dt_CFL_min);
            dt = min(dt, dt_CFL);
        }
    }

    

    // Update time.
    t += dt;

    // Output variables.
    out(0) = t;
    out(1) = dt;

    return out;
}


