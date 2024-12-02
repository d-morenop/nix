

// NIX DYNAMICS MODULE.
// All velocity-related calculations.


ArrayXd tridiagonal_solver(ArrayXd A, ArrayXd B, ArrayXd C, \
                           ArrayXd F, int n)
{
    ArrayXd P(n), Q(n), x(n);
    double m;
    int j;
    
    // This allows us to perform O(n) iterations, rather than 0(n³).
    // A subdiagonal, B diagonal and C uppder-diagonal.
    
    // Ensure tridiagonal definition is satisfied.
    A(0)   = 0.0;
    C(n-1) = 0.0;

    // First element. Ensure that B(0) is non-zero!
    P(0) = - C(0) / B(0);
    Q(0) = F(0) / B(0);

    // Forward elimination.
    for (int i=1; i<n; i++)
    {
        m    = 1.0 / ( B(i) + A(i) * P(i-1) );
        P(i) = - C(i) * m;
        Q(i) = ( F(i) - A(i) * Q(i-1) ) * m;
    }

    // From notes: x(n-1) = Q(n-1).    
    x(n-1) = Q(n-1);      
    //cout << "\n ub(n-1) = " << ub(n-1);
    
    // Back substitution.
    for (int j = n-2; j>0; --j)
    {
        x(j) = P(j) * x(j+1) + Q(j);
    }

    return x;
}


ArrayXXd F_int_all(ArrayXXd visc, ArrayXd H, ArrayXd dz, int n_z, int n) {
    
    ArrayXXd F_all(n,n_z+1), F_1(n,n_z);
    ArrayXd F_2(n);

    double z, sum_1, sum_2, value_1, value_2, H_minus_z;
    int n_2 = 2;
    
    // Horizontal loop.
    for (int i = 0; i < n; ++i) 
    {
        z = 0;
        sum_1 = 0;
        sum_2 = 0;

        
        // Vertical integration.
        for (int j = 0; j < n_z; ++j) 
        {
            // Current vertical height.
            H_minus_z = ( H(i) - z ) / H(i);
            
            // F_2 integral is a 1D array.
            value_2 = pow(H_minus_z, n_2) / visc(i, j);
            sum_2  += value_2;

            // F_1 integral is a 2D array.
            value_1  = H_minus_z / visc(i, j);
            sum_1   += value_1;
            F_1(i,j) = sum_1;

            // Update vertical height.
            z += dz(i);
        }
        
        // Integral value.
        F_1.row(i) = dz(i) * F_1.row(i);
        F_2(i)     = dz(i) * sum_2;
        
    }
    
    // Allocate solutions.
    //F_all.block(0,0,n,1)   = F_2;
    //F_all.block(0,1,n,n_z) = F_1;

    F_all << F_2, F_1;
    
    return F_all;
}


ArrayXXd solver_2D(int n, int n_z, ArrayXd dx, ArrayXd dz, \
            ArrayXXd visc, ArrayXd F, ArrayXXd u_0) 
{

    ArrayXd dz_2_inv = 1.0 / pow(dz,2);
    ArrayXd gamma    = 4.0 / pow(dx, 2); // 4.0 / pow(dx, 2)

    // Inhomogeneous term: A*x = b.
    VectorXd b = VectorXd::Zero(n * n_z);

    // Build initial guess from previous velocity solution u_0.
    // Border should be zero as the solver does not include boundary conditions.
    /*
    u_0.col(0)     = ArrayXd::Zero(n);
    u_0.col(n_z-1) = ArrayXd::Zero(n);
    u_0.row(0)     = ArrayXd::Zero(dom.n_z);
    u_0.row(n-1)   = ArrayXd::Zero(dom.n_z);

    Map<VectorXd> x_0(u_0.data(), n*n_z);
    */
    
    // Initialize a triplet list to store non-zero entries.
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // Reserve memory for triplets.
    // 5 unkowns in a n*n_z array.
    tripletList.reserve(5 * (n-2) * (n_z-2));  

    
    // Loop through grid points
    for (int i=1; i<n-1; i++) 
    {
        for (int j=1; j<n_z-1; j++) 
        {
            // New index.
            int idx = i*n_z + j;

            // Compute coefficients. This works!!
            /*
            double c_x1 = gamma * visc(i+1,j);
            double c_x  = gamma * visc(i,j);

            double c_z1 = dz_2_inv(i) * visc(i,j+1);
            double c_z  = dz_2_inv(i) * visc(i,j);
            */
            
            double c_x1 = gamma(i) * visc(i+1,j);
            double c_x  = gamma(i-1) * visc(i,j);

            double c_z1 = dz_2_inv(i) * visc(i,j+1);
            double c_z  = dz_2_inv(i) * visc(i,j);
            
           
           // From the staggered grid definition, we should not take points at n_z-1 (j+1)
           // to calculate the velocities at j = n_z-2. 
           if ( j == n_z-2 )
           {
            c_z1 = 0.0;
           }
            

            // Add non-zero entries to the triplet list
            tripletList.push_back(T(idx, idx, - ( c_x1 + c_x + c_z1 + c_z )));
            tripletList.push_back(T(idx, idx+n_z, c_x1));
            tripletList.push_back(T(idx, idx-n_z, c_x));
            tripletList.push_back(T(idx, idx+1, c_z1));
            tripletList.push_back(T(idx, idx-1, c_z));
            

            // Fill vector b.
            b(idx) = F(i);
        }
    }

    // Set the triplets in the sparse matrix
    // declares a column-major sparse matrix type of double.
    //SparseMatrix<double> A_sparse(n*n_z, n*n_z); 
    SparseMatrix<double,RowMajor> A_sparse(n*n_z, n*n_z); 
    
    // Define your sparse matrix A_spare from triplets.
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    // Solver.
    BiCGSTAB<SparseMatrix<double> > solver;
    //solver.compute(A_sparse);

    // Preconditioner. It works as fast as using the previous vel sol as guess x_0.
    // If we use an initial guess x_0 from previous iter, do not use a preconditioner.
    IncompleteLUT<double> preconditioner;
    preconditioner.setDroptol(1.0e-4); // 1.0e-4. Set ILU preconditioner parameters
    solver.preconditioner().compute(A_sparse);
    solver.compute(A_sparse);
    

    // Set tolerance and maximum number of iterations.
    int maxIter = 100;                   // 100, 50
    double tol = 1.0e-3;                // 1.0e-3, 1.0e-2
    solver.setMaxIterations(maxIter);
    solver.setTolerance(tol);

    // Solve without guess (assumes x = 0).
    VectorXd x = solver.solve(b);

    // Solve with first guess x_0.
    //solver.compute(A_sparse);
    //VectorXd x = solver.solveWithGuess(b, x_0);
    
    cout << "\n #iterations:     " << solver.iterations();
    cout << "\n Estimated error: " << solver.error();
    //cout << "\n Solver info:     " << solver.info();
    
    /* ... update b ... */
    // THINK ABOUT THIS!!!!!!!!!!!
    //x = solver.solve(b); // solve again??? No need to update b in our case.
        

    /* Copy values from x into u (no memory reallocation).
    Unlike u = x.reshaped<RowMajor>(n,n_z), where there is reallocation.

    We create a mapped view u using Eigen::Map. 
    This mapped view u will have the desired shape without memory reallocation.
    
    When you add elements to a dynamically allocated data structure and it reaches its 
    current capacity, the data structure may need to allocate a larger chunk of memory, 
    copy the existing data to the new memory location, and then release the old memory.
    
    In row-major order, the elements of a matrix are stored in memory row by row. 
    This means that consecutive elements in the same row are stored next to each other in memory.*/
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> u(x.data(), n, n_z);

    return u;
}


ArrayXXd vel_solver(ArrayXd H, ArrayXd ds, ArrayXd ds_inv, ArrayXd dz, ArrayXd visc_bar, \
                    ArrayXd bed, double L, ArrayXd C_bed, \
                    double t, ArrayXd beta, double A_ice, ArrayXXd A_theta, ArrayXXd visc, ArrayXXd u, ArrayXXd u_z, \
                    DynamicsParams& dyn , DomainParams& dom, ConstantsParams& cnst, ViscosityParams& vis)
{

    // Prepare variables.
    ArrayXXd out(dom.n,dom.n_z+1), u_sol(dom.n,dom.n_z); 

    ArrayXd u_bar(dom.n), dhds(dom.n), visc_H(dom.n), h(dom.n), A_bar(dom.n), \
            A(dom.n), B(dom.n), C(dom.n), F(dom.n), dz_inv_2(dom.n), \
            ds_inv_2(dom.n-1), gamma(dom.n-1);

    double D, u_x_bc, L_inv;
    
    // Handy definitions.
    L_inv    = 1.0 / L;
    ds_inv_2 = pow(ds_inv, 2);
    dz_inv_2 = pow(1.0/dz, 2);


    // Note that ds has only (dom.n-1) point rather than (dom.n).
    gamma = 4.0 * ds_inv_2 * pow(L_inv, 2); // Factor 2 difference from Vieli and Payne (2005).
    
    // Handy definitions.
    h = bed + H;           // Ice surface elevation.

    ///////////////////////////////////////
    ///////////////////////////////////////
    // Staggered grid (Vieli and Payne solutions, appendix).

    // SSA and DIVA solvers.
    if ( dyn.vel_meth == "SSA" || dyn.vel_meth == "DIVA" )
    {
        // Handy definitions.
        visc_H = visc_bar * H;

        // Staggered grid (Vieli and Payne solutions, appendix).
        for (int i=1; i<dom.n-1; i++)
        {
            // Surface elevation gradient. Centred stencil.
            dhds(i) = 0.5 * ( H(i) + H(i+1) ) * ( h(i+1) - h(i) ) * ds_inv(i);

            // Diagonal, B; lower diagonal, A; upper diagonal, C.
            A(i) = gamma(i-1) * visc_H(i);
            B(i) = - gamma(i) * ( visc_H(i) + visc_H(i+1) ) - beta(i);
            C(i) = gamma(i) * visc_H(i+1);

        }

        // Derivatives at the boundaries O(x).
        dhds(0)       = 0.5 * ( H(0) + H(1) ) * ( h(1) - h(0) ) * ds_inv(0);
        dhds(dom.n-1) = H(dom.n-1) * ( h(dom.n-1) - h(dom.n-2) ) * ds_inv(dom.n-2); 
        // Failed attempt for second order derivative.
        //dhds(dom.n-1) = H(dom.n-1) * 0.5 * ( 3.0 * h(dom.n-1) - 4.0 * h(dom.n-2) + h(dom.n-3) );

        // Tridiagonal boundary values. 
        A(0) = 0.0;
        B(0) = - gamma(0) * ( visc_H(0) + visc_H(1) ) - beta(0);
        C(0) = gamma(1) * visc_H(1); // gamma(0)

        A(dom.n-1) = gamma(dom.n-2) * visc_H(dom.n-1);
        B(dom.n-1) = - gamma(dom.n-2) * visc_H(dom.n-1) - beta(dom.n-1);
        C(dom.n-1) = 0.0;

        
        // Inhomogeneous term.
        F = cnst.rho * cnst.g * dhds * L_inv;


        // Grounding line sigma = 1 (x = L). 
        D = abs( min(0.0, bed(dom.n-1)) );   

        // Imposed rate factor.
        if ( vis.therm == false )
        {
            u_x_bc = L * A_ice * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
                        (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);
        }

        // Temperature-dependent ice rate factor.
        else if ( vis.therm == true )
        {
            // Vertically averaged ice rate factor.
            A_bar = A_theta.rowwise().mean();

            // Boundary condition.
            u_x_bc = L * A_bar(dom.n-1) * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
                        (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);
        }
           
        // TRIDIAGONAL SOLVER.
        u_bar = tridiagonal_solver(A, B, C, F, dom.n); 

        // Replace potential negative values with given value. (result.array() < 0).select(0, result);
        // Sometimes, u_bar(1) < 0 so that u_bar(0) > 0 after the BC and the model crashes.
        // Necessary for SSA solver, not DIVA.
        //u_bar = (u_bar < 0.0).select(0.0, u_bar);
        u_bar = (u_bar < 0.0).select(0.25 * u_bar(2), u_bar);

        // This works but yields extremely thick ice at the divde.
        //u_bar = (u_bar < 0.0).select(0.0, u_bar);

        // Boundary conditions.
        // Ice divide: symmetry x = 0.
        u_bar(0)   = - u_bar(1);
        u_bar(dom.n-1) = u_bar(dom.n-2) + ds(dom.n-2) * u_x_bc;
    }
    
    else if ( dyn.vel_meth == "Blatter-Pattyn" )
    {

        ArrayXd dx(dom.n-1), dx_inv(dom.n-1), dx_2_inv(dom.n-1);
        dx       = ds * L;
        dx_inv   = 1.0 / dx;
        dx_2_inv = pow(dx_inv,2);    

        //cout << "\n dx = " << dx;

        for (int i=0; i<dom.n-1; i++)
        {
            // Surface elevation gradient.
            dhds(i) = ( h(i+1) - h(i) ) * dx_inv(i);

            // Unstable.
            //dhds(i) = 0.5 * ( h(i+1) - h(i-1) );
        }
        
        // Boundaries.
        dhds(dom.n-1) = ( h(dom.n-1) - h(dom.n-2) ) * dx_inv(dom.n-2); 

        // Inhomogeneous term.
        F = cnst.rho * cnst.g * dhds;

        // Blatter-Pattyn solution.
        u_sol = solver_2D(dom.n, dom.n_z, dx, dz, visc, F, u); 

        
        // VELOCITY BOUNDARY CONDITIONS.
        // Eq. 25, Pattyn (2003).
        for (int i=1; i<dom.n-1; i++)
        {
            // Derivative from current velocity sol.
            // Centred differences and changed sign.
            
            //cout << "\n i              = " << i;
            //cout << "\n u_sol(i,1)     = " << u(i,1);

            // Use previous iteration solution to calculate basal friction.
            double alpha = 4.0 * ( u_sol(i,1) - u_sol(i-1,1) ) * abs( bed(i+1) - bed(i) ) * dx_2_inv(i) + \
                             0.5 * beta(i) * u(i,0) / visc(i,0);

            u_sol(i,0) = u_sol(i,1) - dz(i) * alpha;
            u_sol(i,0) = max(1.0, u_sol(i,0));
            
            //cout << "\n i          = " << i;
            //cout << "\n visc(i,1)    = " <<  visc(i,1);
            //cout << "\n u_sol(i,0) = " << u_sol(i,0);

            /*
            double alpha_0 = dz(i) * 4.0 * ( 0.5 * beta(i) * u_sol(i,0) / visc(i,0) + \
                                        0.5 * ( u_sol(i+1,0) - u_sol(i-1,0) ) *  \
                                            abs( bed(i+1) - bed(i) ) * dx_2_inv );
            
            u_sol(i,0) = ( 4.0 * u_sol(i,1) - u_sol(i,2) - alpha_0 ) / 3.0;
            */
            
            // Free surface. Pattyn (2003).
            //cout << "\n dx_2_inv     = " << dx_2_inv;
            //cout << "\n abs(du/dx)      = " << abs(u_sol(i+1,n_z-2) - u_sol(i,n_z-2)) * dx_inv;
            //cout << "\n abs(dh/dx) = " << abs(dhds(i)) * dx_inv;
            
            // Three-point vertical derivative. As Pattyn (2003).
            double alpha_h = dz(i) * 4.0 * ( u_sol(i+1,dom.n_z-2) - u_sol(i,dom.n_z-2) ) *  \
                                         abs(dhds(i)) * dx_2_inv(i);
            
            // MIT calculator.
            // du/dz = 0.5 * ( 3.0 * u(dom.n_z-1) - 4.0 * u(dom.n_z-2) + 1.0 * u(dom.n_z-3) ) 
            u_sol(i,dom.n_z-1) = ( 4.0 * u_sol(i,dom.n_z-2) - u_sol(i,dom.n_z-3) + 2.0 * alpha_h ) / 3.0;

            // Vieli 1 (Eq. B13). BEST ONE!
            // du/dz = ( - 4.0 * u(dom.n_z-1) + 3.0 * u(dom.n_z-2) + 1.0 * u(dom.n_z-3) ) / 3.0
            //u_sol(i,n_z-1) = ( 4.0 * u_sol(i,n_z-2) + 3.0 * u_sol(i,n_z-3) - 3.0 * alpha_h ) / 4.0;

            // Test to avoid velocity differences between u_sol.col(dom.n_z-1) and u_sol.col(dom.n_z-2).
            // It works, but too mild effect.
            //u_sol(i,n_z-2) = 0.25 * ( 4.0 * u_sol(i,n_z-1) - 3.0 * u_sol(i,n_z-3) + 3.0 * alpha_h );
        }

        // Test to avoid velocity differences between u_sol.col(dom.n_z-1) and u_sol.col(dom.n_z-2).
        //u_sol.col(dom.n_z-2) = u_sol.col(dom.n_z-1);

        

        // Hydrostatic equilibrium with the ocean.
        for (int j=0; j<dom.n_z; j++)
        {
            // Stable.
            //u_x_bc = dx * A_ice * pow( 0.25 * ( cnst.rho * cnst.g * ( H(dom.n-1) - j * dz(dom.n-1) ) * \
            //                          (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);

            //u_x_bc = dx(dom.n-2) * A_ice * pow( 0.25 * ( cnst.rho * cnst.g * ( H(dom.n-1) - j * dz(dom.n-1) ) * \
                                      (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);
            
            // Same BC regardless of particular vertical layer depth?
            u_x_bc = dx(dom.n-2) * A_ice * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
                                        (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);
            u_sol(dom.n-1,j) = u_sol(dom.n-2,j) + u_x_bc;
        }

        
        // Ensure positive velocities.
        //u_sol = (u_sol < 0.0).select(0.0, u_sol);
        u_sol = (u_sol < 0.1).select(0.1, u_sol);

        // Symmetry at the ice divide.
        u_sol.row(0) = - u_sol.row(1);

        // Vertically averaged velocity from full Blatter-Pattyn solution.
        u_bar = u_sol.rowwise().mean();
        
    }
    

    // Allocate solutions.
    out << u_bar, u_sol;

    
    return out;
}

// Vertical velocity calculation.
ArrayXXd f_w(ArrayXd u_bar_x, ArrayXd H, ArrayXd dz, ArrayXd b_melt, ArrayXd u_bar, \
             ArrayXd bed, ArrayXd ds, double L, DomainParams& dom)
{
    ArrayXXd w(dom.n, dom.n_z);
    ArrayXd dx_inv = 1.0 / ( ds * L );

    // Evaluate bedrock geometry gradient.
    ArrayXd bed_x(dom.n), w_b(dom.n);
    for (int i=0; i<dom.n-1; i++)
    { 
        // Forward derivative (just to be consistent with surface gradient in vel solver).
        bed_x(i) = dx_inv(i) * ( bed(i+1) - bed(i) );
    }

    // Boundary: grounding line.
    bed_x(dom.n-1) = ( bed(dom.n-1) - bed(dom.n-2) ) * dx_inv(dom.n-2);

    // Vertical velocity at the base is basal melt plus sliding term (u_bar if SSA or DIVA; u in BP).
    // Eq. 5 in In-Woo Park et al. (2024): https://tc.copernicus.org/articles/18/1139/2024/tc-18-1139-2024.html
    w_b = u_bar * bed_x - b_melt;

    // For the SSA and DIVA. H(i) = n_z * dz(i) --> w.col(j) = u_bar_x * H_norm * ( j / dom.n_z )
    // Add contribution from basal melt in [m / yr].
    ArrayXd H_norm = H / dom.n_z;
    for (int j=0; j<dom.n_z; j++)
    { 
        // Include basal melting b_melt. Eq. 4, In-Woo Park et al. (2024).
        w.col(j) = w_b - u_bar_x * H_norm * j;
    }


    // For now, test in (n-1) and (n-2) with the same vertical velocity profile??
    // To avoid instabilities at the grounding line?
    //w.row(dom.n-1) = w.row(dom.n-2);

    // Ensure negative values to be consisten with vertical adv discretisation.
    //w = - abs(w);
    w = (w > 0.0).select(0.0, w);
    
    
    // Currently assume an evenly-spaced vertical coordinate dz(x).
    //w.row(i) = w.row(i) * H(i);

    return w;
}
