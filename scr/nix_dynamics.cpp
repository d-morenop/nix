

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
    Map<ArrayXXd,RowMajor> x_0(u_0.data(), n,n_z);
    u_0.col(0)     = ArrayXd::Zero(n);
    u_0.col(n_z-1) = ArrayXd::Zero(n);
    u_0.row(0)     = ArrayXd::Zero(n_z);
    u_0.row(n-1)   = ArrayXd::Zero(n_z);

    Map<VectorXd> x_0(u_0.data(), n*n_z);*/

    /*u_0 = 0.5 * u_0.reshaped<RowMajor>(n,n_z);
    Map<VectorXd> x_0(u_0.data(), n*n_z);*/

    //VectorXd x_0 = VectorXd::Constant(n * n_z, 1.0);
    
    // Initialize a triplet list to store non-zero entries.
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // Reserve memory for triplets.
    // 5 unkowns in a n*n_z array.
    tripletList.reserve(5 * (n-2) * (n_z-2));  

    // Handy definitions.
    ArrayXXd gamma_mat = gamma.replicate(1, n_z);
    ArrayXXd dz_2_mat  = dz_2_inv.replicate(1, n_z);
 
    ArrayXXd c_x1 = gamma_mat * shift_2D(visc,-1,0);
    ArrayXXd c_x  = shift_2D(gamma_mat,1,0) * visc;

    ArrayXXd c_z1 = dz_2_mat * shift_2D(visc,0,-1); //(i,j+1);
    ArrayXXd c_z  = dz_2_mat * visc;

    // From the staggered grid definition, we should not take points at n_z-1 (j+1)
    // to calculate the velocities at j = n_z-2. 
    c_z1.col(n_z-2) = 0.0;

    //c_x1.row(n-2) = 0.0;

    ArrayXXd alpha = - ( c_x1 + c_x + c_z1 + c_z );

    
    // Loop through grid points
    for (int i=1; i<n-1; i++) 
    {
        for (int j=1; j<n_z-1; j++) 
        {
            // New index.
            int idx = i*n_z + j;

            // Add non-zero entries to the triplet list
            tripletList.push_back(T(idx, idx, alpha(i,j)));
            tripletList.push_back(T(idx, idx+n_z, c_x1(i,j)));
            tripletList.push_back(T(idx, idx-n_z, c_x(i,j)));
            tripletList.push_back(T(idx, idx+1, c_z1(i,j)));
            tripletList.push_back(T(idx, idx-1, c_z(i,j)));
            

            // Fill vector b.
            b(idx) = F(i);
        }
    }


    /*typedef Eigen::Triplet<double> T;

    std::vector<T> globalTripletList;

    // Loop through grid points
    #pragma omp parallel
    {
        std::vector<T> tripletList;

        #pragma omp for collapse(2) schedule(static)
        for (int i = 1; i < n - 1; i++) 
        {
            for (int j = 1; j < n_z - 1; j++) 
            {
                int idx = i * n_z + j;

                // Compute coefficients
                double c_x1 = gamma(i) * visc(i + 1, j);
                double c_x  = gamma(i - 1) * visc(i, j);
                double c_z1 = dz_2_inv(i) * visc(i, j + 1);
                double c_z  = dz_2_inv(i) * visc(i, j);

                // Handle boundary condition
                if (j == n_z - 2) {
                    c_z1 = 0.0;
                }

                // Add entries to local triplet list
                tripletList.push_back(T(idx, idx, -(c_x1 + c_x + c_z1 + c_z)));
                tripletList.push_back(T(idx, idx + n_z, c_x1));
                tripletList.push_back(T(idx, idx - n_z, c_x));
                tripletList.push_back(T(idx, idx + 1, c_z1));
                tripletList.push_back(T(idx, idx - 1, c_z));

                // Fill vector b
                b(idx) = F(i);
            }
        }

        // Merge local triplet lists into a global list
        #pragma omp critical
        {
            globalTripletList.insert(globalTripletList.end(), tripletList.begin(), tripletList.end());
        }
    }

    // Construct the sparse matrix
    SparseMatrix<double, RowMajor> A_sparse(n * n_z, n * n_z);
    A_sparse.setFromTriplets(globalTripletList.begin(), globalTripletList.end());*/



    // Set the triplets in the sparse matrix
    // declares a column-major sparse matrix type of double.
    SparseMatrix<double,RowMajor> A_sparse(n*n_z, n*n_z); 
    //SparseMatrix<double,ColMajor> A_sparse(n*n_z, n*n_z); 
    
    // Define your sparse matrix A_spare from triplets.
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());


    // Solver.
    BiCGSTAB<SparseMatrix<double,RowMajor> > solver;
    
    //ConjugateGradient<SparseMatrix<double> > solver;
    //LeastSquaresConjugateGradient<SparseMatrix<double> > solver;
    //solver.compute(A_sparse);

    // Preconditioner. It works as fast as using the previous vel sol as guess x_0.
    // If we use an initial guess x_0 from previous iter, do not use a preconditioner.
    //IncompleteLUT<double> preconditioner;
    //preconditioner.setDroptol(1.0e-4); // 1.0e-4. Set ILU preconditioner parameters
    //solver.preconditioner().compute(A_sparse);
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
    //Map<Matrix<double,Dynamic,Dynamic,ColMajor>> u(x.data(), n, n_z);

    return u;
}


ArrayXXd vel_solver(ArrayXd H, ArrayXd ds, ArrayXd ds_inv, ArrayXd ds_u_inv, ArrayXd dz, ArrayXd visc_bar, \
                    ArrayXd bed, double L, ArrayXd C_bed, \
                    double t, ArrayXd beta, double A_ice, ArrayXXd A_theta, ArrayXXd visc, ArrayXXd u, ArrayXXd u_z, \
                    DynamicsParams& dyn , DomainParams& dom, ConstantsParams& cnst, ViscosityParams& vis)
{

    // Prepare variables.
    ArrayXXd out(dom.n,dom.n_z+1), u_sol(dom.n,dom.n_z); 

    ArrayXd u_bar(dom.n), dhds(dom.n), visc_H(dom.n), h(dom.n), A_bar(dom.n), \
            A(dom.n), B(dom.n), C(dom.n), F(dom.n), dz_inv_2(dom.n), \
            ds_inv_2(dom.n-1), gamma(dom.n-1), gamma_2(dom.n-1);

    double D, u_x_bc, L_inv;
    
    // Handy definitions.
    L_inv    = 1.0 / L;
    ds_inv_2 = pow(ds_inv, 2);
    dz_inv_2 = pow(1.0/dz, 2);


    // Note that ds has only (dom.n-1) point rather than (dom.n).
    gamma_2 = 4.0 * ds_inv_2 * pow(L_inv, 2); // Factor 2 difference from Vieli and Payne (2005).
    gamma   = 2.0 * ds_inv * pow(L_inv, 2);
    
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
        /*for (int i=1; i<dom.n-1; i++)
        {
            // Surface elevation gradient. Centred stencil.
            dhds(i) = 0.5 * ( H(i) + H(i+1) ) * ( h(i+1) - h(i) ) * ds_inv(i);

            // Diagonal, B; lower diagonal, A; upper diagonal, C.
            //A(i) = gamma(i-1) * visc_H(i);
            //B(i) = - gamma(i) * ( visc_H(i) + visc_H(i+1) ) - beta(i);
            //C(i) = gamma(i) * visc_H(i+1);

            //////////////////////////////////////////////////////////////////////////
            // Diagonal, B; lower diagonal, A; upper diagonal, C.
            A(i) = 2.0 * gamma(i) * ds_u_inv(i-1) * visc_H(i);
            B(i) = - gamma(i) * ( ds_u_inv(i-1) + ds_u_inv(i) ) * \
                                ( visc_H(i) + visc_H(i+1) ) - beta(i);
            C(i) = 2.0 * gamma(i) * ds_u_inv(i) * visc_H(i+1);

            //////////////////////////////////////////////////////////////////////////

        }*/


        // Surface elevation gradient. Centred stencil.
        //dhds(i) = 0.5 * ( H(i) + H(i+1) ) * ( h(i+1) - h(i) ) * ds_inv(i);
        dhds = 0.5 * ( H + shift(H,-1,dom.n) ) * ( shift(h,-1,dom.n) - h ) * ds_inv;

        A = 2.0 * gamma * shift(ds_u_inv,1,dom.n) * visc_H;

        B = - gamma * ( shift(ds_u_inv,1,dom.n) + ds_u_inv ) * \
                            ( visc_H + shift(visc_H,-1,dom.n) ) - beta;
        
        C = 2.0 * gamma * ds_u_inv * shift(visc_H,-1,dom.n);
        

        // Derivatives at the boundaries O(x).
        dhds(0)       = 0.5 * ( H(0) + H(1) ) * ( h(1) - h(0) ) * ds_inv(0);
        dhds(dom.n-1) = H(dom.n-1) * ( h(dom.n-1) - h(dom.n-2) ) * ds_inv(dom.n-2); 
        // Failed attempt for second order derivative.
        //dhds(dom.n-1) = H(dom.n-1) * 0.5 * ( 3.0 * h(dom.n-1) - 4.0 * h(dom.n-2) + h(dom.n-3) );

        // Tridiagonal boundary values. 
        A(0) = 0.0;
        B(0) = - gamma_2(0) * ( visc_H(0) + visc_H(1) ) - beta(0);
        C(0) = gamma_2(1) * visc_H(1); // gamma(0)

        A(dom.n-1) = gamma_2(dom.n-2) * visc_H(dom.n-1);
        B(dom.n-1) = - gamma_2(dom.n-2) * visc_H(dom.n-1) - beta(dom.n-1);
        C(dom.n-1) = 0.0;

        
        // Inhomogeneous term.
        F = cnst.rho * cnst.g * dhds * L_inv;


        // Grounding line sigma = 1 (x = L). 
        //D = abs( min(0.0, bed(dom.n-1)) );  
        
        // Temperature-dependent rate factor.
        if ( vis.therm == true )
        {
            // Vertically averaged ice rate factor.
            A_bar = A_theta.rowwise().mean();
            A_ice = A_bar(dom.n-1);
        }

        // Hydrostatic boundary condition.
        u_x_bc = L * A_ice * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
                        (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);

        // Imposed rate factor.
        /*if ( vis.therm == false )
        {
            u_x_bc = L * A * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
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
        }*/
           
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
        ArrayXd dx       = ds * L;
        ArrayXd dx_inv   = 1.0 / dx;
        ArrayXd dx_2_inv = pow(dx_inv,2);   


        /*for (int i=0; i<dom.n-1; i++)
        {
            // Surface elevation gradient.
            dhds(i) = ( h(i+1) - h(i) ) * dx_inv(i);

            // Unstable.
            //dhds(i) = 0.5 * ( h(i+1) - h(i-1) );
        }*/

        // Surface elevation gradient.
        dhds = ( shift(h,-1,dom.n) - h ) * dx_inv;
        
        // Boundaries.
        dhds(dom.n-1) = ( h(dom.n-1) - h(dom.n-2) ) * dx_inv(dom.n-2); 

        // Inhomogeneous term.
        F = cnst.rho * cnst.g * dhds;

        // Blatter-Pattyn solution.
        u_sol = solver_2D(dom.n, dom.n_z, dx, dz, visc, F, u); 

        
        // VELOCITY BOUNDARY CONDITIONS.
        // Eq. 25, Pattyn (2003).
        /*for (int i=1; i<dom.n-1; i++)
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
        }*/

        
        // Vector form.
        // Basal boundary condition (friction).
        ArrayXd alpha = 4.0 * ( u_sol.col(1) - shift(u_sol.col(1),1,dom.n) ) \
                            * abs( shift(bed,-1,dom.n) - bed ) * dx_2_inv + \
                            0.5 * beta * u.col(0) / visc.col(0);

        u_sol.col(0) = u_sol.col(1) - dz * alpha;

        // Upper boundary condition (free surface).
        // Three-point vertical derivative. As Pattyn (2003).
        ArrayXd alpha_h = dz * 4.0 * ( shift(u_sol.col(dom.n_z-2),-1,dom.n) - u_sol.col(dom.n_z-2) ) \
                                    * abs(dhds) * dx_2_inv;
        
        // MIT calculator.
        // du/dz = 0.5 * ( 3.0 * u(dom.n_z-1) - 4.0 * u(dom.n_z-2) + 1.0 * u(dom.n_z-3) ) 
        u_sol.col(dom.n_z-1) = ( 4.0 * u_sol.col(dom.n_z-2) - u_sol.col(dom.n_z-3) + 2.0 * alpha_h ) / 3.0;


        // Test to avoid velocity differences between u_sol.col(dom.n_z-1) and u_sol.col(dom.n_z-2).
        //u_sol.col(dom.n_z-2) = u_sol.col(dom.n_z-1);

        
        // Hydrostatic equilibrium with the ocean.
        /*for (int j=0; j<dom.n_z; j++)
        {
            u_x_bc = dx(dom.n-2) * A_theta(dom.n-1,j) * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
                                      (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);
                                        
            u_sol(dom.n-1,j) = u_sol(dom.n-2,j) + u_x_bc;
        }*/


        // Hydrostatic equilibrium with the ocean.
        // Vector form. Depth averaged is correct since the bounadry condition
        // does not include vertical derivatives. See MALI ice sheet paper and also
        // Eq. 6.59 (Greve and Blatter, 2009).
        u_x_bc = dx(dom.n-2) * A_ice * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
                                        (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);

        //u_x_bc = dx(dom.n-2) * A_theta(dom.n-1,0) * pow( 0.25 * ( cnst.rho * cnst.g * H(dom.n-1) * \
                                        (1.0 - cnst.rho / cnst.rho_w) ), vis.n_gln);
        u_sol.row(dom.n-1) = u_sol.row(dom.n-2) + u_x_bc;

        
        // Ensure positive velocities.
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
             ArrayXd bed, ArrayXXd u, ArrayXXd u_x, ArrayXd ds, double L, \
             DomainParams& dom, DynamicsParams& dyn)
{
    ArrayXXd w(dom.n, dom.n_z);
    ArrayXd dx_inv = 1.0 / ( ds * L );

    // Evaluate bedrock geometry gradient.
    /*ArrayXd bed_x(dom.n), w_b(dom.n);
    for (int i=0; i<dom.n-1; i++)
    { 
        // Forward derivative (just to be consistent with surface gradient in vel solver).
        bed_x(i) = dx_inv(i) * ( bed(i+1) - bed(i) );
    }*/

    ArrayXd bed_x = dx_inv * ( shift(bed,-1,dom.n) - bed );

    // Boundaries
    bed_x(0)       = ( bed(1) - bed(0) ) * dx_inv(0);
    bed_x(dom.n-1) = ( bed(dom.n-1) - bed(dom.n-2) ) * dx_inv(dom.n-2);

    // Create 2D velocity array for dimensions consistency.
    if ( dyn.vel_meth == "SSA" || dyn.vel_meth == "DIVA" )
    {
        u_x = u_bar_x.replicate(1, dom.n_z);
    }
    

    // Vertical velocity at the base is basal melt plus sliding term (u_bar if SSA or DIVA; u in BP).
    // Eq. 5 in In-Woo Park et al. (2024): https://tc.copernicus.org/articles/18/1139/2024/tc-18-1139-2024.html
    /*ArrayXd w_b = u_bar * bed_x - b_melt;

    // For the SSA and DIVA. H(i) = n_z * dz(i) --> w.col(j) = u_bar_x * H_norm * ( j / dom.n_z )
    // Ideally, the BP model should consider u_x and not u_bar_x, but it gives numerical issues now.
    ArrayXd H_norm = H / dom.n_z;
    for (int j=0; j<dom.n_z; j++)
    { 
        // Include basal melting b_melt. Eq. 4, In-Woo Park et al. (2024).
        w.col(j) = w_b - u_bar_x * H_norm * j;
    }*/



    // THIS IS THE NECESSARY CALCULATION FOR THE BLATTER-PATTYN MODEL.
    // HOWEVER, THERE ARE STILL SOME ISSUES WITH THE DERIVATIVES.
    ArrayXd w_b = u.col(0) * bed_x - b_melt;
    //ArrayXd w_b = abs(u.col(0)) * bed_x - b_melt;

    // For the SSA and DIVA. H(i) = n_z * dz(i) --> w.col(j) = u_bar_x * H_norm * ( j / dom.n_z )
    // Add contribution from basal melt in [m / yr].
    ArrayXd H_norm = H / dom.n_z;
    for (int j=0; j<dom.n_z; j++)
    { 
        // Include basal melting b_melt. Eq. 4, In-Woo Park et al. (2024).
        w.col(j) = w_b - u_x.col(j) * H_norm * j;
        //w.col(j) = w_b - abs(u_x.col(j)) * H_norm * j;
    }

    // Vertical velocity defined in H-grid.
    //w.row(0) = w.row(2);


    // For now, test in (n-1) and (n-2) with the same vertical velocity profile??
    // To avoid instabilities at the grounding line?
    //w.row(dom.n-1) = w.row(dom.n-2);

    // Ensure negative values to be consisten with vertical adv discretisation.
    w = (w > 0.0).select(0.0, w);
    //w = (w > -1.0e-1).select(-1.0e-1, w);


    // Test for BP.
    //w = ArrayXXd::Zero(dom.n, dom.n_z);

    /*for (int j=1; j<dom.n_z; j++)
    {
        w.col(j) = - 0.1 * j / dom.n_z;
    }*/
    

    return w;
}
