
// NIX MODULE WITH HANDY COMPUTATIONAL TOOLS.

ArrayXd gaussian_filter(ArrayXd w, double sigma, int p, int n)
{
    ArrayXd smth(n);
    ArrayXd summ = ArrayXd::Zero(n);

    double x, y;
    //double dx = 1.0e-3 * ds * L; // [m] --> [km]. L in metres but bed is in km.
    
    // Test forcing dx = 1.0. It must be 1.0 for y-amplitude consistency.
    double dx = 1.0;
    //sigma = 2.0 * dx;
    
    // Handy definition. Standard deviation is a multiple of dx (real distance among gridpoints).
    double sigma_inv = 1.0 / sigma;
    double A = sigma_inv / sqrt(2.0 * M_PI);
    
    // Weierstrass transform.
    for (int i=p; i<n-p; i++)
    {
        x = i * dx;

        for (int j=0; j<n; j++)
        {
            y = j * dx;
            summ(i) += w(j) * exp(- 0.5 * pow((x - y) * sigma_inv, 2) );
        }
    }

    // Normalizing Kernel.
    smth = A * summ;

    // The edges are identical to the original array.
    // (p-1) since the very element (n-1-p) must be also filled.
    smth.block(0,0,p,1)       = w.block(0,0,p,1);
    smth.block(n-1-p,0,p+1,1) = w.block(n-1-p,0,p+1,1); 

    return smth;
}


ArrayXd running_mean(ArrayXd x, int p, int n)
{
    ArrayXd y(n);
    double sum, k;

    // Assign values at the borders f(p).
    y = x;

    // Average.
    k = 1.0 / ( 2.0 * p + 1.0 );
 
    // Loop.
    for (int i=p; i<n-p; i++) 
    {
        sum = 0.0;

        for (int j=i-p; j<i+p+1; j++) 
        {
            //sum = sum + x(j);
            sum += x(j);
        }
        y(i) = k * sum;
    }

    // It cannot be out of the loop cause points at the boundaries arent averaged.
    //y = k * sum;
 
    return y;
}


