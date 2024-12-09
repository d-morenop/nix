
// NIX MODULE WITH HANDY COMPUTATIONAL TOOLS.

ArrayXd shift(ArrayXd array, int shift, int n) 
{
    ArrayXd result(n);

    // Normalize the shift value to be within the array size
    shift = ((shift % n) + n) % n;

    // Rearrange elements
    /*for (int i = 0; i < n; ++i) 
    {
        //cout << (i + shift) % n;
        result((i + shift) % n) = array(i);
    }*/

    result.block(shift,0,n-shift,1) = array.block(0,0,n-shift,1);
    result.block(0,0,shift,1)       = array.block(n-shift,0,shift,1);


    result.block(0,shift,1,n-shift) = array.block(0,0,1,n-shift);
    result.block(0,0,1,shift)       = array.block(0,n-shift,1,shift);

    return result;
}



ArrayXd shift_2D(ArrayXd x, int shift_x, int shift_z) 
{
    int n   = x.rows();    // Number of rows
    //int n_z = x.cols(); 

    ArrayXd result(n);

    // Normalize the shift value to be within the array size.
    shift_x = ((shift_x % n) + n) % n;
    
    result.block(shift_x,0,n-shift_x,1) = x.block(0,0,n-shift_x,1);
    result.block(0,0,shift_x,1)       = x.block(n-shift_x,0,shift_x,1);

    // 2D arrays.
    /*if ( shift_z != 0 )
    {
        shift_z = ((shift_z % n_z) + n_z) % n_z;
        result.block(0,shift_z,1,n_z-shift_z) = x.block(0,0,1,n_z-shift_z);
        result.block(0,0,1,shift_z)         = x.block(0,n_z-shift_z,1,shift_z);
    }*/
    

    return result;
}


ArrayXd gaussian_filter(ArrayXd w, ArrayXd sigma, ArrayXd ds, double sigma_gauss, int p, int n)
{
    ArrayXd smth(n);
    ArrayXd summ = ArrayXd::Zero(n);
    ArrayXd norm = ArrayXd::Zero(n);

    double kernel;

    // Sigma value is refered as a factor of the minimum separation among grid points.
    double sigma_new = ds(n-2) * sigma_gauss;
    
    // Handy definition. Standard deviation is a multiple of ds (dimensionless distance among gridpoints).
    double sigma_inv = 1.0 / sigma_new;
    
    // Weierstrass transform.
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            // Create gaussian kernel with distance from uniform grid sigma.
            kernel = exp(- 0.5 * pow((sigma(i) - sigma(j)) * sigma_inv, 2) );

            // Smoothed variable and normalisation from kernel.
            summ(i) += w(j) * kernel;
            norm(i) += kernel;
        }
    }

    // Normalised smooth. Equivalent to factor sigma_inv / sqrt(2.0 * M_PI).
    smth = summ / norm;
    
    //cout << "\n bed = " << w;
    //cout << "\n smth = " << smth;

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


