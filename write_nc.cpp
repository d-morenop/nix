

// NETCDF
// https://www.unidata.ucar.edu/software/netcdf/
// https://www.unidata.ucar.edu/software/netcdf/docs/pres_temp_4D_wr_8c-example.html#a9


// NETCDF PARAMETERS.
/* This is the name of the data file we will create. */
#define FILE_NAME "output/test/years/eps.1.0e-10.taub.min.0.0e3_dhds.0.0.nc"
 
/* We are writing 1D data, n grid points*/
#define NDIMS 2
#define NDIMS_0 1
#define NDIMS_Z 3
#define X_NAME "x"
#define TIME_NAME "time"
#define Z_NAME "z"
 
/* Variable names. */
#define H_NAME "H"
#define U1_NAME "u"
#define U2_NAME "du_dx"
#define VISC_NAME "visc"
#define S_NAME "S"
#define TAU_NAME "tau_b"
#define BETA_NAME "beta"
#define TAUD_NAME "tau_d"
#define LMBD_NAME "lambda"
#define B_NAME "b"
#define L_NAME "L"
#define DT_NAME "dt"
#define C_PIC_NAME "c_picard"
#define MU_NAME "mu"
#define OMEGA_NAME "omega"
#define T_NAME "t"
#define U2_BC_NAME "dudx_bc"
#define U2_DIF_NAME "BC_error"
#define PICARD_ERROR_NAME "picard_error"
#define THETA_NAME "theta"
#define C_BED_NAME "C_bed"
#define U2_DIF_VEC_NAME "u2_dif_vec"
#define U2_0_VEC_NAME "u2_0_vec"
#define UNITS "units"
 
/* For the units attributes. */
#define H_UNITS "m"
#define X_UNITS "m"
#define Z_UNITS "m"
#define U1_UNITS "m/yr"
#define U2_UNITS "1/yr"
#define VISC_UNITS "Pa s"
#define S_UNITS "mm/day"
#define TAU_UNITS "Pa"
#define BETA_UNITS "Pa yr / m"
#define TAUD_UNITS "Pa"
#define LMBD_UNITS "dimensionless"
#define B_UNITS "m/km"
#define L_UNITS "km"
#define DT_UNITS "yr"
#define C_PIC_UNITS "dimensionless"
#define MU_UNITS "dimensionless"
#define OMEGA_UNITS "dimensionless"
#define T_UNITS "yr"
#define U2_BC_UNITS "1/yr"
#define U2_DIF_UNITS "1/yr"
#define PICARD_ERROR_UNITS "1/yr"
#define THETA_UNITS "K"
#define C_BED_UNITS "Pa m^-1/3 yr^1/3"
#define U2_DIF_VEC_UNITS "1/yr"
#define U2_0_VEC_UNITS "1/yr"

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); return 2;}

/* Error handling. */
int retval;

////////////////////////////////////////////////////
////////////////////////////////////////////////////
// SOLUTION SAVED IN .NC FILE: flow_line.nc
/* IDs for the netCDF file, dimensions, and variables. */
int ncid, x_dimid, z_dimid, time_dimid;
int x_varid, z_varid, u1_varid, u2_varid, H_varid, visc_varid, s_varid, \
    tau_varid, beta_varid, lmbd_varid, taud_varid, b_varid, L_varid, dt_varid, \
    c_pic_varid, t_varid, mu_varid, omega_varid, u2_bc_varid, u2_dif_varid, \
    picard_error_varid, u2_0_vec_varid, u2_dif_vec_varid, theta_varid, C_bed_varid;
int dimids[NDIMS];

// For the 0D plots (append the current length of domain L)
int dimids_0[NDIMS_0];

// For the theta plots: theta(x,z,t).
int dimids_z[NDIMS_Z];

    
/* The start and cnt arrays will tell the netCDF library where to
    write our data. */
size_t start[NDIMS], cnt[NDIMS];
size_t start_0[NDIMS_0], cnt_0[NDIMS_0];
size_t start_z[NDIMS_Z], cnt_z[NDIMS_Z];
    

int f_nc(int N, int N_Z)
{
    /* These program variables hold one spatial variable x. */
    double xs[N];    
    double xz[N_Z];
        
    /* Create the file. */
    if ((retval = nc_create(FILE_NAME, NC_CLOBBER, &ncid)))
        ERR(retval);
        
    /* Define the dimensions. The record dimension is defined to have
    * unlimited length - it can grow as needed. Here it is
    * the time dimension.*/
    if ((retval = nc_def_dim(ncid, X_NAME, N, &x_dimid)))
        ERR(retval);
    if ((retval = nc_def_dim(ncid, Z_NAME, N_Z, &z_dimid)))
        ERR(retval);
    if ((retval = nc_def_dim(ncid, TIME_NAME, NC_UNLIMITED, &time_dimid)))
        ERR(retval);

        
    /* Define the coordinate variables. We will only define coordinate
        variables for x and z. Ordinarily we would need to provide
        an array of dimension IDs for each variable's dimensions, but
        since coordinate variables only have one dimension, we can
        simply provide the address of that dimension ID (&x_dimid). */
    if ((retval = nc_def_var(ncid, X_NAME, NC_DOUBLE, 1, &x_dimid,
                    &x_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, Z_NAME, NC_DOUBLE, 1, &z_dimid,
                    &z_varid)))
        ERR(retval);
        
    /* Assign units attributes to coordinate variables. */
    if ((retval = nc_put_att_text(ncid, x_varid, UNITS,
                    strlen(X_UNITS), X_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, z_varid, UNITS,
                    strlen(Z_UNITS), Z_UNITS)))
        ERR(retval);
        
    /* The dimids array is used to pass the dimids of the dimensions of
        the netCDF variables. In C, the unlimited dimension must come 
        first on the list of dimids. */
    dimids[0] = time_dimid;
    dimids[1] = x_dimid;

    dimids_0[0] = time_dimid;

    dimids_z[0] = time_dimid;
    dimids_z[1] = x_dimid;
    dimids_z[1] = z_dimid;
        
    /* Define the netCDF variables */
    if ((retval = nc_def_var(ncid, U1_NAME, NC_DOUBLE, NDIMS,
                    dimids, &x_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U2_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u2_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, H_NAME, NC_DOUBLE, NDIMS,
                    dimids, &H_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, VISC_NAME, NC_DOUBLE, NDIMS,
                    dimids, &visc_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, S_NAME, NC_DOUBLE, NDIMS,
                    dimids, &s_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, TAU_NAME, NC_DOUBLE, NDIMS,
                    dimids, &tau_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, BETA_NAME, NC_DOUBLE, NDIMS,
                    dimids, &beta_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, LMBD_NAME, NC_DOUBLE, NDIMS,
                    dimids, &lmbd_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, TAUD_NAME, NC_DOUBLE, NDIMS,
                    dimids, &taud_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, B_NAME, NC_DOUBLE, NDIMS,
                    dimids, &b_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, C_BED_NAME, NC_DOUBLE, NDIMS,
                    dimids, &C_bed_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U2_DIF_VEC_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u2_dif_vec_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U2_0_VEC_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u2_0_vec_varid)))
        ERR(retval);

    if ((retval = nc_def_var(ncid, L_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &L_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, DT_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &dt_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, C_PIC_NAME, NC_INT, NDIMS_0,
                    dimids_0, &c_pic_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, MU_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &mu_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, OMEGA_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &omega_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, T_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &t_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U2_BC_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &u2_bc_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U2_DIF_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &u2_dif_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, PICARD_ERROR_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &picard_error_varid)))
        ERR(retval);

    if ((retval = nc_def_var(ncid, THETA_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &theta_varid)))
        ERR(retval);

    /* Assign units attributes to the netCDF variables. */
    if ((retval = nc_put_att_text(ncid, x_varid, UNITS,
                    strlen(U1_UNITS), U1_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u2_varid, UNITS,
                    strlen(U2_UNITS), U2_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, H_varid, UNITS,
                    strlen(H_UNITS), H_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, visc_varid, UNITS,
                    strlen(VISC_UNITS), VISC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, s_varid, UNITS,
                    strlen(S_UNITS), S_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, tau_varid, UNITS,
                    strlen(TAU_UNITS), TAU_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, beta_varid, UNITS,
                    strlen(BETA_UNITS), BETA_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, lmbd_varid, UNITS,
                    strlen(LMBD_UNITS), LMBD_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, taud_varid, UNITS,
                    strlen(TAUD_UNITS), TAUD_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, b_varid, UNITS,
                    strlen(B_UNITS), B_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, C_bed_varid, UNITS,
                    strlen(C_BED_UNITS), C_BED_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u2_dif_vec_varid, UNITS,
                    strlen(U2_DIF_VEC_UNITS), U2_DIF_VEC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u2_0_vec_varid, UNITS,
                    strlen(U2_0_VEC_UNITS), U2_0_VEC_UNITS)))
        ERR(retval);

    if ((retval = nc_put_att_text(ncid, L_varid, UNITS,
                    strlen(L_UNITS), L_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, dt_varid, UNITS,
                    strlen(DT_UNITS), DT_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, c_pic_varid, UNITS,
                    strlen(C_PIC_UNITS), C_PIC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, mu_varid, UNITS,
                    strlen(MU_UNITS), MU_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, omega_varid, UNITS,
                    strlen(OMEGA_UNITS), OMEGA_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, t_varid, UNITS,
                    strlen(T_UNITS), T_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u2_bc_varid, UNITS,
                    strlen(U2_BC_UNITS), U2_BC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u2_dif_varid, UNITS,
                    strlen(U2_DIF_UNITS), U2_DIF_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, picard_error_varid, UNITS,
                    strlen(PICARD_ERROR_UNITS), PICARD_ERROR_UNITS)))
        ERR(retval);

    if ((retval = nc_put_att_text(ncid, theta_varid, UNITS,
                    strlen(THETA_UNITS), THETA_UNITS)))
        ERR(retval);
        
    /* End define mode. */
    if ((retval = nc_enddef(ncid)))
        ERR(retval);
        
    /* Write the coordinate variable data. This will put the x and z
     of our data grid into the netCDF file. */
    if ((retval = nc_put_var_double(ncid, x_varid, &xs[0])))
        ERR(retval);
    if ((retval = nc_put_var_double(ncid, z_varid, &xz[0])))
        ERR(retval);
        
    /* These settings tell netcdf to write one timestep of data. (The
        setting of start[0] inside the loop below tells netCDF which
        timestep to write.) */
    cnt[0]   = 1;
    cnt[1]   = N;
    start[1] = 0;

    cnt_0[0]   = 1;
    start_0[0] = 0;

    cnt_z[0]   = 1;
    cnt_z[1]   = N_Z;
    cnt_z[2]   = N;
    start_z[1] = 0;
    start_z[2] = 0;

    return N;
}
