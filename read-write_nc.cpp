
// NETCDF
// https://www.unidata.ucar.edu/software/netcdf/
// https://www.unidata.ucar.edu/software/netcdf/docs/pres_temp_4D_wr_8c-example.html#a9


// NETCDF PARAMETERS.
/* This is the name of the data file we will create. */
//#define FILE_NAME "output/mismip/exp3/exp3_n.250/exp3_n.250.nc"
#define FILE_NAME "/home/dmoreno/flowline/ub_new_test.F.all.opt/flowline.nc"
#define FILE_NAME_READ "/home/dmoreno/c++/flowline/output/glacier_ews/noise_sigm_ocn.12.0.nc"



// path: /home/dmoren07/c++/flowline/output/mismip/exp3/exp3_n.250

/* We are writing 1D data, n grid points*/
#define NDIMS 2
#define NDIMS_0 1
#define NDIMS_Z 3
#define X_NAME "x"
#define TIME_NAME "time"
#define Z_NAME "z"
 
/* Variable names. */
#define H_NAME "H"
#define U_BAR_NAME "u_bar"
#define U_B_NAME "ub"
#define U_X_BAR_NAME "u_x"
#define U_X_DIVA_NAME "u_x_diva"
#define U_Z_NAME "u_z"
#define U_NAME "u"
#define VISC_NAME "visc"
#define VISC_BAR_NAME "visc_bar"
#define A_NAME "A"
#define S_NAME "S"
#define TAU_NAME "tau_b"
#define BETA_NAME "beta"
#define TAUD_NAME "tau_d"
#define LMBD_NAME "lambda"
#define B_NAME "b"
#define L_NAME "L"
#define DL_DT_NAME "dL_dt"
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
#define Q_FRIC_NAME "Q_fric"
#define U2_DIF_VEC_NAME "u2_dif_vec"
#define U2_0_VEC_NAME "u2_0_vec"
#define M_STOCH_NAME "m_stoch"
#define SMB_STOCH_NAME "smb_stoch"
#define F_1_NAME "F_1"
#define F_2_NAME "F_2"
#define UNITS "units"
 
/* For the units attributes. */
#define H_UNITS "m"
#define X_UNITS "m"
#define Z_UNITS "m"
#define T_UNITS "yr"
#define U_BAR_UNITS "m/yr"
#define U_B_UNITS "m/yr"
#define U_X_UNITS "1/yr"
#define U_X_DIVA_UNITS "1/yr"
#define U_Z_UNITS "1/yr"
#define U_UNITS "m/yr"
#define VISC_UNITS "Pa s"
#define VISC_BAR_UNITS "Pa s"
#define A_UNITS "Pa^-3 s^-1"
#define S_UNITS "mm/day"
#define TAU_UNITS "Pa"
#define BETA_UNITS "Pa yr / m"
#define TAUD_UNITS "Pa"
#define LMBD_UNITS "dimensionless"
#define B_UNITS "m/km"
#define L_UNITS "km"
#define DL_DT_UNITS "km/yr"
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
#define Q_FRIC_UNITS "W/m^2"
#define U2_DIF_VEC_UNITS "1/yr"
#define U2_0_VEC_UNITS "1/yr"
#define M_STOCH_UNITS "m/yr"
#define SMB_STOCH_UNITS "m/yr"
#define F_1_UNITS "m / (Pa yr)"
#define F_2_UNITS "m / (Pa yr)"

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
int x_varid, z_varid, u_bar_varid, ub_varid, u_x_varid, u_x_diva_varid, u_z_varid, u_varid,H_varid, \
    visc_varid, visc_bar_varid, a_varid, s_varid, \
    tau_varid, beta_varid, lmbd_varid, taud_varid, b_varid, L_varid, dL_dt_varid, dt_varid, \
    c_pic_varid, t_varid, mu_varid, omega_varid, u2_bc_varid, u2_dif_varid, \
    picard_error_varid, u2_0_vec_varid, u2_dif_vec_varid, theta_varid, C_bed_varid, Q_fric_varid, \
    F_1_varid, F_2_varid, m_stoch_varid, smb_stoch_varid;
int dimids[NDIMS];

// For the 0D plots (append the current length of domain L)
int dimids_0[NDIMS_0];

// For the theta, visc plots: theta(x,z,t), visc(x,z,t).
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
    if ((retval = nc_def_var(ncid, U_BAR_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u_bar_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U_B_NAME, NC_DOUBLE, NDIMS,
                    dimids, &ub_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U_X_BAR_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u_x_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, H_NAME, NC_DOUBLE, NDIMS,
                    dimids, &H_varid)))
        ERR(retval);
    

    if ((retval = nc_def_var(ncid, VISC_BAR_NAME, NC_DOUBLE, NDIMS,
                    dimids, &visc_bar_varid)))
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
    if ((retval = nc_def_var(ncid, Q_FRIC_NAME, NC_DOUBLE, NDIMS,
                    dimids, &Q_fric_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U2_DIF_VEC_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u2_dif_vec_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U2_0_VEC_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u2_0_vec_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, F_1_NAME, NC_DOUBLE, NDIMS,
                    dimids, &F_1_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, F_2_NAME, NC_DOUBLE, NDIMS,
                    dimids, &F_2_varid)))
        ERR(retval);


    if ((retval = nc_def_var(ncid, L_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &L_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, DL_DT_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &dL_dt_varid)))
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
    if ((retval = nc_def_var(ncid, A_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &a_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, M_STOCH_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &m_stoch_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, SMB_STOCH_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &smb_stoch_varid)))
        ERR(retval);

    if ((retval = nc_def_var(ncid, THETA_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &theta_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, VISC_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &visc_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U_Z_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &u_z_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U_X_DIVA_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &u_x_diva_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &u_varid)))
        ERR(retval);

    /* Assign units attributes to the netCDF variables. */
    if ((retval = nc_put_att_text(ncid, u_bar_varid, UNITS,
                    strlen(U_BAR_UNITS), U_BAR_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, ub_varid, UNITS,
                    strlen(U_B_UNITS), U_B_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u_x_varid, UNITS,
                    strlen(U_X_UNITS), U_X_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, H_varid, UNITS,
                    strlen(H_UNITS), H_UNITS)))
        ERR(retval);
    
    if ((retval = nc_put_att_text(ncid, visc_bar_varid, UNITS,
                    strlen(VISC_BAR_UNITS), VISC_BAR_UNITS)))
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
    if ((retval = nc_put_att_text(ncid, Q_fric_varid, UNITS,
                    strlen(Q_FRIC_UNITS), Q_FRIC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u2_dif_vec_varid, UNITS,
                    strlen(U2_DIF_VEC_UNITS), U2_DIF_VEC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u2_0_vec_varid, UNITS,
                    strlen(U2_0_VEC_UNITS), U2_0_VEC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, F_1_varid, UNITS,
                    strlen(F_1_UNITS), F_1_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, F_2_varid, UNITS,
                    strlen(F_2_UNITS), F_2_UNITS)))
        ERR(retval);

    if ((retval = nc_put_att_text(ncid, L_varid, UNITS,
                    strlen(L_UNITS), L_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, dL_dt_varid, UNITS,
                    strlen(DL_DT_UNITS), DL_DT_UNITS)))
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
    if ((retval = nc_put_att_text(ncid, a_varid, UNITS,
                    strlen(A_UNITS), A_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, m_stoch_varid, UNITS,
                    strlen(M_STOCH_UNITS), M_STOCH_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, smb_stoch_varid, UNITS,
                    strlen(SMB_STOCH_UNITS), SMB_STOCH_UNITS)))
        ERR(retval);

    if ((retval = nc_put_att_text(ncid, theta_varid, UNITS,
                    strlen(THETA_UNITS), THETA_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, visc_varid, UNITS,
                    strlen(VISC_UNITS), VISC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u_z_varid, UNITS,
                    strlen(U_Z_UNITS), U_Z_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u_x_diva_varid, UNITS,
                    strlen(U_X_DIVA_UNITS), U_X_DIVA_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u_varid, UNITS,
                    strlen(U_UNITS), U_UNITS)))
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


int f_write(int c, ArrayXd u_bar, ArrayXd ub, ArrayXd u_x, ArrayXd H, ArrayXd visc_bar, \
            ArrayXd S, ArrayXd tau_b, ArrayXd beta, ArrayXd tau_d, ArrayXd bed, \
            ArrayXd C_bed, ArrayXd Q_fric, ArrayXd u2_dif_vec, ArrayXd u2_0_vec, \
            double L, double t, double u2_bc, double u2_dif, double error, \
            double dt, int c_picard, double mu, double omega, ArrayXXd theta, \
            ArrayXXd visc, ArrayXXd u_z, ArrayXXd u_x_diva, ArrayXXd u, double A, double dL_dt, \
            ArrayXd F_1, ArrayXd F_2, double m_stoch, double smb_stoch)
{
    start[0]   = c;
    start_0[0] = c;
    start_z[0] = c;

    // 2D variables.
    if ((retval = nc_put_vara_double(ncid, u_bar_varid, start, cnt, &u_bar(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, ub_varid, start, cnt, &ub(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_x_varid, start, cnt, &u_x(0))))
    ERR(retval);
    
    if ((retval = nc_put_vara_double(ncid, H_varid, start, cnt, &H(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, visc_bar_varid, start, cnt, &visc_bar(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, s_varid, start, cnt, &S(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, tau_varid, start, cnt, &tau_b(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, beta_varid, start, cnt, &beta(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, taud_varid, start, cnt, &tau_d(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, b_varid, start, cnt, &bed(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, C_bed_varid, start, cnt, &C_bed(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, Q_fric_varid, start, cnt, &Q_fric(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u2_dif_vec_varid, start, cnt, &u2_dif_vec(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u2_0_vec_varid, start, cnt, &u2_0_vec(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, F_1_varid, start, cnt, &F_1(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, F_2_varid, start, cnt, &F_2(0))))
    ERR(retval);

    // 1D variables.
    if ((retval = nc_put_vara_double(ncid, L_varid, start_0, cnt_0, &L)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, dL_dt_varid, start_0, cnt_0, &dL_dt)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, t_varid, start_0, cnt_0, &t)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u2_bc_varid, start_0, cnt_0, &u2_bc)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u2_dif_varid, start_0, cnt_0, &u2_dif)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, picard_error_varid, start_0, cnt_0, &error)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, dt_varid, start_0, cnt_0, &dt)))
    ERR(retval);
    if ((retval = nc_put_vara_int(ncid, c_pic_varid, start_0, cnt_0, &c_picard)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, mu_varid, start_0, cnt_0, &mu)))
    ERR(retval); // currently mu
    if ((retval = nc_put_vara_double(ncid, omega_varid, start_0, cnt_0, &omega)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, a_varid, start_0, cnt_0, &A)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, m_stoch_varid, start_0, cnt_0, &m_stoch)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, smb_stoch_varid, start_0, cnt_0, &smb_stoch)))
    ERR(retval);

    // 3D variables.
    if ((retval = nc_put_vara_double(ncid, theta_varid, start_z, cnt_z, &theta(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, visc_varid, start_z, cnt_z, &visc(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_z_varid, start_z, cnt_z, &u_z(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_x_diva_varid, start_z, cnt_z, &u_x_diva(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_varid, start_z, cnt_z, &u(0,0))))
    ERR(retval);


    return c;
}


// Function to read nc file and load it to vairabe.
ArrayXXd f_nc_read(int N)
{
    /* This will be the netCDF ID for the file and data variable. */
    int ncid, varid_ocn, varid_smb;

    // Number of variables.
    int n_var = 2;

    // Allocate variables.
    ArrayXXd data(n_var,N);
    ArrayXd noise_ocn(N);
    ArrayXd noise_smb(N);

    /* Loop indexes, and error handling. */
    int retval;

    // Avoid int function.
    nc_open(FILE_NAME_READ, NC_NOWRITE, &ncid);

    /* Get the varid of the data variable, based on its name. */
    nc_inq_varid(ncid, "Noise_ocn", &varid_ocn);
    nc_inq_varid(ncid, "Noise_smb", &varid_smb);

    /* Read the data. */
    nc_get_var_double(ncid, varid_ocn, &noise_ocn(0));
    nc_get_var_double(ncid, varid_smb, &noise_smb(0));

    // Allocate in output variable.
    data.row(0) = noise_ocn;
    data.row(1) = noise_smb;

    //nc_get_var_double(ncid, varid_ocn, &data_in(0,0));
    //nc_get_var_double(ncid, varid_smb, &data_in(1,0));

    /* Close the file, freeing all resources. */
    nc_close(ncid);

    printf("*** SUCCESS reading example file %s!\n", FILE_NAME_READ);
    
    return data;
}