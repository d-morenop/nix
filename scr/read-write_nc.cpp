
// NETCDF
// https://www.unidata.ucar.edu/software/netcdf/
// https://www.unidata.ucar.edu/software/netcdf/docs/pres_temp_4D_wr_8c-example.html#a9


// NETCDF PARAMETERS.
/* This is the name of the data file we will create. */

//#define FILE_NAME "/home/dmoreno/nix/test_modules/nix.nc"
//#define FILE_NAME_HR "/home/dmoreno/nix/blatter-pattyn/test/nix_hr.nc"
//#define FILE_NAME_READ "/home/dmoreno/nix/data/noise_sigm_ocn.12.0.nc"



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
#define U_BAR_X_NAME "u_bar_x"
#define U_X_NAME "u_x"
#define U_Z_NAME "u_z"
#define U_NAME "u"
#define W_NAME "w"
#define VISC_NAME "visc"
#define VISC_BAR_NAME "visc_bar"
#define A_NAME "A"
#define A_THETA_NAME "A_theta"
#define S_NAME "S"
#define TAU_NAME "tau_b"
#define BETA_NAME "beta"
#define TAUD_NAME "tau_d"
#define LMBD_NAME "lmbd"
#define B_NAME "b"
#define L_NAME "L"
#define DL_DT_NAME "dL_dt"
#define DT_NAME "dt"
#define C_PIC_NAME "c_picard"
#define MU_NAME "mu"
#define OMEGA_NAME "omega"
#define T_NAME "t"
#define SPEED_NAME "speed"
#define U2_BC_NAME "dudx_bc"
#define U2_DIF_NAME "BC_error"
#define PICARD_ERROR_NAME "picard_error"
#define THETA_NAME "theta"
#define T_OCE_DET_NAME "T_oce_det"
#define T_OCE_STOCH_NAME "T_oce_stoch"
#define T_AIR_NAME "T_air"
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
#define U_BAR_X_UNITS "1/yr"
#define U_X_UNITS "1/yr"
#define U_Z_UNITS "1/yr"
#define U_UNITS "m/yr"
#define W_UNITS "m/yr"
#define VISC_UNITS "Pa s"
#define VISC_BAR_UNITS "Pa s"
#define A_UNITS "Pa^-3 s^-1"
#define A_THETA_UNITS "Pa^-3 s^-1"
#define S_UNITS "mm/day"
#define TAU_UNITS "Pa"
#define BETA_UNITS "Pa yr / m"
#define TAUD_UNITS "Pa"
#define LMBD_UNITS "Pa / m"
#define B_UNITS "m/km"
#define L_UNITS "km"
#define DL_DT_UNITS "km/yr"
#define DT_UNITS "yr"
#define C_PIC_UNITS "dimensionless"
#define MU_UNITS "dimensionless"
#define OMEGA_UNITS "dimensionless"
#define T_UNITS "yr"
#define SPEED_UNITS "kyr/hr"
#define U2_BC_UNITS "1/yr"
#define U2_DIF_UNITS "1/yr"
#define PICARD_ERROR_UNITS "1/yr"
#define THETA_UNITS "K"
#define T_OCE_UNITS "K"
#define T_AIR_UNITS "K"
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
int x_varid, z_varid, u_bar_varid, ub_varid, u_bar_x_varid, u_x_varid, u_z_varid, u_varid, w_varid, 
    H_varid, visc_varid, visc_bar_varid, a_varid, a_theta_varid, s_varid, \
    tau_varid, beta_varid, lmbd_varid, taud_varid, b_varid, L_varid, dL_dt_varid, dt_varid, \
    c_pic_varid, t_varid, speed_varid, mu_varid, omega_varid, u2_bc_varid, u2_dif_varid, \
    picard_error_varid, u2_0_vec_varid, u2_dif_vec_varid, theta_varid, T_oce_det_varid, T_oce_stoch_varid, \
    T_air_varid, C_bed_varid, Q_fric_varid, F_1_varid, F_2_varid, m_stoch_varid, smb_stoch_varid;

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

////////////////////////////////////////////////////
////////////////////////////////////////////////////



////////////////////////////////////////////////////
////////////////////////////////////////////////////

// IDs for high resolution.
int ncid_hr, x_dimid_hr, z_dimid_hr, time_dimid_hr;
int x_varid_hr, z_varid_hr, u_bar_varid_hr, H_varid_hr, L_varid_hr, t_varid_hr, a_varid_hr, \
    u2_bc_varid_hr, u2_dif_varid_hr, picard_error_varid_hr, dt_varid_hr, c_pic_varid_hr, mu_varid_hr, \
    omega_varid_hr, A_varid_hr, dL_dt_varid_hr, m_stoch_varid_hr, smb_stoch_varid_hr, T_air_varid_hr, \
    T_oce_det_varid_hr, T_oce_stoch_varid_hr;

int dimids_hr[NDIMS];

// For the 0D plots (append the current length of domain L)
int dimids_0_hr[NDIMS_0];

// For the theta, visc plots: theta(x,z,t), visc(x,z,t).
int dimids_z_hr[NDIMS_Z];

/* The start and cnt arrays will tell the netCDF library where to
    write our data. */
size_t start_hr[NDIMS], cnt_hr[NDIMS];
size_t start_0_hr[NDIMS_0], cnt_0_hr[NDIMS_0];
size_t start_z_hr[NDIMS_Z], cnt_z_hr[NDIMS_Z];

////////////////////////////////////////////////////
////////////////////////////////////////////////////
    

    

int f_nc(int N, int N_Z, string path)
{
    /* File names. We need a const char to be passed to nc_create().
      The asterisk (*) is used to declare a pointer. In the context of 
      const char*, it indicates that the variable is a pointer to a constant 
      character (i.e., a C-style string). If you want to work with strings or
      character arrays, you should use the * to declare a pointer (for a 
      single character you don't need it: "A"). */
    string full_path = path+"nix.nc";
    const char* FILE_NAME = full_path.c_str();
    

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
        variables for x and z. Ordinarily, we would need to provide
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
    if ((retval = nc_def_var(ncid, U_BAR_X_NAME, NC_DOUBLE, NDIMS,
                    dimids, &u_bar_x_varid)))
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
    if ((retval = nc_def_var(ncid, SPEED_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &speed_varid)))
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
    if ((retval = nc_def_var(ncid, T_OCE_DET_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &T_oce_det_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, T_OCE_STOCH_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &T_oce_stoch_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, T_AIR_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0, &T_air_varid)))
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
    if ((retval = nc_def_var(ncid, U_X_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &u_x_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, U_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &u_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, W_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &w_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, LMBD_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &lmbd_varid)))
        ERR(retval);
    if ((retval = nc_def_var(ncid, A_THETA_NAME, NC_DOUBLE, NDIMS_Z,
                    dimids_z, &a_theta_varid)))
        ERR(retval);

    /* Assign units attributes to the netCDF variables. */
    if ((retval = nc_put_att_text(ncid, u_bar_varid, UNITS,
                    strlen(U_BAR_UNITS), U_BAR_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, ub_varid, UNITS,
                    strlen(U_B_UNITS), U_B_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u_bar_x_varid, UNITS,
                    strlen(U_BAR_X_UNITS), U_BAR_X_UNITS)))
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
    if ((retval = nc_put_att_text(ncid, speed_varid, UNITS,
                    strlen(SPEED_UNITS), SPEED_UNITS)))
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
    if ((retval = nc_put_att_text(ncid, T_oce_det_varid, UNITS,
                    strlen(T_OCE_UNITS), T_OCE_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, T_oce_stoch_varid, UNITS,
                    strlen(T_OCE_UNITS), T_OCE_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, T_air_varid, UNITS,
                    strlen(T_AIR_UNITS), T_AIR_UNITS)))
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
    if ((retval = nc_put_att_text(ncid, u_x_varid, UNITS,
                    strlen(U_X_UNITS), U_X_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, u_varid, UNITS,
                    strlen(U_UNITS), U_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, w_varid, UNITS,
                    strlen(W_UNITS), W_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid, lmbd_varid, UNITS,
                    strlen(LMBD_UNITS), LMBD_UNITS)))
        ERR(retval);
     if ((retval = nc_put_att_text(ncid, a_theta_varid, UNITS,
                    strlen(A_THETA_UNITS), A_THETA_UNITS)))
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
    //return ncid;
}



// Create another nc file just for high resolution output.
int f_nc_hr(int N, int N_Z, string path)
{
    // Create file names.
    string full_path = path+"nix_hr.nc";
    const char* FILE_NAME_HR = full_path.c_str();

    /* These program variables hold one spatial variable x. */
    double xs[N];    
    double xz[N_Z];
        
    /* Create the file. */
    if ((retval = nc_create(FILE_NAME_HR, NC_CLOBBER, &ncid_hr)))
        ERR(retval);
        
    /* Define the dimensions. The record dimension is defined to have
    * unlimited length - it can grow as needed. Here it is
    * the time dimension.*/
    if ((retval = nc_def_dim(ncid_hr, X_NAME, N, &x_dimid_hr)))
        ERR(retval);
    if ((retval = nc_def_dim(ncid_hr, Z_NAME, N_Z, &z_dimid_hr)))
        ERR(retval);
    if ((retval = nc_def_dim(ncid_hr, TIME_NAME, NC_UNLIMITED, &time_dimid_hr)))
        ERR(retval);
        
    /* Define the coordinate variables. We will only define coordinate
        variables for x and z. Ordinarily we would need to provide
        an array of dimension IDs for each variable's dimensions, but
        since coordinate variables only have one dimension, we can
        simply provide the address of that dimension ID (&x_dimid). */
    if ((retval = nc_def_var(ncid_hr, X_NAME, NC_DOUBLE, 1, &x_dimid_hr,
                    &x_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, Z_NAME, NC_DOUBLE, 1, &z_dimid_hr,
                    &z_varid_hr)))
        ERR(retval);
        
    /* Assign units attributes to coordinate variables. */
    if ((retval = nc_put_att_text(ncid_hr, x_varid_hr, UNITS,
                    strlen(X_UNITS), X_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, z_varid_hr, UNITS,
                    strlen(Z_UNITS), Z_UNITS)))
        ERR(retval);
        
    /* The dimids array is used to pass the dimids of the dimensions of
        the netCDF variables. In C, the unlimited dimension must come 
        first on the list of dimids. */
    dimids_hr[0] = time_dimid_hr;
    dimids_hr[1] = x_dimid_hr;

    dimids_0_hr[0] = time_dimid_hr;

    dimids_z_hr[0] = time_dimid_hr;
    dimids_z_hr[1] = x_dimid_hr;
    dimids_z_hr[1] = z_dimid_hr;
        
    // DEFINE NETCDF VARIABLES.
    // 1D (2D variables evaluated at the grounding line).
    if ((retval = nc_def_var(ncid_hr, U_BAR_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &u_bar_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, H_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &H_varid_hr)))
        ERR(retval);
    
    // 1D variables.
    if ((retval = nc_def_var(ncid_hr, L_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &L_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, DL_DT_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &dL_dt_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, DT_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &dt_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, C_PIC_NAME, NC_INT, NDIMS_0,
                    dimids_0_hr, &c_pic_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, MU_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &mu_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, OMEGA_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &omega_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, T_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &t_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, U2_BC_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &u2_bc_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, U2_DIF_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &u2_dif_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, PICARD_ERROR_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &picard_error_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, A_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &a_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, M_STOCH_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &m_stoch_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, SMB_STOCH_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &smb_stoch_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, T_AIR_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &T_air_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, T_OCE_DET_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &T_oce_det_varid_hr)))
        ERR(retval);
    if ((retval = nc_def_var(ncid_hr, T_OCE_STOCH_NAME, NC_DOUBLE, NDIMS_0,
                    dimids_0_hr, &T_oce_stoch_varid_hr)))
        ERR(retval);
    


    /* Assign units attributes to the netCDF variables. */
    if ((retval = nc_put_att_text(ncid_hr, u_bar_varid_hr, UNITS,
                    strlen(U_BAR_UNITS), U_BAR_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, H_varid_hr, UNITS,
                    strlen(H_UNITS), H_UNITS)))
        ERR(retval);

    if ((retval = nc_put_att_text(ncid_hr, L_varid_hr, UNITS,
                    strlen(L_UNITS), L_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, dL_dt_varid_hr, UNITS,
                    strlen(DL_DT_UNITS), DL_DT_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, dt_varid_hr, UNITS,
                    strlen(DT_UNITS), DT_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, c_pic_varid_hr, UNITS,
                    strlen(C_PIC_UNITS), C_PIC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, mu_varid_hr, UNITS,
                    strlen(MU_UNITS), MU_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, omega_varid_hr, UNITS,
                    strlen(OMEGA_UNITS), OMEGA_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, t_varid_hr, UNITS,
                    strlen(T_UNITS), T_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, u2_bc_varid_hr, UNITS,
                    strlen(U2_BC_UNITS), U2_BC_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, u2_dif_varid_hr, UNITS,
                    strlen(U2_DIF_UNITS), U2_DIF_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, picard_error_varid_hr, UNITS,
                    strlen(PICARD_ERROR_UNITS), PICARD_ERROR_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, a_varid_hr, UNITS,
                    strlen(A_UNITS), A_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, m_stoch_varid_hr, UNITS,
                    strlen(M_STOCH_UNITS), M_STOCH_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, smb_stoch_varid_hr, UNITS,
                    strlen(SMB_STOCH_UNITS), SMB_STOCH_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, T_air_varid_hr, UNITS,
                    strlen(T_AIR_UNITS), T_AIR_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, T_oce_det_varid_hr, UNITS,
                    strlen(T_OCE_UNITS), T_OCE_UNITS)))
        ERR(retval);
    if ((retval = nc_put_att_text(ncid_hr, T_oce_stoch_varid_hr, UNITS,
                    strlen(T_OCE_UNITS), T_OCE_UNITS)))
        ERR(retval);
        
    /* End define mode. */
    if ((retval = nc_enddef(ncid_hr)))
        ERR(retval);
        
    /* Write the coordinate variable data. This will put the x and z
     of our data grid into the netCDF file. */
    if ((retval = nc_put_var_double(ncid_hr, x_varid_hr, &xs[0])))
        ERR(retval);
    if ((retval = nc_put_var_double(ncid_hr, z_varid_hr, &xz[0])))
        ERR(retval);
        
    /* These settings tell netcdf to write one timestep of data. (The
        setting of start[0] inside the loop below tells netCDF which
        timestep to write.) */
    cnt_hr[0]   = 1;
    cnt_hr[1]   = N;
    start_hr[1] = 0;

    cnt_0_hr[0]   = 1;
    start_0_hr[0] = 0;

    cnt_z_hr[0]   = 1;
    cnt_z_hr[1]   = N_Z;
    cnt_z_hr[2]   = N;
    start_z_hr[1] = 0;
    start_z_hr[2] = 0;

    return N;
}


int f_write(int c, ArrayXd u_bar, ArrayXd ub, ArrayXd u_bar_x, ArrayXd H, ArrayXd visc_bar, \
            ArrayXd S, ArrayXd tau_b, ArrayXd beta, ArrayXd tau_d, ArrayXd bed, \
            ArrayXd C_bed, ArrayXd Q_fric, ArrayXd u2_dif_vec, ArrayXd u2_0_vec, \
            double L, double t, double u2_bc, double u2_dif, double error, \
            double dt, int c_picard, double mu, double omega, ArrayXXd theta, \
            ArrayXXd visc, ArrayXXd u_z, ArrayXXd u_x, ArrayXXd u, ArrayXXd w, double A, double dL_dt, \
            ArrayXd F_1, ArrayXd F_2, double m_stoch, double smb_stoch, ArrayXXd A_theta, double T_oce_det, 
            double T_oce_stoch, double T_air, ArrayXXd lmbd, double speed)
{
    start[0]   = c;
    start_0[0] = c;
    start_z[0] = c;

    // 2D variables.
    if ((retval = nc_put_vara_double(ncid, u_bar_varid, start, cnt, &u_bar(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, ub_varid, start, cnt, &ub(0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_bar_x_varid, start, cnt, &u_bar_x(0))))
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
    if ((retval = nc_put_vara_double(ncid, dt_varid, start_0, cnt_0, &dt)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, picard_error_varid, start_0, cnt_0, &error)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, speed_varid, start_0, cnt_0, &speed)))
    ERR(retval);
    if ((retval = nc_put_vara_int(ncid, c_pic_varid, start_0, cnt_0, &c_picard)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u2_bc_varid, start_0, cnt_0, &u2_bc)))
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
    if ((retval = nc_put_vara_double(ncid, T_oce_det_varid, start_0, cnt_0, &T_oce_det)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, T_oce_stoch_varid, start_0, cnt_0, &T_oce_stoch)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, T_air_varid, start_0, cnt_0, &T_air)))
    ERR(retval);

    // 3D variables.
    if ((retval = nc_put_vara_double(ncid, theta_varid, start_z, cnt_z, &theta(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, visc_varid, start_z, cnt_z, &visc(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_z_varid, start_z, cnt_z, &u_z(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_x_varid, start_z, cnt_z, &u_x(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, u_varid, start_z, cnt_z, &u(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, w_varid, start_z, cnt_z, &w(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, lmbd_varid, start_z, cnt_z, &lmbd(0,0))))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid, a_theta_varid, start_z, cnt_z, &A_theta(0,0))))
    ERR(retval);


    return c;
}



int f_write_hr(int c, double u_bar_L, double H_L, \
               double L, double t, double u2_bc, double u2_dif, double error, \
               double dt, int c_picard, double mu, double omega, double A, double dL_dt, \
               double m_stoch, double smb_stoch, double T_air, double T_oce_det, double T_oce_stoch)
{
    start_hr[0]   = c;
    start_0_hr[0] = c;
    start_z_hr[0] = c;

    // 1D (2D variables evaluated at the grounding line).
    if ((retval = nc_put_vara_double(ncid_hr, u_bar_varid_hr, start_hr, cnt_hr, &u_bar_L)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, H_varid_hr, start_hr, cnt_hr, &H_L)))
    ERR(retval);


    // 1D variables.
    if ((retval = nc_put_vara_double(ncid_hr, L_varid_hr, start_0_hr, cnt_0_hr, &L)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, dL_dt_varid_hr, start_0_hr, cnt_0_hr, &dL_dt)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, t_varid_hr, start_0_hr, cnt_0_hr, &t)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, u2_bc_varid_hr, start_0_hr, cnt_0_hr, &u2_bc)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, u2_dif_varid_hr, start_0_hr, cnt_0_hr, &u2_dif)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, picard_error_varid_hr, start_0_hr, cnt_0_hr, &error)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, dt_varid_hr, start_0_hr, cnt_0_hr, &dt)))
    ERR(retval);
    if ((retval = nc_put_vara_int(ncid_hr, c_pic_varid_hr, start_0_hr, cnt_0_hr, &c_picard)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, mu_varid_hr, start_0_hr, cnt_0_hr, &mu)))
    ERR(retval); // currently mu
    if ((retval = nc_put_vara_double(ncid_hr, omega_varid_hr, start_0_hr, cnt_0_hr, &omega)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, a_varid_hr, start_0_hr, cnt_0_hr, &A)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, m_stoch_varid_hr, start_0_hr, cnt_0_hr, &m_stoch)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, smb_stoch_varid_hr, start_0_hr, cnt_0_hr, &smb_stoch)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, T_air_varid_hr, start_0_hr, cnt_0_hr, &T_air)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, T_oce_det_varid_hr, start_0_hr, cnt_0_hr, &T_oce_det)))
    ERR(retval);
    if ((retval = nc_put_vara_double(ncid_hr, T_oce_stoch_varid_hr, start_0_hr, cnt_0_hr, &T_oce_stoch)))
    ERR(retval);

    return c;
}


// Function to read nc file and load it to vairabe.
ArrayXXd f_nc_read(int N, string path)
{
    // Convert string to const char.
    //string path_new = path;
    const char* FILE_NAME_READ = path.c_str();
    
    /*cout << "\n file_name = " << FILE_NAME_READ;
    cout << "\n path      = " << path;
    cout << "\n path_new  = " << path_new;*/

    /* This will be the netCDF ID for the file and data variable. */
    int ncid, varid_ocn, varid_smb, varid_T_oce;

    // Number of variables.
    int n_var = 3;

    // Allocate variables.
    ArrayXXd data(n_var,N);
    ArrayXd noise_ocn(N);
    ArrayXd noise_smb(N);
    ArrayXd noise_T_oce(N);

    /* Loop indexes, and error handling. */
    int retval;

    // Avoid int function.
    nc_open(FILE_NAME_READ, NC_NOWRITE, &ncid);
    
    
    cout << "\n ncid = " << ncid;
    cout << "\n file_name = " << FILE_NAME_READ;

    /* Get the varid of the data variable, based on its name. */
    nc_inq_varid(ncid, "Noise_ocn", &varid_ocn);
    nc_inq_varid(ncid, "Noise_smb", &varid_smb);
    nc_inq_varid(ncid, "Noise_T_oce", &varid_T_oce);

    /* Read the data. */
    nc_get_var_double(ncid, varid_ocn, &noise_ocn(0));
    nc_get_var_double(ncid, varid_smb, &noise_smb(0));
    nc_get_var_double(ncid, varid_T_oce, &noise_T_oce(0));

    // Allocate in output variable.
    data.row(0) = noise_ocn;
    data.row(1) = noise_smb;
    data.row(2) = noise_T_oce;

    //data << noise_ocn, noise_smb, noise_T_oce;

    //nc_get_var_double(ncid, varid_ocn, &data_in(0,0));
    //nc_get_var_double(ncid, varid_smb, &data_in(1,0));

    /* Close the file, freeing all resources. */
    nc_close(ncid);

    printf("*** SUCCESS reading example file %s!\n", FILE_NAME_READ);
    
    return data;
}