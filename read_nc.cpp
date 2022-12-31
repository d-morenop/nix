

// NETCDF
// https://www.unidata.ucar.edu/software/netcdf/
// https://www.unidata.ucar.edu/software/netcdf/docs/pres_temp_4D_wr_8c-example.html#a9

/* This is the name of the data file we will read. */
#define FILE_NAME_READ "/home/dmoreno/c++/flowline/output/glacier_ews/test/noise.nc"


/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
//#define ERR_READ(e) {printf("Error: %s\n", nc_strerror(e)); return 2;}

int f_nc_read(int N)
{
    /* This will be the netCDF ID for the file and data variable. */
    int ncid, varid;

    //ArrayXd noise_ocn(N);
    double noise_ocn(N);

    /* Loop indexes, and error handling. */
    int retval;

    /* Open the file. NC_NOWRITE tells netCDF we want read-only access
    * to the file.*/
    if ((retval = nc_open(FILE_NAME_READ, NC_NOWRITE, &ncid)))
        ERR(retval);

    /* Get the varid of the data variable, based on its name. */
    if ((retval = nc_inq_varid(ncid, "Noise_ocn", &varid)))
        ERR(retval);

    /* Read the data. */
    if ((retval = nc_get_var_double(ncid, varid, &noise_ocn(0))))
        ERR(retval);

    /* Close the file, freeing all resources. */
    if ((retval = nc_close(ncid)))
        ERR(retval);

    printf("*** SUCCESS reading example file %s!\n", FILE_NAME);
    return 0;

}