#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:01:22 2022

@author: dmoreno

This scripts creates frontal ablation and SMB time series with varying degrees
and types of persistence. Based on Christian et al. (2022). Assumed that dt = 1 year.

The idea is to save the data in a external file that will be read by the flowline
as a boundary condition. 

Note: flow line time step may be shorter than 1 year (adaptative), we can further 
interpolate as the file is read or simpy take the same values within a given time step. 

Netcdf documentation for python:
http://pyhogs.github.io/intro_netcdf4.html

"""

import sys
import os
import subprocess
import numpy as np
from numpy.random import seed
from numpy.random import rand
import netCDF4 as nc4
from datetime import datetime
import matplotlib.pyplot as plt


# Stochastic noise function.
def stochastic_noise(t, tf, dt, sigm_ocn, sigm_smb, tau_ocn, tau_smb):

    # Defined a function that returns a normalised IFFT.
    def normalised_ifft(x, sigm):
        noise = np.fft.ifft(x) 
        noise = noise - np.mean(noise)
        noise = noise / np.std(noise)
        noise = sigm * noise
        return noise

    #t  = dt * np.linspace(0, N, N)
    df = 1.0 / ( dt * N )
    f0 = 0.5 / dt

    f1 = np.arange(0, f0, df)

    # Auto-correlation at a lag of dt.
    r_ocn = 1.0 - ( dt / tau_ocn )
    r_smb = 1.0 - ( dt / tau_smb )

    # Scales total variance.
    P_0 = 1.0

    # Analytical power spectra given auto-correlation.
    P_ocn = np.sqrt( P_0 / ( 1.0 + r_ocn**2 - 2.0 * r_ocn * np.cos(2.0 * np.pi * dt * f1) ) )
    P_smb = np.sqrt( P_0 / ( 1.0 + r_smb**2 - 2.0 * r_smb * np.cos(2.0 * np.pi * dt * f1) ) )

    # Half of the frequencies and length.
    P_freq = P_ocn[1:int(np.ceil(0.5 * N))]
    l_freq = len(P_freq)

    # Seed random number generator
    seed(1)

    # Create array with random phase.
    phase_half = 1j * 2.0 * np.pi * rand(l_freq)
    phase_all  = 1j * 2.0 * np.pi * rand(N)


    # Prepare variable.
    phase = np.zeros(N, dtype='complex_')

    # Fill phase array.
    phase[1:(l_freq+1)]       = phase_half
    phase[(N-l_freq-1):(N-1)] = np.conj(np.flip(phase_half))

    # Concatenate power spectra with different persistence time tau.
    P_r2_ocn = np.concatenate( [P_ocn, np.flip( P_ocn[0:int(np.floor(0.5*N))] ) ] )
    P_r2_smb = np.concatenate( [P_smb, np.flip( P_smb[0:int(np.floor(0.5*N))] ) ] )
    
    # Include identical random phase.
    # Christian et al. (2022) uses hal, why?! The time series then has a minimum
    # halfway on its lenght.
    #P_rand_ocn = P_r2_ocn * np.exp(phase)
    P_rand_ocn = P_r2_ocn * np.exp(phase_all)
    #P_rand_smb = P_r2_smb * np.exp(phase)
    P_rand_smb = P_r2_smb * np.exp(phase_all)

    # Transform back to time space to save stochastic signal.
    noise_ocn = normalised_ifft(P_rand_ocn, sigm_ocn)
    noise_smb = normalised_ifft(P_rand_smb, sigm_smb)

    out = [noise_ocn, noise_smb]

    return out


# Options.
save_nc   = False
overwrite = True

# Path and file name to write solution.
path      = '/home/dmoreno/c++/flowline/output/glacier_ews/'
file_name = 'noise.nc'

# Definitions.
tf = 1.0e4                            # End time as defined in flowline.cpp [yr].
dt = 1     # keep this at 1 for now... averaging for longer model timesteps happens in main script
N  = int(tf)
t  = dt * np.linspace(0, N, N)

# Persistence parameter frontal ablation.
tau_ocn = 10.0   # [yr]
tau_smb = 1.5    # [yr]

sigm_ocn = 12.0  # [m/yr]
sigm_smb = 0.4   # [m/yr]



# Calculate stochastic noise from accumulation and frontal ablation.
noise     = stochastic_noise(t, N, dt, sigm_ocn, sigm_smb, tau_ocn, tau_smb)
noise_ocn = noise[0]
noise_smb = noise[1]


# Save noise in a nc file.
if save_nc == True:

    # Boolean to check if file exists.
    isfile = os.path.isfile(path+file_name)
    
    # Overwriting options.
    if isfile == True:

        if overwrite == True:
            print('')
            print('Overwriting nc file.')
            subprocess.run("rm "+path+file_name, shell=True, check=True, \
                           stdout=subprocess.PIPE, universal_newlines=True)
        
        else:
            print('')
            print('WARNING!')
            print('')
            print('File already exists in this directory. Plase, select a new'+
                  'directory or delete existing noise.nc file.')
            
            # Terminate script.
            sys.exit()
        
   
    # Create nc file.
    f = nc4.Dataset(path+file_name,'w', format='NETCDF4') #'w' stands for write

    # A netCDF group is basically a directory or folder within the netCDF dataset. 
    # This allows you to organize data as you would in a unix file system.
    noise_grp = f.createGroup('Noise')

    # Dimensions. If unlimitted: noise_ocn_grp.createDimension('time', None)
    noise_grp.createDimension('time', N)

    # Create variables. Ex: tempgrp.createVariable('Noise_ocn', 'f4', ('time', 'lon', 'lat', 'z')).
    time = noise_grp.createVariable('Time', np.float64, ('time'))
    noise_ocn_nc = noise_grp.createVariable('Noise_ocn', np.float64, ('time'))
    noise_smb_nc = noise_grp.createVariable('Noise_smb', np.float64, ('time'))

    # Pass data into variables. Just real part of complex noise values.
    time[:] = t
    noise_ocn_nc[:] = np.real(noise_ocn)
    noise_smb_nc[:] = np.real(noise_smb)

    # Attributes.
    # Add global attributes.
    f.description = "Dataset containing frontal ablation and SMB time series with varying degrees"+\
                    "and types of persistence. Based on Christian et al. (2022)."
    
    today     = datetime.today()
    f.history = "Created " + today.strftime("%d/%m/%y")

    # Add local attributes to variable instances.
    time.units = 'Years'
    noise_ocn_nc.units = 'm/yr'
    noise_smb_nc.units = 'm/yr'
    noise_ocn_nc.warning = 'Assumed that dt = 1 year.'
    noise_smb_nc.warning = 'Assumed that dt = 1 year.'

    # Close dataset.
    f.close()



# PLOT.
fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

plt.rcParams['text.usetex'] = True

ax1.bar(t, noise_ocn, width=1.5, bottom=None, align='center', data=None, color='blue')

ax2.bar(t, noise_smb, width=1.5, bottom=None, align='center', data=None, color='red')

ax1.set_ylabel(r'$ \tilde{a}_{\mathrm{ocn}}(t) $', fontsize=18)
ax2.set_ylabel(r'$ \tilde{a}_{\mathrm{smb}}(t) $', fontsize=18)
ax2.set_xlabel(r'$\mathrm{Time} \ (yr) $', fontsize=18)

ax1.set_xticks([])
#ax.set_yticks([320, 330, 340, 350, 360])


#ax.legend(loc='best', ncol = 1, frameon = True, framealpha = 1.0, \
#            fontsize = 12, fancybox = True)


ax1.tick_params(axis='y', which='major', length=4, colors='black')
ax2.tick_params(axis='both', which='major', length=4, colors='black')

#ax1.grid(axis='x', which='major', alpha=0.85)

ax1.set_xlim(0, tf)
ax2.set_xlim(0, tf)
#ax.set_ylim(100, 400)

plt.tight_layout()

plt.show()
plt.close(fig)