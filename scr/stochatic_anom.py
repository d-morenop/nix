#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:01:22 2022

@author: dmoreno

This scripts creates frontal ablation and SMB time series with varying degrees
and types of persistence. Based on Christian et al. (2022). Assumed that dt = 1 year.

The idea is to save the data in a external file that will be read by the flowline 
model as a boundary condition. 

Note: the flowline time step may be shorter than 1 year (adaptative), we can further 
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
import scipy
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
    # Christian et al. (2022) uses half, why?! The time series then has a minimum
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
read_nc   = True
save_nc   = False
overwrite = False

plot_time_series = True
plot_frames      = False
save_fig         = True

# Path and file name to write solution.
path      = '/home/dmoreno/flowline/data/'
path_fig  = '/home/dmoreno/figures/transition_indicators/'
file_name = 'noise_sigm_ocn.12.0.nc'

# Definitions.
tf = 5.0e4                            # End time as defined in flow_line.cpp [yr].
dt = 1     # keep this at 1 for now... averaging for longer model timesteps happens in main script
N  = int(tf)
t  = dt * np.linspace(0, N, N)

# Persistence parameter frontal ablation and SMB.
tau_ocn = 9.0   # [yr]
tau_smb = 1.5    # [yr]

# Maximum variability in frontal ablation and SMB (standard deviation).
# Flowline crashes for values above sigm_ocn > 5.0 m/yr.
sigm_ocn = 12.0   # 12.0 [m/yr] 
sigm_smb = 0.3   # 0.4 [m/yr]


# Save noise in a nc file.
if save_nc == True:

    # Calculate stochastic noise from accumulation and frontal ablation.
    noise     = stochastic_noise(t, N, dt, sigm_ocn, sigm_smb, tau_ocn, tau_smb)
    noise_ocn = noise[0]
    noise_smb = noise[1]

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
    f = nc4.Dataset(path+file_name, 'w', format='NETCDF4') #'w' stands for write

    # A netCDF group is basically a directory or folder within the netCDF dataset. 
    # This allows you to organize data as you would in a unix file system.
    #noise_grp = f.createGroup('Noise')

    # Dimensions. If unlimitted: noise_ocn_grp.createDimension('time', None)
    #noise_grp.createDimension('time', N)
    f.createDimension('time', N)

    # Create variables. Ex: tempgrp.createVariable('Noise_ocn', 'f4', ('time', 'lon', 'lat', 'z')).
    time = f.createVariable('Time', np.float64, ('time'))
    noise_ocn_nc = f.createVariable('Noise_ocn', np.float64, ('time'))
    noise_smb_nc = f.createVariable('Noise_smb', np.float64, ('time'))

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

    # Message.
    print(file_name+' has been saved.')

# Read nc file to check.
if read_nc == True:

    # Open nc file.
    g = nc4.Dataset(path+file_name,'r')
    
    # Read variables.
    noise_ocn = g.variables["Noise_ocn"][:]
    noise_smb = g.variables["Noise_smb"][:]



##################################################################
##################################################################
# PLOT.
# Stochastic anomalies time series.

if plot_time_series == True:


    fig = plt.figure(dpi=400, figsize=(10,4))
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(212)

    ax2 = ax1.twinx()

    plt.rcParams['text.usetex'] = True

    ax1.bar(t, noise_ocn, width=2.5, bottom=None, align='center', data=None, color='blue')

    ax2.bar(t, noise_smb, width=2.5, bottom=None, align='center', data=None, color='red')

    #ax1.bar(1.0e-3*t, noise_ocn, width=0.5, bottom=None, align='center', data=None, color='blue')
    #ax2.bar(1.0e-3*t, noise_smb, width=0.5, bottom=None, align='center', data=None, color='red')

    ax1.set_ylabel(r'$ M \ (\mathrm{m/yr}) $', fontsize=20)
    ax2.set_ylabel(r'$ \mathrm{SMB}  \ (\mathrm{m/yr}) $', fontsize=20)
    ax1.set_xlabel(r'$\mathrm{Time} \ (\mathrm{kyr}) $', fontsize=20)

    
    #ax1.set_xticks([])

    ax1.set_xticks([0, 10e3, 20e3, 30e3, 40e3, 50e3])
    ax1.set_xticklabels(['$0$', '$10$', '$20$', \
                            '$30$', '$40$', '$50$'], fontsize=17)

    ax2.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    ax2.set_yticklabels(['$-1$', '$0$', '$1$', \
                            '$2$', '$3$','$4$'], fontsize=17)

    ax1.set_yticks([-45, -30, -15, 0, 15, 30])
    ax1.set_yticklabels(['$-45.0$', '$-30.0$', '$-15.0$', '$0.0$', \
                            '$15.0$', '$30.0$'], fontsize=17)
    
    
    #ax.legend(loc='best', ncol = 1, frameon = True, framealpha = 1.0, \
    #            fontsize = 12, fancybox = True)

    ax1.yaxis.label.set_color('blue')
    ax2.yaxis.label.set_color('red')

    ax1.tick_params(axis='y', which='major', length=4, colors='blue')
    ax2.tick_params(axis='y', which='major', length=4, colors='red')

    #ax1.grid(axis='x', which='major', alpha=0.85)

    ax1.set_xlim(0, tf)
    ax2.set_xlim(0, tf)
    ax1.set_ylim(-45, 30)
    ax2.set_ylim(-1.0, 4)

    #plt.tight_layout()

    if save_fig == True:
        plt.savefig(path_fig+'stoch_bc_poster.png', bbox_inches='tight')

    plt.show()
    plt.close(fig)


######################################################################
##################################################################
# SMB spatial dependency.
def f_smb(x, smb_stoch):

    # SMB stochastic variability sigmoid.
    var_pattern = var_mult + ( 1.0 - var_mult ) * 0.5 * \
                ( 1.0 + scipy.special.erf((x - x_varmid) / x_varsca) )
                
    # Total SMB: stochastic sigmoid (smb_stoch * var_pattern) + deterministic sigmoid.
    var_determ = S_0 + 0.5 * dlta_smb * ( 1.0 + scipy.special.erf((x - x_mid) / x_sca) )
    
    S = smb_stoch * var_pattern + var_determ

    out = [var_pattern, var_determ, S]

    return out

if plot_frames == True:

    S_0      = 0.4
    dlta_smb = -4.0                    
    x_acc    = 300.0e3                 
    x_mid    = 3.5e5                   
    x_sca    = 4.0e4                  
    x_varmid = 2.0e5                  
    x_varsca = 8.0e4                   
    var_mult = 0.25

    x = np.linspace(0.0, 360.0e3, 500)


    for i in range(0, N, int(0.05*N)):

        fig = plt.figure(dpi=400)
        plt.rcParams['text.usetex'] = True

        ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=3, rowspan=2)
        ax2 = plt.subplot2grid(shape=(9, 5), loc=(7, 0), colspan=2, rowspan= 2)
        ax3 = plt.subplot2grid(shape=(9, 5), loc=(7, 3), colspan=2, rowspan=2)

        # Grey shade spanning all random SMB values.
        for j in range(0, N, int(0.001*N)):
            smb   = f_smb(x, noise_smb[j])
            S_now = smb[2]
            ax1.plot(x, S_now, linestyle='-', color='grey', marker='None', \
                markersize=3.0, linewidth=2.5, alpha=0.1, label=r'$u_{b}(x)$') 
        
        smb = f_smb(x, noise_smb[i])
        var_pattern = smb[0]
        var_determ  = smb[1]
        S_now       = smb[2]

        ax1.plot(x, S_now, linestyle='-', color='red', marker='None', \
                markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

        ax2.plot(x, var_determ, linestyle='-', color='darkblue', marker='None', \
                markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

        ax3.plot(x, var_pattern, linestyle='-', color='purple', marker='None', \
                markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

        # Title.
        ax1.set_title(r'$i = \ $'+str(i)+r'$, \ t = \ $'+\
                     str(np.round(1e-3*t[i],2))+r'$ \ kyr$', fontsize=16)

        ax1.set_ylabel(r'$ \mathrm{SMB} \ (m/yr) $', fontsize=17)
        ax2.set_ylabel(r'$ S_0 (x) $', fontsize=17)
        ax3.set_ylabel(r'$ \xi (x) $', fontsize=17)
        #ax1.set_xlabel(r'$ x \ (km) $', fontsize=18)

        ax1.set_xlim(0.0, 360.0e3)
        ax2.set_xlim(0.0, 360.0e3)
        ax3.set_xlim(0.0, 360.0e3)
        ax1.set_ylim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.0)
        #ax3.set_ylim(-2.5, 2.5)

        ax1.grid(axis='x', which='major', alpha=0.85)
        ax2.grid(axis='x', which='major', alpha=0.85)
        ax3.grid(axis='x', which='major', alpha=0.85)

        ax2.set_xticklabels([])
        ax3.set_xticklabels([])

        ax1.set_yticks([-2, -1, 0, 1, 2])
        ax1.set_yticklabels(['$-2$', '$-1$', '$0$', '$-1$', '$2$'], fontsize=15)

        ax2.set_yticks([-2, -1, 0])
        ax2.set_yticklabels(['$-2$', '$-1$', '$0$'], fontsize=15)

        ax3.set_yticks([0.0, 0.5, 1.0])
        ax3.set_yticklabels(['$0.0$', '$0.5$', '$1.0$'], fontsize=15)

        ax1.set_xticks([0, 5e4, 10e4, 15e4, 20e4, 25e4, 30e4, 35e4])
        ax1.set_xticklabels(['$0$', '$50$', '$100$', '$150$', \
                             '$200$', '$250$', '$300$', 
                             '$350$'], fontsize=15)

        ax1.tick_params(axis='x', which='major', length=1, colors='black')
        ax2.tick_params(axis='x', which='major', length=0, colors='blue')
        ax3.tick_params(axis='x', which='major', length=0, colors='blue')

        plt.tight_layout()
        
        if save_fig == True:
            if i < 10:
                frame = '00000'+str(i)
            elif i > 9 and i < 100:
                frame = '0000'+str(i)
            elif i > 99 and i < 1000:
                frame = '000'+str(i)
            elif i > 999 and i < 10000:
                frame = '00'+str(i)
            elif i > 9999 and i < 100000:
                frame = '0'+str(i)
            else:
                frame = str(i)
			
            print('Frame = ', frame)
            plt.savefig(path_fig+'smb_stoch_'+frame+'.png', bbox_inches='tight')

        plt.show()
        plt.close(fig)