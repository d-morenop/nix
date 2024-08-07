#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:40:49 2021

@author: dmoreno
"""


import os
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
from dimarray import get_datadir
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
import pylab as plt_lab
from scipy.signal import argrelextrema


path_fig        = '/home/dmoreno/figures/nix/oscillations/C_thw-w_0/'
path_now        = '/home/dmoreno/nix/oscillations/S-w_0/w0.0.3_S.0.20_nz.15/'
path_stoch      = '/home/dmoreno/nix/data/'
file_name_stoch = 'noise_sigm_ocn.12.0.nc'


# '/home/dmoreno/nix/ews/M_rates/smooth/sigma.1.0_A.3.0e-26_yp.176/''
# /home/dmoreno/flowline/mismip_bp/exp_3/n.100_nz.10/

# Select plots to be saved (boolean integer).
save_series        = 0
save_series_comp   = 0
save_shooting      = 0
save_domain        = 0
coloured_domain    = 0
save_var_frames    = 0
save_series_frames = 0
save_theta         = 0
save_visc          = 0
save_u             = 0
save_u_der         = 0
time_series_gif    = 0
save_L             = 0
save_series_2D     = 1
heat_map_fourier   = 0
save_fig           = False
read_stoch_nc      = False
bed_smooth         = False

smth_series        = 0


# MISMIP bedrock experiments.
# exp = 1: inclined bed; exp = 3: overdeepening bed.
exp_name = ['mismip_1', 'mismip_3', 'glacier_ews']
idx = 0
exp = exp_name[idx]

# Create figures directory if it does not exist.
if not os.path.exists(path_fig):
	os.makedirs(path_fig)
	print(f"Directory '{path_fig}' created.")

# Open nc file in read mode.
nc_SSA = os.path.join(get_datadir(), path_now+'nix.nc')
data   = Dataset(nc_SSA, mode='r')


# Let us create a dictionary with variable names.
nix_name = ['u_bar', 'ub', 'u_bar_x', 'u_z', 'u', 'H', 'visc_bar', 'tau_b', 'tau_d', \
				 'L', 'dL_dt', 't', 'b', 'C_bed', 'dudx_bc', \
				 'BC_error', 'u2_0_vec', 'u2_dif_vec', 'picard_error', \
				 'c_picard', 'dt', 'mu', 'omega', 'A', 'theta', 'S', \
				 'm_stoch', 'smb_stoch', 'Q_fric', 'beta', 'visc', 'u_x', 'F_1', 'F_2', \
				 'A_theta', 'T_oce', 'lmbd']

var_name 	  = ['u_bar', 'ub', 'u_bar_x', 'u_z', 'u', 'H', 'visc_bar', 'tau_b', 'tau_d', \
				 'L', 'dL_dt', 't', 'b', 'C_bed', 'u_x_bc', \
				 'dif', 'u2_0_vec', 'u2_dif_vec', 'picard_error', \
				 'c_picard', 'dt', 'mu', 'omega_picard', 'A_s', 'theta', 'S', \
				 'm_stoch', 'smb_stoch', 'Q_fric', 'beta', 'visc', 'u_x', 'F_1', 'F_2', \
				 'A_theta', 'T_oce', 'lmbd']


# Dimension.
l_var = len(var_name)


# Load data from flowline.nc. t_n can be an array, np.shape(x) = (len(t_n), y, x)
# Access the globals() dictionary to introduce new variables.
for i in range(l_var):
	globals()[var_name[i]] = data.variables[nix_name[i]][:]


# Desired units.
L = 1.0e-3 * L
b = 1.0e-3 * b
H = 1.0e-3 * H
tau_b = 1.0e-3 * tau_b
beta  = 1.0e-3 * beta
theta = theta - 273.15


# Get dimensions.
l = len(t)
s = np.shape(theta)

# GENEREAL PARAMETERS
sec_year = 3.154e7

# f_cb
T       = 50.0       # years  
omega   = 2.0 * np.pi / T     # Real period is 0.5 * omega due to abs(cos(omega*t))
x_omega = 5.0e3               # m  

# f_visc_bar
n_gln = 3
A     = 4.9e-25               # 4.9e-25 (T=-10ºC) # Pa³ / s (Greve and Blatter, 2009)
B     = A**( -1 / n_gln )     # Pa³ / s  
eps   = 1.0e-12**2             # 1.0e-21



# Number of points and domain.
n = s[2]
n = 1000

# Obtain the corresponding T_air temperatures from the A forcing (MISMIP).
R = 8.314                  # [J / K mol]
Q_act = 60.0 * 1.0e3       # [kJ/mol] --> [J/mol]
A_0   = 3.985e-13 * sec_year         # [Pa^-3 s^-1]
T_air_s = - Q_act / ( R * np.log(A_s / A_0) ) - 273.15


#print('A_s = ', A_s)
#print('T_air_s = ', T_air_s)

def f_bed(x, exp, n):
	
	# Bedrock geometry options.
	if exp == 'mismip_1':
		x_tilde = x / 750.0   # in km.
		bed = 720 - 778.5 * x_tilde
		
		# Transform to kmto plot.
		bed = 1.0e-3 * bed

	elif exp == 'mismip_3':      
		x_tilde = x / 750.0   # in km.              
		# Schoof 2184.8
		bed = ( 729.0 - 2148.8 * x_tilde**2 + \
						+ 1031.72 * x_tilde**4 + \
						- 151.72 * x_tilde**6 )

		# Transform to kmto plot.
		bed = 1.0e-3 * bed

	elif exp == 'glacier_ews':

		# Prepare variable.
		bed = np.empty(n)

		# Horizontal domain extenstion [km].
		x_1 = 346.0
		x_2 = 350.0

		# Initial bedrock elevation (x = 0) [km].
		y_0 = 0.07

		# Peak height [km]. 88.0e-3 up to 100.0e-3
		y_p = 88.0e-3

		# Intermideiate slope.
		m_bed = y_p / ( x_2 - x_1 )

		# Counters.
		c_x1 = 0
		c_x2 = 0

		for i in range(n):

			# First part.
			if x[i] <= x_1:
				bed[i] = y_0 - 1.5e-3 * x[i]
			
			# Second.
			elif x[i] >= x_1 and x[i] < x_2:
				
				# Save index of last point in the previous interval.
				if c_x1 == 0:
					y_1  = bed[i-1]
					c_x1 = c_x1 + 1	
				
				# Bedrock function.
				bed[i] = y_1 + m_bed * ( x[i] - x_1 )
			
			# Third.
			
			elif x[i] >= x_2:
				# Save index of last point in the previous interval.
				if c_x2 == 0:
					y_2  = bed[i-1]
					c_x2 = c_x2 + 1	
				
				# Bedrock function.
				bed[i] = y_2 - 5.0e-3 * ( x[i] - x_2 )
	
	return bed

# Account for unevenly-spaced horizontal grid.
sigma = np.linspace(0, 1.0, s[2])
sigma_plot = sigma**(0.5) # 0.5 (uneven), 1.0 (even)



# Horizontal dimension to plot. x_plot [km].
if exp == 'mismip_1' or 'mismip_3':
	x_plot = np.linspace(0, 2000.0, n)

elif exp == 'glacier_ews':
	x_plot = np.linspace(0, 400.0, n)


# Bedrock test [km].
# PROBLEM HERE, WE HAVE TO FORCE IT AGAIN.
#x_plot = np.linspace(0, 400.0, n)
#print(x_plot[n-1])

bed = f_bed(x_plot, exp, n)

if bed_smooth == True:
	n_smth = n
	dx = 1.0 # This must be 1.0 to leave unchaged the max/min values in the y axis.
	sigma = 2.0 * dx
	A = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

	summ = np.zeros(n_smth)
	kernel = np.empty([n_smth,n_smth])
	
	smth = bed

	# Number of points at the edges of the array that are not smoothed out.
	p = 3
	summ[p:(n_smth-1-p)] = 0.0

	# Weierstrass transform.
	for i in range(p, n_smth-p, 1):
		x = i * dx
		for j in range(n_smth):
			y = j * dx
			kernel[i,j] = np.exp(- 0.5 * ((x - y) / sigma)**2)
			summ[i] += bed[j] * kernel[i,j]
		

	# Normalizing Kernel.
	smth[p:(n_smth-1-p)] = A * summ[p:(n_smth-1-p)]
	bed = smth


# Read stochastic noise to plot.
if read_stoch_nc == True:
	
	# Open nc file.
	g = nc4.Dataset(path_stoch+file_name_stoch,'r')

	# Read variables.
	noise_ocn = g.variables["Noise_ocn"][:]
	noise_smb = g.variables["Noise_smb"][:]
	t_stoch   = g.variables["Time"][:]

	# We only allow for possitive ablation values.
	noise_ocn = np.where(noise_ocn > 0.0, noise_ocn, 0.0)


# Sea level array.
sl    = np.zeros(n)


#############################################
#############################################
# TIME SERIES
if save_series == 1:

	# Avoid large values for visualization.
	#dL_dt[0] = np.nan

	# Kyr to plot.
	t_plot = 1.0e-3 * t

	# Ice flux.
	q = u_bar * 1.0e3 * H

	# Plot bed peak position.
	y_p = np.full(len(t_plot), 350)

	# Mean variables.
	theta_bar     = np.mean(theta[:,s[1]-1,:], axis=1)
	visc_bar_mean = np.mean(visc_bar, axis=1)

	# T_air
	T_air = theta[:,s[1]-1,0]

	T_oce = T_oce - 273.15
	
	# Figure.
	fig = plt.figure(dpi=600, figsize=(5.5,6))

	ax = fig.add_subplot(311)
	ax6 = ax.twinx()
	ax2 = fig.add_subplot(312)
	ax4 = ax2.twinx()
	ax3 = fig.add_subplot(313)
	ax5 = ax3.twinx()
	

	"""
	fig = plt.figure(dpi=600, figsize=(5.5,4.5))

	ax = fig.add_subplot(211)
	ax6 = ax.twinx()
	ax2 = fig.add_subplot(212)
	ax4 = ax2.twinx()
	"""
	
	plt.rcParams['text.usetex'] = True

	# Vertically-averaged velocity.
	ax6.plot(t_plot, u_bar[:,s[2]-1], linestyle='-', color='blue', marker='None', \
			 markersize=3.0, linewidth=2.0, alpha=1.0, label=r'$u_{b}(x)$') 

	# Grounding line position.
	ax.plot(t_plot, L, linestyle='-', color='red', marker='None', \
			markersize=3.0, linewidth=2.0, alpha=1.0, label=r'$u_{b}(x)$') 

	# Ice thickness at the grounding line.
	ax2.plot(t_plot, H[:,s[2]-1], linestyle='-', color='black', marker='None', \
			 markersize=3.0, linewidth=2.0, alpha=1.0, label=r'$u_{b}(x)$') 
	
	# Basal temperature at the grounding line.
	ax4.plot(t_plot, theta[:,0,s[2]-1], linestyle='-', color='brown', marker='None', \
					 markersize=3.0, linewidth=1.5, alpha=1.0, label=r'$u_{b}(x)$')
	

	# Smooth.
	if smth_series == 1:
		u_L = signal.savgol_filter(u_L,
							20, # window size used for filtering
							8) # order of fitted polynomial							

	
	#ax3.plot(t_plot, T_air, linestyle='-', color='purple', marker='None', \
	#		 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	#ax3.plot(t_plot, T_oce, linestyle='-', color='purple', marker='None', \
	#			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

	#ax3.plot(t_plot, b[:,s[2]-1], linestyle='-', color='purple', marker='None', \
#			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

	
	
	# Ice rate factor.
	#ax4.plot(t_plot, A_s, linestyle='-', color='darkgreen', marker='None', \
	#		 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
			 
	# Histograms for stochastic time series.
	if read_stoch_nc == True:

		# Plot bed peak position.
		ax.plot(t_plot, y_p, linestyle='--', color='red', marker='None', \
				markersize=3.0, linewidth=1.5, alpha=1.0, label=r'$u_{b}(x)$') 

		ax3.bar(t_plot, smb_stoch, width=0.025, bottom=None, \
		  			align='center', data=None, color='darkgreen')
		ax5.bar(t_plot, m_stoch, width=0.025, bottom=None, \
		  			align='center', data=None, color='purple')

		
		"""ax3.bar(t_stoch, noise_smb, width=1.0, bottom=None, \
				align='center', data=None, color='darkgreen')
		ax5.bar(t_stoch, noise_ocn, width=1.0, bottom=None, \
				align='center', data=None, color='purple')"""
		
		#ax3.set_xlim(t_stoch[0], t_stoch[len(t_stoch)-1])

		# Labels.
		ax3.set_ylabel(r'$ \mathrm{SMB} \ (m / yr) $', fontsize=18)
		ax5.set_ylabel(r'$ \dot{m} \ (m/yr)$', fontsize=18)

		# Axis limits.
		ax.set_ylim(200.0, 400.0)
		ax3.set_ylim(-2.0, 0.5)
		ax5.set_ylim(0.0, 50.0)
	
	#ax.set_xlim(0, t_plot[s[0]-1])
	#ax2.set_xlim(0, t_plot[s[0]-1])
	#ax3.set_xlim(0, t_plot[s[0]-1])

	ax.set_xlim(t_plot[int(0.15*s[0])], t_plot[s[0]-1])
	ax2.set_xlim(t_plot[int(0.15*s[0])], t_plot[s[0]-1])
	ax3.set_xlim(t_plot[int(0.15*s[0])], t_plot[s[0]-1]) #0.5
	
	ax.set_ylabel(r'$L \ (\mathrm{km})$', fontsize=18)
	ax2.set_ylabel(r'$H_{gl} \ (\mathrm{km})$', fontsize=18)

	
	#ax3.set_ylabel(r'$ T{\mathrm{air}} \ (^{\circ} \mathrm{C})$', fontsize=18)
	#ax3.set_ylabel(r'$ \Delta T_{\mathrm{oce}} \ (^{\circ} \mathrm{C})$', fontsize=18)
	#ax4.set_ylabel(r'$ \theta(z=0,L) $', fontsize=18)
	
	ax4.set_ylabel(r'$  \theta(0,L) \ (^{\circ} \mathrm{C})$', fontsize=17)
	#ax4.set_ylabel(r'$ A \ (\mathrm{Pa}^{-3} \mathrm{yr}^{-1})$', fontsize=17)
	#ax5.set_ylabel(r'$ \dot{m} \ (\mathrm{m/yr})$', fontsize=17)
	
	#ax5.set_ylabel(r'$ \bar{\eta} \ (\mathrm{Pa \cdot s}) $', fontsize=18)
	ax6.set_ylabel(r'$ \bar{u}(L) \ (\mathrm{m/yr})$', fontsize=18)
	#ax3.set_xlabel(r'$\mathrm{Time} \ (\mathrm{kyr})$', fontsize=18)
	ax3.set_xlabel(r'$\mathrm{Time} \ (\mathrm{kyr})$', fontsize=18)

	#ax.set_xticks([0, 10, 20, 30, 40])
	#ax.set_xticklabels(['', '', '', '', ''], fontsize=15)
	
		
	ax.yaxis.label.set_color('red')
	ax2.yaxis.label.set_color('black')
	ax3.yaxis.label.set_color('darkgreen')
	ax5.yaxis.label.set_color('purple')
	ax4.yaxis.label.set_color('brown')
	ax6.yaxis.label.set_color('blue')
	
	ax.set_xticklabels([])
	ax2.set_xticklabels([])


	ax.tick_params(axis='y', which='major', length=4, colors='red', labelsize=16)
	ax2.tick_params(axis='y', which='major', length=4, colors='black', labelsize=16)
	ax2.tick_params(axis='x', which='major', length=4, colors='black', labelsize=16)
	ax5.tick_params(axis='y', which='major', length=4, colors='purple', labelsize=16)
	ax3.tick_params(axis='x', which='major', length=4, colors='black', labelsize=16)
	ax3.tick_params(axis='y', which='major', length=4, colors='darkgreen', labelsize=16)
	ax4.tick_params(axis='y', which='major', length=4, colors='brown', labelsize=16)
	#ax4.tick_params(axis='y', which='major', length=4, colors='brown', labelsize=16)
	ax6.tick_params(axis='y', which='major', length=4, colors='blue', labelsize=16)
	
	ax.grid(axis='x', which='major', alpha=0.85)
	ax2.grid(axis='x', which='major', alpha=0.85)
	ax3.grid(axis='x', which='major', alpha=0.85)

	
	plt.tight_layout()

	if save_fig == True:
		plt.savefig(path_fig+'time_series.png', bbox_inches='tight')

	# Display and close figure.
	plt.show()
	plt.close(fig)




#############################################
#############################################
# TIME SERIES COMPUTATIONAL PERFORMANCE
if save_series_comp == 1:
	
	# Time in kyr.
	t_plot = 1.0e-3 * t

	# Avoid first error points as it is imposed.
	visc_bar[0]     = np.nan
	picard_error[0] = np.nan
	
	# Figure.
	fig = plt.figure(dpi=600, figsize=(5.5,6))
	ax = fig.add_subplot(311)
	ax6 = ax.twinx()
	ax2 = fig.add_subplot(312)
	ax4 = ax2.twinx()
	ax3 = fig.add_subplot(313)
	ax5 = ax3.twinx()
	
	plt.rcParams['text.usetex'] = True
	
	ax.plot(t_plot[1:n-1], c_picard[1:n-1], linestyle='-', color='darkblue', marker='None', \
			markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
	#ax6.plot(t_plot, visc_bar, linestyle='-', color='purple', marker='None', \
	#		markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
	ax2.plot(t_plot, dt, linestyle='-', color='black', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

	ax4.plot(t_plot, np.log10(picard_error), linestyle='-', color='red', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$')
	
	ax3.plot(t_plot, omega_picard/np.pi, linestyle='-', color='blue', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
	ax5.plot(t_plot, mu, linestyle='-', color='black', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
	
	ax.set_ylabel(r'$N_{\mathrm{pic}}$',fontsize=18)
	ax2.set_ylabel(r'$ \Delta t \ (yr)$',fontsize=18)
	ax4.set_ylabel(r'$ \mathrm{log}_{10} (\varepsilon) $',fontsize=18)
	ax3.set_ylabel(r'$ \omega \ (\pi \ \mathrm{rad}) $',fontsize=18)
	ax5.set_ylabel(r'$ \mu $',fontsize=18)
	ax6.set_ylabel(r'$ \eta \ (Pa \cdot s) $',fontsize=18)
	ax3.set_xlabel(r'$\mathrm{Time} \ (kyr)$',fontsize=18)
	
	ax.set_xlim(t_plot[0], t_plot[l-1])
	ax2.set_xlim(t_plot[0], t_plot[l-1])
	ax3.set_xlim(t_plot[0], t_plot[l-1])
		
	ax.yaxis.label.set_color('darkblue')
	ax2.yaxis.label.set_color('black')
	ax3.yaxis.label.set_color('blue')
	ax4.yaxis.label.set_color('red')
	ax5.yaxis.label.set_color('black')
	ax6.yaxis.label.set_color('purple')
	
	ax.set_xticklabels([])
	ax2.set_xticklabels([])
	
	ax.tick_params(axis='y', which='major', length=4, colors='darkblue')
	ax2.tick_params(axis='y', which='major', length=4, colors='black')
	ax3.tick_params(axis='y', which='major', length=4, colors='blue')
	ax4.tick_params(axis='y', which='major', length=4, colors='red')
	ax5.tick_params(axis='y', which='major', length=4, colors='black')
	ax6.tick_params(axis='y', which='major', length=4, colors='purple')
	
	ax.grid(axis='x', which='major', alpha=0.85)
	ax2.grid(axis='x', which='major', alpha=0.85)
	ax3.grid(axis='x', which='major', alpha=0.85)
	
	plt.tight_layout()
	if save_fig == True:
		plt.savefig(path_fig+'time_series_comp.png', bbox_inches='tight')
	plt.show()
	plt.close(fig)

#######################################
#######################################
# SHOOTING CONVERGENCE

if save_shooting == 1:
	
	for i in range(12,23):
		
		n_c = 5

		fig = plt.figure(dpi=600, figsize=(5.5,6))
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		ax3 = ax2.twinx()
	
		ax.plot(u_x_0_vec[i,0:n_c], u_x_dif_vec[i,0:n_c], linestyle='-', color='purple', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$sl(x)$') 
		
		ax2.plot(u_x_0_vec[i,0:n_c], linestyle='-', color='darkgreen', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$z_s(x)$') 
		ax3.plot(u_x_dif_vec[i,0:n_c], linestyle='-', color='purple', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$bed(x)$') 

	
		ax.set_ylabel(r'$\Delta u_{2}(L) \ (1/s)$', fontsize=20)
		ax.set_xlabel(r'$u_{2}(0) \ (1/s) $', fontsize=20)
		
		ax2.set_ylabel(r'$u_{2}(0) \ (1/s) $', fontsize=20)
		ax3.set_ylabel(r'$\Delta u_{2}(L) \ (1/s)$', fontsize=20)
		ax2.set_xlabel(r'$ \mathrm{Iteration} $', fontsize=20)
	
		ax.yaxis.label.set_color('purple')
		ax2.yaxis.label.set_color('darkgreen')
		ax3.yaxis.label.set_color('purple')
		
		ax2.tick_params(axis='y', which='major', length=4, colors='darkgreen')
		ax3.tick_params(axis='y', which='major', length=4, colors='purple')
		
		
		ax.set_xlim(np.nanmin(u_x_0_vec), np.nanmax(u_x_0_vec))
		ax.set_ylim(np.nanmin(u_x_dif_vec), np.nanmax(u_x_dif_vec))
		ax2.set_xlim(0, n_c)
		ax2.set_ylim(np.nanmin(u_x_0_vec), np.nanmax(u_x_0_vec))
		ax3.set_ylim(np.nanmin(u_x_dif_vec), np.nanmax(u_x_dif_vec))
	 	
		ax.tick_params(axis='both', which='major', length=4, colors='black')
		
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t = \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
	 	
		ax.grid(axis='x', which='major', alpha=0.85)
		ax2.grid(axis='x', which='major', alpha=0.85)
	
		
		##### Frame name ########
		if i < 10:
			frame = '000'+str(i)
		elif i > 9 and i < 100:
			frame = '00'+str(i)
		elif i > 99 and i < 1000:
			frame = '0'+str(i)
		else:
			frame = str(i)
		
		plt.tight_layout()
		if save_fig == True:
			plt.savefig(path_fig+'shooting_iter_'+frame+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)
		print('Saved')




#######################################
#######################################
# ENTIRE DOMAIN

if save_domain == 1:
	
	for i in range(l-1, l, 1): # range(0, l, 2), (l-1, l, 20)
		
		# Horizontal dimension [km].
		#L_plot  = np.linspace(0, L[i], s[2])
		L_plot_sigma = sigma_plot * L[i]
		
		# Ice surface elevation [km].
		#H[i,1] = H[i,2]
		#H[i,0] = H[i,2]
		z_s = H[i,:] + b[i,:]
		
		# Gaussian smooth for resolution jiggling.
		#z_s = gaussian_filter1d(z_s, 1.5)
		
		# Vertical gray line in ice front.
		n_frnt = 100
		frnt   = np.linspace(b[i,s[2]-1], z_s[s[2]-1], n_frnt)
		frnt_L = np.full(n_frnt, L[i])
		
		# Ocean.
		bed_p = np.where(x_plot > L[i], bed, np.nan)
		sl    = np.zeros(n)
		sl    = np.where(x_plot > L[i], sl, np.nan)	
		
		# Figure.
		fig = plt.figure(dpi=400) # (5,7)
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)
	
		# Ocean surface.
		ax.plot(x_plot, sl, linestyle='-', color='blue', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$sl(x)$') 
		
		# Dotted ocean surface for geometry.
		#ax.plot(x_plot, np.zeros(n), linestyle=':', color='black', marker='None', \
	  	#		linewidth=1.0, alpha=1.0, label=r'$sl(x)$') 
			
		# Vertical line (ice front).
		ax.plot(frnt_L, frnt, linestyle='-', color='darkgrey', marker='None', \
	  			linewidth=3.0, alpha=1.0, label=r'$z_s(x)$') 
		
		# Bedrock elevation.
		ax.plot(x_plot, bed, linestyle='-', color='saddlebrown', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$bed(x)$') 


		# Ice surface elevation.
		ax.plot(L_plot_sigma, z_s, linestyle='-', color='darkgrey', marker='None', \
	  			linewidth=3.0, alpha=1.0, label=r'$z_s(x)$')  
	
		
		# Shade colours.
		ax.fill_between(x_plot, bed_p, 0.0,\
	 						   facecolor='blue', alpha=0.4)
		ax.fill_between(x_plot, bed, -2.5e3,\
	 						   facecolor='saddlebrown', alpha=0.4)

		if coloured_domain == 0:
			ax.fill_between(L_plot_sigma, b[i,:], z_s,\
								facecolor='grey', alpha=0.4)

		
		########################################################################
		########################################################################
		# PLOT DOMAIN WITH A COLOUR MAP THAT REPRESENTS THE TEMPERATURE WITHIN THE ICE SHEET.
		# Create the colored plot
		elif coloured_domain == 1:
			theta_min = np.min(-theta)
			theta_max = np.max(-theta)
			
			#theta_min = np.min(u)
			#theta_max = np.max(u)

			# Minus sign just for visualization purposes.
			color_theta = - np.flip(theta[i,:,:],axis=0)
			#color_theta = np.flip(u[i,:,:],axis=0)

			# Plot a rectangle.
			def rect(ax, x, b, y, w, h, c,**kwargs):
				
				# Varying only in x.
				if len(c.shape) is 1:
					rect = plt_lab.Rectangle((x, y), w, h, color=c, ec=c,**kwargs)
					ax.add_patch(rect)
				
				# Varying in x and y.
				else:
					# Split into a number of bins
					N = c.shape[0]
					#hb = h/float(N); yl = y - hb    
					
					# Consider a non-evenly spaced vertical grid.
					#dz = np.linspace(0.0,1.0,N+1)**2
					dz = np.linspace(0.0,1.0,N+1)
					hb = dz * h; yl = y - hb

					for i in range(N-1):
						dz_H = hb[i+1]-hb[i]
						#yl += hb
						#rect = plt_lab.Rectangle((x, yl), w, hb, 
						#					color=c[i,:], ec=c[i,:],**kwargs)
						
						rect = plt_lab.Rectangle((x, b-hb[i+1]), w, dz_H, 
											color=c[i,:], ec=c[i,:],**kwargs)
						ax.add_patch(rect)

					# Plot the uppermost region as len(dz_H)=N+1 but the loop goes to N-1.
					rect = plt_lab.Rectangle((x, b-hb[N]), w, dz_H, 
											color=c[N-1,:], ec=c[N-1,:],**kwargs)
					ax.add_patch(rect)

			# Fill a contour between two lines.
			def rainbow_fill_between(ax, X, Y1, Y2, colors=None, 
									cmap=plt.get_cmap("Spectral").reversed(),**kwargs):
				
				plt.plot(X,Y1,lw=0)  # Plot so the axes scale correctly

				dx = X[1]-X[0]
				N  = X.size

				# Pad a float or int to same size as x.
				if (type(Y2) is float or type(Y2) is int):
					Y2 = np.array([Y2]*N)

				# No colors -- specify linear.
				if colors is None:
					colors = []
					for n in range(N):
						colors.append(cmap(n/float(N)))
				
				# Varying only in x.
				elif len(colors.shape) is 1:
					colors = cmap((colors-colors.min())
								/(colors.max()-colors.min()))
				
				# Varying only in x and y.
				else:
					cnp = np.array(colors)
					colors = np.empty([colors.shape[0],colors.shape[1],4])
					for i in range(colors.shape[0]):
						for j in range(colors.shape[1]):
							#colors[i,j,:] = cmap(1.0 - (cnp[i,j]-cnp[:,:].min())
							#					/(cnp[:,:].max()-cnp[:,:].min()))
							
							#colors[i,j,:] = cmap(1.0 - ( (cnp[i,j]-cnp[:,:].min())
							#					 / (theta_max-theta_min) ) )

							colors[i,j,:] = cmap(1.0 - ( abs(cnp[i,j]-theta_min)
												/ abs(theta_max-theta_min) ) )

						
							

				colors = np.array(colors)
				#colors = 1.0 - np.array(colors)

				# Create the patch objects.
				for (color,x,y1,y2) in zip(colors,X,Y1,Y2):
					rect(ax,x,y1,y2,dx,y1-y2,color,**kwargs)


			# Data    
			X  = L_plot_sigma # L_plot
			Y1 = b[i,:] 
			Y2 = z_s
			g  = color_theta

			# Colourmap.
			cmap = plt.get_cmap("Spectral")
			reversed_cmap = cmap.reversed()

			# Plot fill and curves changing in x and y.
			colors = np.rot90(g,3)
			rainbow_fill_between(ax, X, Y1, Y2, colors=colors)

			# Add a colorbar based on the colormap
			#cbar_ax = fig.add_axes([1.025, 0.17, 0.045, 0.779]) 
			cbar_ax = fig.add_axes([1.01, 0.19, 0.045, 0.76]) 
			cb = fig.colorbar(plt.cm.ScalarMappable(cmap=reversed_cmap), \
					 				cax=cbar_ax, extend='neither')

			# Set the modified ticks and tick labels
			"""
			ticks = np.linspace(0, 1, 5)
			ticks_lab = np.round(np.linspace(-theta_max, -theta_min, 5), 0)
			cb.set_ticks(ticks)
			cb.set_ticklabels([r'$-80$', r'$-60$', r'$-40$', r'$-20$', r'$0$',], \
								fontsize=13)
			"""
			cb.set_label(r'$ \theta (x,z) \ (^{\circ} \mathrm{C}) $', \
							rotation=90, labelpad=8, fontsize=22)

		########################################################################
		########################################################################

		ax.set_ylabel(r'$z \ (\mathrm{km})$', fontsize=20)
		ax.set_xlabel(r'$x \ (\mathrm{km}) $', fontsize=20)
	
		ax.yaxis.label.set_color('black')
	 	
		ax.tick_params(axis='both', which='major', length=4, colors='black', labelsize=20)
	
		if idx == 2:
			"""
			ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
			ax.set_xticklabels(['$0$', '$250$', '$500$', '$750$',\
							'$1000$', '$1250$','$1500$', '$1750$'], fontsize=15)
			
			ax.set_yticks([-1, 0, 1, 2, 3, 4, 5, 6])
			ax.set_yticklabels(['$-1$', '$0$', '$1$',\
							'$2$', '$3$','$4$','$5$','$6$'], fontsize=15)
			"""
			ax.set_xlim(0, 400)
			ax.set_ylim(-0.75, 3.0)
		
		else:
			ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750])
			ax.set_xticklabels(['$0$', '$250$', '$500$', '$750$',\
							'$1000$', '$1250$','$1500$', '$1750$'], fontsize=15)
			
			ax.set_yticks([-1, 0, 1, 2, 3, 4, 5, 6])
			ax.set_yticklabels(['$-1$', '$0$', '$1$',\
							'$2$', '$3$','$4$','$5$','$6$'], fontsize=15)
			ax.set_xlim(0, 1750)
			ax.set_ylim(-1.0, 6.0)
		
		# Title.
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t = \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
	 	
		ax.grid(axis='x', which='major', alpha=0.85)
		
		##### Frame name ########
		if i < 10:
			frame = '000'+str(i)
		elif i > 9 and i < 100:
			frame = '00'+str(i)
		elif i > 99 and i < 1000:
			frame = '0'+str(i)
		else:
			frame = str(i)
		
		plt.tight_layout()
		
		if save_fig == True:
			plt.savefig(path_fig+'domain_'+frame+'.png', bbox_inches='tight')
			print('Saved')
		
		plt.show()
		plt.close(fig)
		




#############################################
#############################################
# VARIABLES FRAMES

if save_var_frames == 1:
	
	for i in range(int(0.8*l), l, 50): # (0, l, 10), (l-1, l, 1)
		
		#L_plot  = np.linspace(0, L[i], s[2])
		L_plot = sigma_plot * L[i]
		x_tilde = L_plot / 750.0  
		bed_L   = ( 729.0 - 2184.8 * x_tilde**2 + \
			               1031.72 * x_tilde**4 - \
					       151.72 * x_tilde**6 )
		

		# Vertically averaged du/dx.
		u_x_bar   = np.mean(u_x[i,:,:], axis=0)
		theta_bar = np.mean(theta[i,:,:], axis=0)
			
		######################################
		######################################
		# L PLOTS.
		
		fig = plt.figure(dpi=600, figsize=(5.5,6)) # (5,7)
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(311)
		ax2 = fig.add_subplot(312)
		ax3 = fig.add_subplot(313)
		ax4 = ax3.twinx()
		ax7 = ax3.twinx()
		ax5 = ax.twinx()
		ax6 = ax2.twinx()
		 
	
		ax.plot(L_plot, u_bar[i,:], linestyle='-', color='blue', marker='None', \
	 			linewidth=2.0, alpha=1.0, label=r'$u_{b}(x)$') 
		#ax5.plot(L_plot, u_x_bar, linestyle='-', color='darkgreen', marker='None', \
	 	#		linewidth=2.0, alpha=1.0, label=r'$\partial u_{b}/\partial x$')  
		ax5.plot(L_plot, theta_bar, linestyle='-', color='darkgreen', marker='None', \
	 			linewidth=2.0, alpha=1.0, label=r'$\partial u_{b}/\partial x$')  
		ax3.plot(L_plot, visc_bar[i,:], linestyle='-', color='purple', marker='None', \
	  	 		linewidth=2.0, alpha=1.0, label=r'$\partial H/\partial x$') 
		#ax4.plot(L_plot, beta[i,:], linestyle='-', color='brown', marker='None', \
	  	# 		linewidth=2.0, alpha=1.0, label=r'$S(x) $') 
		ax4.plot(L_plot, b[i,:], linestyle='-', color='brown', marker='None', \
	  	 		linewidth=2.0, alpha=1.0, label=r'$S(x) $') 
		ax2.plot(L_plot, H[i,:], linestyle='-', color='black', marker='None', \
	   			linewidth=2.0, alpha=1.0, label=r'$H(x)$')  
		ax6.plot(L_plot, tau_b[i,:], linestyle='-', color='red', marker='None', \
	 			linewidth=2.0, alpha=1.0, label=r'$\tau_{b}(x)$')
		#ax4.plot(L_plot, bed_L, linestyle='-', color='brown', marker='None', \
	 	#		linewidth=2.0, alpha=1.0, label=r'$\tau_{d}(x)$') 
		#ax4.plot(L_plot, C_bed[i,:], linestyle='-', color='brown', marker='None', \
	 	#		linewidth=2.0, alpha=1.0, label=r'$\tau_{d}(x)$') 
	
	
		ax.set_ylabel(r'$ \bar{u} (x) \ (\mathrm{m/yr})$',fontsize=16)
		ax3.set_ylabel(r'$\bar{\eta} (x)\ (\mathrm{Pa \cdot s})$',fontsize=16)
		ax4.set_ylabel(r'$ b(x) \ (km)$',fontsize=16)
		#ax4.set_ylabel(r'$ \beta(x) \ (\mathrm{kPa \ yr/m})$',fontsize=16)
		#ax5.set_ylabel(r'$\partial \bar{u}_{b}/\partial x $',fontsize=16)
		ax5.set_ylabel(r'$ \bar{\theta}_{b} \ (^{\circ} \mathrm{C}) $',fontsize=16)
		ax2.set_ylabel(r'$H(x) \ (\mathrm{km})$', fontsize=16)
		ax6.set_ylabel(r'$\tau_{b}(x) \ (\mathrm{kPa})$', fontsize=16)
		ax3.set_xlabel(r'$x \ (\mathrm{km}) $',fontsize=16)
		ax7.set_yticks([])
	
		ax.yaxis.label.set_color('blue')
		ax5.yaxis.label.set_color('darkgreen')
		ax4.yaxis.label.set_color('brown')
		ax6.yaxis.label.set_color('red')
		ax3.yaxis.label.set_color('purple')
		ax2.yaxis.label.set_color('black')
	 	
		ax.tick_params(axis='y', which='major', length=4, colors='blue')
		ax6.tick_params(axis='y', which='major', length=4, colors='red')
		ax3.tick_params(axis='y', which='major', length=4, colors='purple')
		ax2.tick_params(axis='y', which='major', length=4, colors='black')
		ax5.tick_params(axis='y', which='major', length=4, colors='darkgreen')
		ax4.tick_params(axis='y', which='major', length=4, colors='brown')
	
		ax.set_xticklabels([])
		ax2.set_xticklabels([])
		ax6.set_xticklabels([])
		#ax3.set_xticks([0, 5, 10, 15, 20, 25, 30])
		#ax3.set_xticklabels(['$0$', '$5$', '$10$', '$15$', '$20$', '$25$', '$30$'],\
		#				  fontsize=15)
		#ax3.set_xticks([0, 100, 200, 300, 400, 500])
		#ax3.set_xticklabels(['$0$', '$100$', '$200$', '$300$', '$400$', '$500$'],\
	#					  fontsize=15)
	
	# 	ax.set_yticks([0,250,500])
	# 	ax.set_yticklabels(['$0$', '$250$', '$500$'], fontsize=15)
	# 	ax2.set_yticks([500,1000])
	# 	ax2.set_yticklabels(['$0.5$', '$1$'], fontsize=15)
	# 	ax3.set_yticks([100,200])
	# 	ax3.set_yticklabels(['$100$', '$200$'], fontsize=15)
	# 	ax4.set_yticks([20, 40, 60])
	# 	ax4.set_yticklabels(['$20$', '$40$', '$60$'], fontsize=15)
	# 	ax5.set_yticks([-10,0,10])	
	# 	ax5.set_yticklabels(['$-10$', '$0$', '$10$'], fontsize=15)
	# 	ax6.set_yticks([-5.50, -5.20])	
	# 	ax6.set_yticklabels(['$-5.50$', '$-5.20$'], fontsize=15)
	 	
	
		ax.set_xlim(0, L[i])
		ax2.set_xlim(0, L[i])
		ax3.set_xlim(0, L[i])
		#ax4.set_ylim(-2.5, 2.5)
		 
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
	 	
		ax.grid(axis='x', which='major', alpha=0.85)
		ax2.grid(axis='x', which='major', alpha=0.85)
		ax3.grid(axis='x', which='major', alpha=0.85)
		
			
		##### Frame name ########
		if i < 10:
			frame = '000'+str(i)
		elif i > 9 and i < 100:
			frame = '00'+str(i)
		elif i > 99 and i < 1000:
			frame = '0'+str(i)
		else:
			frame = str(i)
		
		plt.tight_layout()
		if save_fig == True:
			plt.savefig(path_fig+'flow_line_var_'+frame+'.png', bbox_inches='tight')
			print('Saved')
		
		# Display and close figure.
		plt.show()
		plt.close(fig)
		
		
	


#############################################
#############################################
# TEMPERATURE FRAMES

if save_theta == 1:

	# Number of x ticks.
	n_ticks = 5
	x_ticks = np.linspace(0, s[2], n_ticks)
	n_z     = np.shape(theta)[1]
	z_ticks = int(0.2 * n_z + 1)

	# n_z-0.5 to avoid half of grid cell in black when plotting.
	y_ticks  = np.linspace(0, n_z-0.5, z_ticks, dtype=int)
	y_labels = np.linspace(0, n_z, z_ticks, dtype=int)

	# Theta limits.
	theta_min = np.round(np.min(theta), 1)
	theta_max = 0.0

	cb_ticks = np.round(np.linspace(theta_min, theta_max, 6),1)
	
	for i in range(20, l, 10):

		# Update x_labels as domain extension changes in each iteration.
		x_labels  = np.linspace(0, L[i], n_ticks, dtype=int)

		# Colourmap.
		cmap = plt.get_cmap("Spectral")
		reversed_cmap = cmap.reversed()
		
		fig = plt.figure(dpi=600, figsize=(6,4))
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)

		# Flip theta matrix so that the plot is not upside down.
		im = ax.imshow(np.flip(theta[i,:,:],axis=0), cmap=reversed_cmap, \
						vmin=theta_min, vmax=theta_max, aspect='auto')
	
		ax.set_ylabel(r'$ \mathbf{n}_{z} $', fontsize=20)
		ax.set_xlabel(r'$\ \mathbf{x} \ (\mathrm{km})$', fontsize=20)

		divider = make_axes_locatable(ax)
		cax     = divider.append_axes("right", size="5%", pad=0.1)
		cb      = fig.colorbar(im, cax=cax, extend='neither')

		cb.set_ticks(cb_ticks)
		cb.set_ticklabels(list(cb_ticks), fontsize=14)

		cb.set_label(r'$\theta (x,z) \ (^{\circ} \mathrm{C})$', \
					 rotation=90, labelpad=6, fontsize=20)

		ax.set_xticks(x_ticks)
		ax.set_xticklabels(list(x_labels), fontsize=15)

		
		ax.set_yticks(y_ticks)
		ax.set_yticklabels(list(y_labels[::-1]), fontsize=15)
		
	
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
		plt.tight_layout()

		if save_fig == True:
			##### Frame name ########
			if i < 10:
				frame = '000'+str(i)
			elif i > 9 and i < 100:
				frame = '00'+str(i)
			elif i > 99 and i < 1000:
				frame = '0'+str(i)
			else:
				frame = str(i)
			
			plt.savefig(path_fig+'flow_line_theta_'+frame+'.png', bbox_inches='tight')
		
		plt.show()
		plt.close(fig)
		



#############################################
#############################################
# VISC(x,z,t) and u_x(x,z,t) FRAMES

if save_visc == 1:

	# Units 10⁶ Pa·s.
	#visc = 1.0e-6 * visc

	# Number of x ticks.
	n_ticks = 5
	x_ticks = np.linspace(0, n, n_ticks)
	n_z     = np.shape(visc)[1]
	z_ticks = int(0.2 * n_z + 1)

	# n_z-0.5 to avoid half of grid cell in black when plotting.
	y_ticks  = np.linspace(0, n_z-0.5, z_ticks, dtype=int)
	y_labels = np.linspace(0, n_z, z_ticks, dtype=int)

	# Var limits.
	#var_min = 1.0e5
	#var_max = 1.0e7
	var_min = np.nanmin(visc)
	var_max = np.nanmax(visc)

	cb_ticks = np.linspace(var_min, var_max, 6)
	
	for i in range(l-1, l, 1):

		# Update x_labels as domain extension changes in each iteration.
		x_labels  = np.linspace(0, L[i], n_ticks, dtype=int)
		
		fig = plt.figure(dpi=600, figsize=(6,4))
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)

		# Flip theta matrix so that the plot is not upside down.
		im = ax.imshow(np.flip(visc[i,:,:],axis=0), cmap='plasma', norm="log", \
						vmin=var_min, vmax=var_max, aspect='auto')
		#im = ax.imshow(np.flip(visc[i,:,:],axis=0), cmap='plasma', aspect='auto')
	
		ax.set_ylabel(r'$ \mathbf{n}_{z} $', fontsize=20)
		ax.set_xlabel(r'$\ \mathbf{x} \ (\mathrm{km})$', fontsize=20)

		divider = make_axes_locatable(ax)
		cax     = divider.append_axes("right", size="5%", pad=0.1)
		cb      = fig.colorbar(im, cax=cax, extend='neither')

		#cb.set_ticks(cb_ticks)
		#cb.set_ticklabels(list(cb_ticks), fontsize=14)

		cb.set_label(r'$\eta (x,z) \ (10^{6} \ \mathrm{Pa \cdot s})$', \
					 rotation=90, labelpad=6, fontsize=20)

		"""
		ax.set_xticks(x_ticks)
		ax.set_xticklabels(list(x_labels), fontsize=15)

		ax.set_yticks(y_ticks)
		ax.set_yticklabels(list(y_labels[::-1]), fontsize=15)
		"""

		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
		plt.tight_layout()

		if save_fig == True:
			##### Frame name ########
			if i < 10:
				frame = '000'+str(i)
			elif i > 9 and i < 100:
				frame = '00'+str(i)
			elif i > 99 and i < 1000:
				frame = '0'+str(i)
			else:
				frame = str(i)
			
			plt.savefig(path_fig+'flow_line_visc_'+frame+'.png', bbox_inches='tight')
		
		plt.show()
		plt.close(fig)
	


if save_u == 1:

	# Number of x ticks.
	n_ticks = 5
	x_ticks = np.linspace(0, n, n_ticks)
	n_z     = np.shape(visc)[1]
	z_ticks = int(0.2 * n_z + 1)

	# n_z-0.5 to avoid half of grid cell in black when plotting.
	y_ticks  = np.linspace(0, n_z-0.5, z_ticks, dtype=int)
	y_labels = np.linspace(0, n_z, z_ticks, dtype=int)

	# Var limits.
	#var_min = np.round(1e-6 * np.nanmin(visc), 0)
	#var_max = np.round(1e-6 * np.nanmax(visc), 0)
	#u_min = np.nanmin(u)
	#u_max = np.nanmax(u)

	#u_x_min = np.nanmin(u_x[s[0]-1])
	#u_x_max = np.nanmax(u_x[s[0]-1])
	u_min = 1.0
	u_max = 1.0e3

	#cb_ticks_u   = np.linspace(u_min, u_max, 6)
	#cb_ticks_u_z = np.round(np.linspace(u_z_min, u_z_max, 6), 4)

	ind_plot = np.array([0, int(0.5*s[0]), s[0]-1])
	
	for i in range(l-1, l, 1): # (l-1, l, 1), ind_plot

		# Update x_labels as domain extension changes in each iteration.
		x_labels  = np.linspace(0, L[i], n_ticks, dtype=int)
		

		# FIGURE FOR U_X.
		fig = plt.figure(dpi=600, figsize=(6,4))
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)

		# Flip theta matrix so that the plot is not upside down.
		#im = ax.imshow(np.flip(u[i,:,:],axis=0), cmap='plasma', \
		#				vmin=u_min, vmax=u_max, aspect='auto')
		im = ax.imshow(np.flip(np.abs(u[i,:,:]),axis=0), vmin=u_min, vmax=u_max,\
		 				 norm='log', cmap='viridis', aspect='auto')
	
	
		ax.set_ylabel(r'$ \mathbf{n}_{z} $', fontsize=20)
		ax.set_xlabel(r'$\ \mathbf{x} \ (\mathrm{km})$', fontsize=20)

		divider = make_axes_locatable(ax)
		cax     = divider.append_axes("right", size="5%", pad=0.1)
		cb      = fig.colorbar(im, cax=cax, extend='neither')

		#cb.set_ticks(cb_ticks_u)
		#cb.set_ticklabels(list(cb_ticks_u), fontsize=14)

		cb.set_label(r'$ u (x,z) \ ( \mathrm{m / yr})$', \
					 rotation=90, labelpad=6, fontsize=20)

		"""
		ax.set_xticks(x_ticks)
		ax.set_xticklabels(list(x_labels), fontsize=15)
		
		ax.set_yticks(y_ticks)
		ax.set_yticklabels(list(y_labels[::-1]), fontsize=15)
		"""
	
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
		plt.tight_layout()

		if save_fig == True:
			##### Frame name ########
			if i < 10:
				frame = '000'+str(i)
			elif i > 9 and i < 100:
				frame = '00'+str(i)
			elif i > 99 and i < 1000:
				frame = '0'+str(i)
			else:
				frame = str(i)
			
			plt.savefig(path_fig+'flow_line_visc_'+frame+'.png', bbox_inches='tight')
		
		plt.show()
		plt.close(fig)

if save_u_der == 1:

	# Number of x ticks.
	n_ticks = 5
	x_ticks = np.linspace(0, n, n_ticks)
	n_z     = np.shape(visc)[1]
	z_ticks = int(0.2 * n_z + 1)

	# n_z-0.5 to avoid half of grid cell in black when plotting.
	y_ticks  = np.linspace(0, n_z-0.5, z_ticks, dtype=int)
	y_labels = np.linspace(0, n_z, z_ticks, dtype=int)

	# Var limits.
	u_z_min = 1.0e-3
	u_z_max = 1.0e-1

	u_z = np.where(u_z < u_z_min, u_z_min, u_z)

	#cb_ticks_u   = np.linspace(u_min, u_max, 6)
	#cb_ticks_u_z = np.round(np.linspace(u_z_min, u_z_max, 6), 4)

	ind_plot = np.array([0, int(0.5*s[0]), s[0]-1])
	
	for i in range(l-1, l, 1): # (l-1, l, 1), ind_plot

		# FIGURE FOR U_Z.
		fig = plt.figure(dpi=600, figsize=(6,4))
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)

		# Flip theta matrix so that the plot is not upside down.
		im = ax.imshow(np.flip(u_z[i,:,:],axis=0), norm='log', cmap='PuOr', \
						vmin=u_z_min, vmax=u_z_max, aspect='auto')
		
		#im = ax.imshow(np.flip(lmbd[i,:,:],axis=0), cmap='cividis', aspect='auto')

		ax.set_ylabel(r'$ \mathbf{n}_{z} $', fontsize=20)
		ax.set_xlabel(r'$\ \mathbf{x} \ (\mathrm{km})$', fontsize=20)

		divider = make_axes_locatable(ax)
		cax     = divider.append_axes("right", size="5%", pad=0.1)
		cb      = fig.colorbar(im, cax=cax, extend='neither')

		#cb.set_ticks(cb_ticks_u_z)
		#cb.set_ticklabels(list(cb_ticks_u_z), fontsize=14)

		#cb.set_label(r'$ u_{z} (x,z) \ ( \mathrm{1 / yr})$', \
		#			 rotation=90, labelpad=6, fontsize=20)

		cb.set_label(r'$ u_{z} (x,z) \ ( \mathrm{1 / yr})$', \
					 rotation=90, labelpad=6, fontsize=20)
		"""
		ax.set_xticks(x_ticks)
		ax.set_xticklabels(list(x_labels), fontsize=15)

		ax.set_yticks(y_ticks)
		ax.set_yticklabels(list(y_labels[::-1]), fontsize=15)
		"""
	
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
		plt.tight_layout()

		if save_fig == True:
			##### Frame name ########
			if i < 10:
				frame = '000'+str(i)
			elif i > 9 and i < 100:
				frame = '00'+str(i)
			elif i > 99 and i < 1000:
				frame = '0'+str(i)
			else:
				frame = str(i)
			
			plt.savefig(path_fig+'flow_line_visc_'+frame+'.png', bbox_inches='tight')
		
		plt.show()
		plt.close(fig)



	#u_x_min = np.nanmin(u_x)
	#u_x_max = np.nanmax(u_x)

	u_x_min = 1.0e-5
	u_x_max = 1.0e-1

	u_x = np.where(u_x < u_x_min, u_x_min, u_x)


	for i in range(l-1, l, 1): # (l-1, l, 1), ind_plot

		# FIGURE FOR U_Z.
		fig = plt.figure(dpi=600, figsize=(6,4))
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)

		# Flip theta matrix so that the plot is not upside down.
		im = ax.imshow(np.flip(u_x[i,:,:],axis=0), norm='log', cmap='Spectral', \
						vmin=u_x_min, vmax=u_x_max, aspect='auto')
		
		#im = ax.imshow(np.flip(lmbd[i,:,:],axis=0), cmap='cividis', aspect='auto')

		ax.set_ylabel(r'$ \mathbf{n}_{z} $', fontsize=20)
		ax.set_xlabel(r'$\ \mathbf{x} \ (\mathrm{km})$', fontsize=20)

		divider = make_axes_locatable(ax)
		cax     = divider.append_axes("right", size="5%", pad=0.1)
		cb      = fig.colorbar(im, cax=cax, extend='neither')

		#cb.set_ticks(cb_ticks_u_z)
		#cb.set_ticklabels(list(cb_ticks_u_z), fontsize=14)

		#cb.set_label(r'$ u_{z} (x,z) \ ( \mathrm{1 / yr})$', \
		#			 rotation=90, labelpad=6, fontsize=20)

		cb.set_label(r'$ u_{x} (x,z) \ ( \mathrm{1 / yr})$', \
					rotation=90, labelpad=6, fontsize=20)
		"""
		ax.set_xticks(x_ticks)
		ax.set_xticklabels(list(x_labels), fontsize=15)

		ax.set_yticks(y_ticks)
		ax.set_yticklabels(list(y_labels[::-1]), fontsize=15)
		"""
	
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
		plt.tight_layout()

		if save_fig == True:
			##### Frame name ########
			if i < 10:
				frame = '000'+str(i)
			elif i > 9 and i < 100:
				frame = '00'+str(i)
			elif i > 99 and i < 1000:
				frame = '0'+str(i)
			else:
				frame = str(i)
			
			plt.savefig(path_fig+'flow_line_visc_'+frame+'.png', bbox_inches='tight')
	
	plt.show()
	plt.close(fig)





#############################################
#############################################
# TIME SERIES
if time_series_gif == 1:

	i_0 = 25

	# Plot bed peak position.
	y_p = np.full(len(t_plot), 350)

	for i in range(i_0, l, 1):

		# Figure.
		fig = plt.figure(dpi=600, figsize=(5.5,6))
		ax = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		ax3 = ax2.twinx()

		plt.rcParams['text.usetex'] = True

		# Plot bed peak position.
		ax3.plot(t_plot, y_p, linestyle='--', color='red', marker='None', \
				markersize=3.0, linewidth=1.5, alpha=1.0, label=r'$u_{b}(x)$') 

		# Grey time series in background.
		ax.plot(t_plot, A_s, linestyle='-', color='grey', marker='None', \
					markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		ax2.plot(t_plot, u_bar[:,s[2]-1], linestyle='-', color='grey', marker='None', \
					markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		ax3.plot(t_plot, L, linestyle='-', color='grey', marker='None', \
				markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		# Coloured line.
		ax.plot(t_plot[i_0:(i+1)], A_s[i_0:(i+1)], linestyle='-', color='darkgreen', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		ax2.plot(t_plot[i_0:(i+1)], u_bar[i_0:(i+1),s[2]-1], linestyle='-', color='blue', marker='None', \
					markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		ax3.plot(t_plot[i_0:(i+1)], L[i_0:(i+1)], linestyle='-', color='red', marker='None', \
				markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
		# Colour point at the current time step.
		ax.plot(t_plot[i], A_s[i], linestyle='None', color='darkgreen', marker='o', \
			 markersize=7.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		ax2.plot(t_plot[i], u_bar[i,s[2]-1], linestyle='None', color='blue', marker='o', \
					markersize=7.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		ax3.plot(t_plot[i], L[i], linestyle='None', color='red', marker='o', \
				markersize=7.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

		
		ax.set_ylabel(r'$ A \ (\mathrm{Pa}^{-3} \mathrm{yr}^{-1})$', fontsize=18)
		ax2.set_ylabel(r'$ \overline{u} \ (\mathrm{m / yr})$', fontsize=18)
		ax3.set_ylabel(r'$ L \ (\mathrm{km})$', fontsize=18)
		
		ax2.set_xlabel(r'$ \mathrm{Time} \ (\mathrm{kyr})$', fontsize=18)
	
			
		ax.yaxis.label.set_color('darkgreen')
		ax2.yaxis.label.set_color('blue')
		ax3.yaxis.label.set_color('red')
		
		ax.set_xticklabels([])
		#ax2.set_xticklabels([])

		ax.tick_params(axis='y', which='major', length=4, colors='darkgreen', labelsize=13)
		ax2.tick_params(axis='y', which='major', length=4, colors='blue', labelsize=13)
		ax2.tick_params(axis='x', which='major', length=4, colors='black', labelsize=13)
		ax3.tick_params(axis='y', which='major', length=4, colors='red', labelsize=13)
		
		ax.grid(axis='x', which='major', alpha=0.85)
		ax2.grid(axis='x', which='major', alpha=0.85)

		# Axis limits.
		ax.set_xlim(10.0, 40.0)
		ax2.set_xlim(10.0, 40.0)
		ax3.set_xlim(10.0, 40.0)

		ax3.set_ylim(280.0, 380.0)
		
		plt.tight_layout()

		##### Frame name ########
		if i < 10:
			frame = '000'+str(i)
		elif i > 9 and i < 100:
			frame = '00'+str(i)
		elif i > 99 and i < 1000:
			frame = '0'+str(i)
		else:
			frame = str(i)
		
		plt.tight_layout()
		
		if save_fig == True:
			plt.savefig(path_fig+'time_series_'+frame+'.png', bbox_inches='tight')
			print('Saved')

		# Display and close figure.
		plt.show()
		plt.close(fig)

#############################################
#############################################
# TIME SERIES GIF.

if save_series_frames == 1:
	
	t = 1.0e-3 * t
	t_plot = []
	L_plot = []
	H_plot = []
	dif_plot = []
	A_plot = []
		
	for i in range(l):
		
		fig = plt.figure(dpi=600, figsize=(5.5,6))
		ax = fig.add_subplot(311)
		ax2 = fig.add_subplot(312)
		ax4 = ax2.twinx()
		ax3 = fig.add_subplot(313)
		
		plt.rcParams['text.usetex'] = True
		
		ax.plot(t, L, linestyle='-', color='grey', marker='None', \
				markersize=3.0, linewidth=2.5, alpha=0.65, label=r'$u_{b}(x)$') 
		
		ax2.plot(t, H[:,n-1], linestyle='-', color='grey', marker='None', \
				 markersize=3.0, linewidth=2.5, alpha=0.65, label=r'$u_{b}(x)$') 
		
		
		ax4.plot(t, 1.0e3 * dif, linestyle='-', color='grey', marker='None', \
				 markersize=3.0, linewidth=2.5, alpha=0.65, label=r'$u_{b}(x)$')
		
		ax3.plot(t, 1.0e-6 * A, linestyle='-', color='grey', marker='None', \
				 markersize=3.0, linewidth=2.5, alpha=0.65, label=r'$u_{b}(x)$') 
		
		t_plot.append(t[i])
		L_plot.append(L[i])
		H_plot.append(H[i,n-1])
		dif_plot.append(1.0e3* dif[i])
		A_plot.append(1.0e-6 * A[i])
		
		ax.plot(t_plot, L_plot, linestyle='-', color='darkblue', marker='None', \
			markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
		ax2.plot(t_plot, H_plot, linestyle='-', color='black', marker='None', \
				 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
		ax4.plot(t_plot, dif_plot, linestyle='-', color='blue', marker='None', \
				 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$')
		
		ax3.plot(t_plot, A_plot, linestyle='-', color='red', marker='None', \
				 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
	
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ kyr$',\
					   fontsize=16)
		
		ax.set_ylabel(r'$L \ (km)$',fontsize=18)
		ax2.set_ylabel(r'$H_{c} \ (m)$',fontsize=18)
		ax4.set_ylabel(r'$\Delta u_{2} \ (10^{3} \ yr^{-1})$',fontsize=18)
		ax3.set_ylabel(r'$A \ (km^2)$',fontsize=18)
		ax3.set_xlabel(r'$\mathrm{Time} \ (kyr)$',fontsize=18)
		
		ax.set_yticks([800, 900, 1000, 1100])
		ax.set_yticklabels(['$800$', '$900$',\
						  '$1000$', '$1100$'], fontsize=12)
		ax2.set_yticks([1150, 1225, 1300, 1375])
		ax2.set_yticklabels(['$1150$', '$1225$', '$1300$', '$1375$'], fontsize=12)
		ax4.set_yticks([0.3, 0.325, 0.350, 0.375, 0.400])
		ax4.set_yticklabels(['$0.300$', '$0.325$', '$0.350$',\
						     '$0.375$', '$0.400$'], fontsize=12)
		ax3.set_yticks([1500, 1750, 2000, 2250])
		ax3.set_yticklabels(['$1500$', '$1750$', '$2000$','$2250$'], fontsize=12)
		
		ax3.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
		ax3.set_xticklabels(['$0$', '$1$', '$2$','$3$', \
						     '$4$', '$5$', '$6$','$7$'], fontsize=15)
		
		ax.set_xlim(t[0], t[l-1])
		ax2.set_xlim(t[0], t[l-1])
		ax3.set_xlim(t[0], t[l-1])
			
		ax.yaxis.label.set_color('darkblue')
		ax2.yaxis.label.set_color('black')
		ax3.yaxis.label.set_color('red')
		ax4.yaxis.label.set_color('blue')
		
		ax.set_xticklabels([])
		ax2.set_xticklabels([])
		
		ax.tick_params(axis='y', which='major', length=4, colors='darkblue')
		ax2.tick_params(axis='y', which='major', length=4, colors='black')
		ax3.tick_params(axis='y', which='major', length=4, colors='red')
		ax3.tick_params(axis='x', which='major', length=4, colors='black')
		ax4.tick_params(axis='y', which='major', length=4, colors='blue')
		
		ax.grid(axis='x', which='major', alpha=0.85)
		ax2.grid(axis='x', which='major', alpha=0.85)
		ax3.grid(axis='x', which='major', alpha=0.85)
		
		##### Frame name ########
		if i < 10:
			frame = '000'+str(i)
		elif i > 9 and i < 100:
			frame = '00'+str(i)
		elif i > 99 and i < 1000:
			frame = '0'+str(i)
		else:
			frame = str(i)
		
		plt.tight_layout()
		if save_fig == True:
			plt.savefig(path_fig+'time_series_gif_'+frame+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)



#############################################
#############################################
# GROUNDING LINE POSITION AS A FUNCTION OF GIVEN VARIABLES.

if save_L == 1:

	# Time points where we want to plot.
	t_L = np.arange(40.0e3, t[l-1], 3.0e4)

	# Find the indexes in the time vector that are closest to our t_L values.
	indices = np.abs(t[:, np.newaxis] - t_L).argmin(axis=0)

	col_1 = np.linspace(0.0, 1.0, l)

	fig = plt.figure(dpi=400)
	ax = fig.add_subplot(111)

	plt.rcParams['text.usetex'] = True

	for i in indices:

		# Current path.
		ax.plot(A_s[i], 1.0e-3*L[i], markeredgecolor=[col_1[i],0,0], marker='o', markeredgewidth=2, \
					markersize=7.0, markerfacecolor='None', alpha=1.0, label=r'$ \mathrm{Flowline} $') 



	ax.set_xlabel(r'$A \ (\mathrm{Pa}^{-3} \ \mathrm{s}^{-1}) $', fontsize=18)
	ax.set_ylabel(r'$L \ (\mathrm{km})$', fontsize=18)


	#ax.legend(loc='best', ncol = 1, frameon = True, framealpha = 1.0, \
	# 		  fontsize = 12, fancybox = True)


	ax.tick_params(axis='both', which='major', length=4, colors='black', labelsize=15)

	#ax.tick_params(axis='both', labelsize=20)


	ax.grid(axis='x', which='major', alpha=0.85)
	ax.grid(axis='x', which='minor', alpha=0.50)

	#ax.set_xlim(180, 1020)
	#ax.set_ylim(100, 400)

	plt.xscale('log')

	plt.tight_layout()

	if save_fig == True:
		plt.savefig(path_fig+'gl_A.png', bbox_inches='tight')

	plt.show()
	plt.close(fig)

	

if save_series_2D == 1:

	# Number of ticks.
	n_x = 3
	n_z = 6

	# Theta limits.
	u_bar_min = 0.0
	u_bar_max = 3.0

	H_min = 0.0
	H_max = 3.0

	# Colourbar ticks.
	cb_ticks_1  = [0.0, 1.0, 2.0, 3.0]
	cb_labels_1 = [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$']

	cb_ticks_2  = [0.0, 1.0, 2.0, 3.0]
	cb_labels_2 = [r'$0$', r'$1$', r'$2$', r'$3$']

	u_bar = np.where(u_bar>0, u_bar, 1.0e-3)
	u_bar = np.log10(u_bar)


	# Update x_labels as domain extension changes in each iteration.
	#x_labels  = np.linspace(0, L[i], n_ticks, dtype=int)
	x_ticks = np.linspace(0, s[2], n_x)
	x_labels = [r'$0$', r'$L/2$', r'$L$', ]

	y_ticks  = np.linspace(0, s[0]-1, n_z, dtype=int)
	y_labels = [r'$0$', r'$8$', r'$16$', r'$24$', r'$32$', r'$40$']

	# Colourmap.
	cmap = plt.get_cmap("RdYlBu") # spectral, rainbow, jet, turbo
	reversed_cmap = cmap.reversed()

	cmap_2 = plt.get_cmap("PRGn") # spectral, rainbow, jet, turbo
	#reversed_cmap = cmap.reversed()
	
	fig = plt.figure(dpi=600, figsize=(8,6))
	plt.rcParams['text.usetex'] = True
	ax1  = fig.add_subplot(121)
	ax2  = fig.add_subplot(122)

	# Flip theta matrix so that the plot is not upside down.
	im1 = ax1.imshow(np.flip(u_bar,axis=0), cmap=reversed_cmap, \
					vmin=u_bar_min, vmax=u_bar_max, aspect='auto', interpolation='bilinear')
	
	im2 = ax2.imshow(np.flip(H,axis=0), cmap=cmap_2, \
					vmin=H_min, vmax=H_max, aspect='auto', interpolation='bilinear')

	ax1.set_ylabel(r'$ t \ (\mathrm{kyr}) $', fontsize=20)
	#ax1.set_xlabel(r'$ x \ (\mathrm{km})$', fontsize=20)
	#ax2.set_xlabel(r'$ x \ (\mathrm{km})$', fontsize=20)

	
	divider_1 = make_axes_locatable(ax1)
	cax_1     = divider_1.append_axes("right", size="10%", pad=0.2, aspect=5)
	cb_1      = fig.colorbar(im1, cax=cax_1, extend='neither', orientation='vertical')

	cb_1.set_ticks(cb_ticks_1)
	cb_1.set_ticklabels(cb_labels_1, fontsize=14)

	cb_1.set_label(r'$ \overline{u} (x,t) \ (\mathrm{m/yr})$', \
					rotation=90, labelpad=6, fontsize=20)
	
	
	divider_2 = make_axes_locatable(ax2)
	cax_2     = divider_2.append_axes("right", size="10%", pad=0.2, aspect=5)
	cb_2      = fig.colorbar(im2, cax=cax_2, extend='neither', orientation='vertical')

	cb_2.set_ticks(cb_ticks_2)
	cb_2.set_ticklabels(cb_labels_2, fontsize=14)

	cb_2.set_label(r'$ H (x,t) \ (\mathrm{km})$', \
					rotation=90, labelpad=6, fontsize=20)

	ax1.set_xticks(x_ticks)
	ax1.set_xticklabels(list(x_labels), fontsize=15)

	ax2.set_xticks(x_ticks)
	ax2.set_xticklabels(list(x_labels), fontsize=15)

	ax1.set_yticks(y_ticks)
	ax1.set_yticklabels(y_labels[::-1], fontsize=15)

	ax2.set_yticks(y_ticks)
	ax2.set_yticklabels([], fontsize=15)

	
	ax1.set_title(r'$ \mathrm{(a)} $', fontsize=16)
	ax2.set_title(r'$ \mathrm{(b)} $', fontsize=16)


	plt.tight_layout()

	if save_fig == True:

		plt.savefig(path_fig+'flow_line_theta.png', bbox_inches='tight')
		
	plt.show()
	plt.close(fig)


# We perform fft for number of variable time series to study the 
# main periodicity in oscillatory Nix sims.
if heat_map_fourier == 1:

	# Method of calculating periodicity.
	meth_period = 'local_max'

	# Fast Fourier Transform.
	def fft(x):
		
		# FFT.
		fft_x = np.fft.fft(x) 

		# Power spectrum.
		fft_x_plot = np.abs(fft_x)**2

		# Only real frequencies.
		#n = int(0.5*len(x)-1)
		n = len(x)

		# Frequencies given a time series with n points.
		nu = np.fft.fftfreq(n)

		# Save period and corresponding power spetrum.
		nu_new = 1.0 / nu[1:n]
		f = fft_x_plot[1:n]

		return [nu_new,f]

	# Parent folder.
	parent_folder = '/home/dmoreno/nix/oscillations/C_thw-w_0/'

	var_fft   = ['u_bar']
	l_var_fft = len(var_fft)

	# List all subfolders in the parent folder
	subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

	# Lists.
	fft_tot    = []
	nu_tot_fft = []
	u_bar_all  = []
	nu_max     = []
	local_max_all = []
	T_mean_all   = []


	# Loop through each subfolder.
	#subfolders = ['/home/dmoreno/nix/oscillations/C_thw-w_0/w0.0.5_C_thw.0.75_nz.15/']
	
	for subfolder in subfolders:

		print('Exp = ', subfolder)
		
		# Define the path to the netCDF file in the current subfolder
		path_nc = os.path.join(get_datadir(), subfolder, 'nix_hr.nc')

		# Check if the file exists before attempting to open it
		if os.path.exists(path_nc):
			
			# Open the netCDF file
			data = Dataset(path_nc, mode='r')

			# Load variables of interest.
			t     = data.variables['t'][:]
			u_bar = data.variables['u_bar'][:]

			# Append values of all experiments.
			#u_bar = np.sin(4*t) + np.sin(2*t)
			u_bar_all.append(u_bar)

		# Define dimensions.
		s = np.shape(u_bar)
		
		# Set beginning and end of the time series. Evaluated at the GL.
		n_t0  = int(0.5*s[0])         # 0.5
		var   = u_bar[n_t0:(s[0]-1)]
		l_var = len(var)

		###########################################################################
		###########################################################################
		# METHOD 1: FFT.
		if meth_period == 'fft':
			# Call FFT function and save period and transformed function to be plotted.
			[nu_fft, u_fft_plot] = fft(var) 

			# Append FFT series, frequencies and main frequencies. 
			fft_tot.append(u_fft_plot)
			nu_tot_fft.append(nu_fft)
			nu_max.append(nu_fft[np.argmax(u_fft_plot)])

			# We convert to real frequencies and then to period.
			# In FFT, the frequency depends on the number of given points.
			# nu_max_real returns w_0 in f(t) = sin(w_0 * t). T = 2*pi/w_0.
			nu_max_real = l_var / np.array(nu_max)
			T_real      = 2.0 * np.pi / nu_max_real

		
		###########################################################################
		###########################################################################
		# METHOD 2: Local maxima.
		elif meth_period == 'local_max':

			# A point will be considered a local maximum if it is greater 
			# than its 15 neighboring points on each side (total of 11 points, including the point itself).
			thresh = 400
			local_max = argrelextrema(u_bar, np.greater, order=thresh)[0] # argrelextrema returs a tuple.
			l_max     = len(local_max)

			# Maximum velocity to find surges.
			u_max = np.max(u_bar)
			print('u_max = ', u_max)
			
			# If the sim does not oscillate
			if u_max < 2.0e3:
			#if l_max < 10:
				T_mean       = np.nan
				local_max_eq = np.nan

			else:
				# Use boolean indexing to keep values above the threshold.
				# We delete all indexes value below n_t0 to focus on the equilibrium.
				local_max_eq = local_max[local_max >= n_t0]
				l_local_max  = len(local_max_eq) 

				# Find local maxima.
				T = np.empty(l_local_max-1)
				n0_local_max = int(0.5*l_local_max)
				nf_local_max = l_local_max-1

				# Distance between two consequtive peaks.
				for i in range(n0_local_max, nf_local_max, 1):

					T[i] = t[local_max_eq[i+1]] - t[local_max_eq[i]]
			
				# Mean value of the spacing among peak gives us the surge preiodicity.
				T_mean = 1.0e-3 * np.mean(T[n0_local_max:nf_local_max])

			# Append results.
			local_max_all.append(local_max_eq)
			T_mean_all.append(T_mean)

			
			print('T_mean = ', T_mean)


	# Reshape for heat map.
	# Rows: w_0 values. Columns: C_thw values.
	# Define the number of rows and columns
	num_rows = 5
	num_cols = 10 #11

	T_mean_all = np.reshape(T_mean_all, [num_rows,num_cols])

	print('T_mean_all = ', T_mean_all)


	# PLOTS.
	# Run over all subfolders.
	plot = True

	#for i in range(len(subfolders)):

	if plot == True:

		total_panels = num_rows * num_cols

		# Create a figure with the specified number of panels
		fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10), dpi=800)
		
		c = 0

		# Iterate through each panel and customize as needed
		for i in range(num_rows):
			for j in range(num_cols):
				
				# Customize each panel if necessary
				axs[i,j].set_title(f'u_max = {np.round(np.max(u_bar_all[i]),0)}')

				# Original time series.
				axs[i,j].plot(t[n_t0:(s[0]-1)], u_bar_all[i][n_t0:(s[0]-1)], linestyle='-', color='blue', \
							marker='None', linewidth=1.5, markersize=4, alpha=1.0)
				
				# Local maxima.
				if np.size(local_max_all[i]) != 1:
					
					axs[i,j].plot(t[local_max_all[i]], u_bar_all[i][local_max_all[i]], linestyle='None', color='black', \
								marker='o', linewidth=1.5, markersize=4, alpha=1.0)

				
				axs[i,j].set_xticklabels([])
				#axs[i,j].set_yticklabels([])

				c += 1

		# FFT series.
		"""ax1.plot(nu_tot_fft[i], fft_tot[i], linestyle='-', color='blue', \
					marker='o', linewidth=1.5, markersize=4, alpha=1.0)"""

		"""ax2.plot(u_bar_all[i][n_t0:(s[0]-1),s[2]-1], linestyle='-', color='red', \
					marker='o', linewidth=1.5, markersize=4, alpha=1.0)"""

		
		"""# Original time series.
		ax2.plot(t[n_t0:(s[0]-1)], u_bar_all[i][n_t0:(s[0]-1)], linestyle='-', color='red', \
					marker='None', linewidth=1.5, markersize=4, alpha=1.0)
		
		# Local maxima.
		if np.isnan(local_max_all[i]) == False:
			
			ax2.plot(t[local_max_all[i]], u_bar_all[i][local_max_all[i]], linestyle='None', color='black', \
						marker='o', linewidth=1.5, markersize=4, alpha=1.0)"""


		#ax1.set_xlim(0,100.0)

		plt.tight_layout()

		if save_fig == True:
			plt.savefig(path_fig+'oscillations.png', bbox_inches='tight')

		plt.show()
		plt.close(fig)




	
	
	# Theta limits.
	T_min = 0.0
	T_max = 2.0

	
	# Colourbar ticks.
	cb_ticks_1  = [0.0, 0.5, 1.0, 1.5, 2.0]
	cb_labels_1 = [r'$0.0$', r'$0.5$', r'$1.0$', r'$1.5$', r'$2.0$']

	cb_ticks_2  = [0.0, 1.0, 2.0, 3.0]
	cb_labels_2 = [r'$0$', r'$1$', r'$2$', r'$3$']


	
	# Update x_labels as domain extension changes in each iteration.
	#x_labels  = np.linspace(0, L[i], n_ticks, dtype=int)
	"""x_ticks = np.linspace(0, s[2], n_x)
	x_labels = [r'$0$', r'$L/2$', r'$L$', ]

	y_ticks  = np.linspace(0, s[0]-1, n_z, dtype=int)
	y_labels = [r'$0$', r'$6$', r'$12$', r'$18$', r'$24$', r'$30$']"""

	# Colourmap.
	cmap = plt.get_cmap("RdYlBu") # spectral, rainbow, jet, turbo
	reversed_cmap = cmap.reversed()

	cmap_2 = plt.get_cmap("PRGn") # spectral, rainbow, jet, turbo
	#reversed_cmap = cmap.reversed()

	fig = plt.figure(dpi=600, figsize=(8,6))
	plt.rcParams['text.usetex'] = True
	ax1  = fig.add_subplot(121)
	ax2  = fig.add_subplot(122)

	# Flip theta matrix so that the plot is not upside down.
	im1 = ax1.imshow(np.flip(T_mean_all,axis=0), cmap=reversed_cmap, \
					vmin=T_min, vmax=T_max, aspect='auto', \
						interpolation='none')

	#im2 = ax2.imshow(np.flip(H,axis=0), cmap=cmap_2, \
	#				vmin=H_min, vmax=H_max, aspect='auto', interpolation='bilinear')

	ax1.set_ylabel(r'$ w_0 \ (\mathrm{m/yr}) $', fontsize=20)
	ax1.set_xlabel(r'$ C_{\mathrm{thw}} $', fontsize=20)


	divider_1 = make_axes_locatable(ax1)
	cax_1     = divider_1.append_axes("right", size="10%", pad=0.2, aspect=5)
	cb_1      = fig.colorbar(im1, cax=cax_1, extend='neither', orientation='vertical')

	cb_1.set_ticks(cb_ticks_1)
	cb_1.set_ticklabels(cb_labels_1, fontsize=14)

	cb_1.set_label(r'$ \overline{u} (x,t) \ (\mathrm{m/yr})$', \
					rotation=90, labelpad=6, fontsize=20)

	"""
	divider_2 = make_axes_locatable(ax2)
	cax_2     = divider_2.append_axes("right", size="10%", pad=0.2, aspect=5)
	cb_2      = fig.colorbar(im2, cax=cax_2, extend='neither', orientation='vertical')

	cb_2.set_ticks(cb_ticks_2)
	cb_2.set_ticklabels(cb_labels_2, fontsize=14)

	cb_2.set_label(r'$ H (x,t) \ (\mathrm{km})$', \
					rotation=90, labelpad=6, fontsize=20)

	ax1.set_xticks(x_ticks)
	ax1.set_xticklabels(list(x_labels), fontsize=15)

	ax2.set_xticks(x_ticks)
	ax2.set_xticklabels(list(x_labels), fontsize=15)

	ax1.set_yticks(y_ticks)
	ax1.set_yticklabels(y_labels[::-1], fontsize=15)

	ax2.set_yticks(y_ticks)
	ax2.set_yticklabels([], fontsize=15)"""


	ax1.set_title(r'$ \mathrm{(a)} $', fontsize=16)
	ax2.set_title(r'$ \mathrm{(b)} $', fontsize=16)


	plt.tight_layout()

	if save_fig == True:

		plt.savefig(path_fig+'nix_period_oscillations.png', bbox_inches='tight')
		
	plt.show()
	plt.close(fig)

