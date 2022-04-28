#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:40:49 2021

@author: dmoren07
"""


from __future__ import division
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from dimarray import read_nc, get_datadir
import os
plt.style.use("seaborn-white")


path_fig  = '/home/dmoren07/figures/theoretical_results/frames/flow.line.cpp/mismip/n.1000/'
path_now = '/home/dmoren07/c++/output/mismip.exp.3/n.1000/taub.min/taub.min.26.0e3/asymmetric.L/'


# Select plots to be saved (boolean integer).
save_series        = 1
save_series_comp   = 0
save_shooting      = 0
save_domain        = 0
save_var_frames    = 0
save_series_frames = 0
save_theta         = 0
	

new = 0
if new == 1:
	os.makedirs(path_fig)

nc_SSA = os.path.join(get_datadir(), path_now+'eps.1.0e-30.L.p.4.nc')
data   = Dataset(nc_SSA, mode='r')
		
u     = data.variables['u'][:]
u2    = data.variables['du_dx'][:]
H     = data.variables['H'][:]
visc  = data.variables['visc'][:]
# S     = data.variables['S'][:]
tau_b = 1.0e-3 * data.variables['tau_b'][:]
tau_d = 1.0e-3 * data.variables['tau_d'][:]
L     = 1.0e-3 * data.variables['L'][:]
t     = data.variables['t'][:]
b     = data.variables['b'][:]
C_bed = data.variables['C_bed'][:]
u2_bc = data.variables['dudx_bc'][:]
dif   = data.variables['BC_error'][:]
u2_0_vec   = data.variables['u2_0_vec'][:]
u2_dif_vec = data.variables['u2_dif_vec'][:]
picard_error = data.variables['picard_error'][:]
c_picard = data.variables['c_picard'][:]
dt = data.variables['dt'][:]
alpha = data.variables['alpha'][:]
omega_picard = data.variables['omega'][:]
theta = data.variables['theta'][:] - 273.15


l = len(t)
s = np.shape(theta)

# GENEREAL PARAMETERS
sec_year = 3.154e7
#sec_mnth = 2.628e6

# f_cb
T       = 50.0       # years  
omega   = 2.0 * np.pi / T     # Real period is 0.5 * omega due to abs(cos(omega*t))
x_omega = 5.0e3               # m  

# f_visc
n_gln = 3
A     = 4.9e-25               # 4.9e-25 (T=-10ºC) # Pa³ / s (Greve and Blatter, 2009)
B     = A**( -1 / n_gln )     # Pa³ / s  
eps   = 1.0e-12**2             # 1.0e-21



# Number of points and domain.
n = s[2]
x_plot = np.linspace(0, 2000, n)


# MISMIP bedrock experiments.
exp = 1

x_tilde = x_plot / 750.0   # in km.

if exp == 1:
	bed = 720 - 778.5 * x_tilde
elif exp == 3:                    
	bed = ( 729.0 - 2184.8 * x_tilde**2 + \
			        + 1031.72 * x_tilde**4 + \
					- 151.72 * x_tilde**6 )

# Sea level function.
sl    = np.empty(n)
sl[:] = 0.0


#############################################
#############################################
# TIME SERIES
if save_series == 1:
	
	t_plot = 1.0e-3 * t
	
	fig = plt.figure(dpi=600, figsize=(5.5,6))
	ax = fig.add_subplot(311)
	ax6 = ax.twinx()
	ax2 = fig.add_subplot(312)
	ax4 = ax2.twinx()
	ax3 = fig.add_subplot(313)
	ax5 = ax3.twinx()
	
	plt.rcParams['text.usetex'] = True
	
	ax.plot(t_plot, L, linestyle='-', color='darkblue', marker='None', \
			markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
	ax2.plot(t_plot, H[:,n-1], linestyle='-', color='black', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
		
	A   = np.empty(l)
	u_L = np.empty(l)
	u2_0 = np.empty(l)
	
	for i in range(l):
		# delta_x times H sum over all points.
		A[i]   = (1.0e3 * L[i] / n) * np.sum(H[i,:])
		
		# Current u2 value minus analytical.
		u2_0[i] = u2[i,0] # u2[i,n-1]
		
		# Ice velocity at the GL.
		u_L[i] = u[i,n-1]
		
	
	#ax4.plot(t, dif, linestyle='-', color='purple', marker='None', \
	#		 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$')
	ax4.plot(t_plot, picard_error, linestyle='-', color='red', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$')
	
	ax3.plot(t_plot, 1.0e-6 * A, linestyle='-', color='brown', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
	ax5.plot(t_plot, u_L, linestyle='-', color='blue', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
	#ax6.plot(t, u2_0, linestyle='-', color='darkgreen', marker='None', \
	#		 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	ax6.plot(t_plot, u2[:,n-1], linestyle='-', color='darkgreen', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
	
	ax.set_ylabel(r'$L \ (km)$',fontsize=18)
	ax2.set_ylabel(r'$H_{gl} \ (m)$',fontsize=18)
	#ax4.set_ylabel(r'$\Delta u_{2}(L) \ (1/yr)$',fontsize=18)
	ax4.set_ylabel(r'$ \phi_{\mathrm{tol}} $',fontsize=18)
	ax3.set_ylabel(r'$A \ (km^2)$',fontsize=18)
	ax5.set_ylabel(r'$u(L) \ (m/yr)$',fontsize=18)
	ax6.set_ylabel(r'$u_{2}(L) \ (1/yr)$',fontsize=18)
	ax3.set_xlabel(r'$\mathrm{Time} \ (kyr)$',fontsize=18)
	
	ax.set_xlim(t_plot[0], t_plot[l-1])
	ax2.set_xlim(t_plot[0], t_plot[l-1])
	ax3.set_xlim(t_plot[0], t_plot[l-1])
		
	ax.yaxis.label.set_color('darkblue')
	ax2.yaxis.label.set_color('black')
	ax3.yaxis.label.set_color('brown')
	ax4.yaxis.label.set_color('red')
	ax5.yaxis.label.set_color('blue')
	ax6.yaxis.label.set_color('darkgreen')
	
	ax.set_xticklabels([])
	ax2.set_xticklabels([])
	
	ax.tick_params(axis='y', which='major', length=4, colors='darkblue')
	ax2.tick_params(axis='y', which='major', length=4, colors='black')
	ax3.tick_params(axis='y', which='major', length=4, colors='brown')
	ax4.tick_params(axis='y', which='major', length=4, colors='red')
	ax5.tick_params(axis='y', which='major', length=4, colors='blue')
	ax6.tick_params(axis='y', which='major', length=4, colors='darkgreen')
	
	ax.grid(axis='x', which='major', alpha=0.85)
	ax2.grid(axis='x', which='major', alpha=0.85)
	ax3.grid(axis='x', which='major', alpha=0.85)
	
	plt.tight_layout()
	plt.savefig(path_fig+'time_series.png', bbox_inches='tight')
	plt.show()
	plt.close(fig)




#############################################
#############################################
# TIME SERIES COMPUTATIONAL PERFORMANCE
if save_series_comp == 1:
	
	t_plot = 1.0e-3 * t
	
	fig = plt.figure(dpi=600, figsize=(5.5,6))
	ax = fig.add_subplot(311)
	ax6 = ax.twinx()
	ax2 = fig.add_subplot(312)
	ax4 = ax2.twinx()
	ax3 = fig.add_subplot(313)
	ax5 = ax3.twinx()
	
	plt.rcParams['text.usetex'] = True
	
	ax.plot(t_plot, c_picard, linestyle='-', color='darkblue', marker='None', \
			markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
	ax6.plot(t_plot, visc[:,n-1], linestyle='-', color='purple', marker='None', \
			markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
	ax2.plot(t_plot, dt, linestyle='-', color='black', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

	ax4.plot(t_plot, picard_error, linestyle='-', color='red', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$')
	
	ax3.plot(t_plot, omega_picard, linestyle='-', color='blue', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
		
	ax5.plot(t_plot, alpha, linestyle='-', color='black', marker='None', \
			 markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
	
	
	ax.set_ylabel(r'$C_{\mathrm{pic}}$',fontsize=18)
	ax2.set_ylabel(r'$ \Delta t \ (yr)$',fontsize=18)
	ax4.set_ylabel(r'$ \phi_{\mathrm{tol}}$',fontsize=18)
	ax3.set_ylabel(r'$ \omega \ (\mathrm{rad}) $',fontsize=18)
	ax5.set_ylabel(r'$ \alpha $',fontsize=18)
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
	plt.savefig(path_fig+'time_series.png', bbox_inches='tight')
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
	
		ax.plot(u2_0_vec[i,0:n_c], u2_dif_vec[i,0:n_c], linestyle='-', color='purple', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$sl(x)$') 
		
		ax2.plot(u2_0_vec[i,0:n_c], linestyle='-', color='darkgreen', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$z_s(x)$') 
		ax3.plot(u2_dif_vec[i,0:n_c], linestyle='-', color='purple', marker='None', \
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
		
		
		ax.set_xlim(np.nanmin(u2_0_vec), np.nanmax(u2_0_vec))
		ax.set_ylim(np.nanmin(u2_dif_vec), np.nanmax(u2_dif_vec))
		ax2.set_xlim(0, n_c)
		ax2.set_ylim(np.nanmin(u2_0_vec), np.nanmax(u2_0_vec))
		ax3.set_ylim(np.nanmin(u2_dif_vec), np.nanmax(u2_dif_vec))
	 	
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
		#plt.savefig(path_fig+'shooting_iter_'+frame+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)
		print('Saved')




#######################################
#######################################
# ENTIRE DOMAIN

if save_domain == 1:
	
	for i in range(l):
		
		L_plot  = np.linspace(0, L[i], n)
		x_tilde = L_plot / 750.0  
		
		if exp == 1:
			bed_L = 720 - 778.5 * x_tilde
		elif exp == 3:                    
			bed_L = ( 729.0 - 2184.8 * x_tilde**2 + \
					        + 1031.72 * x_tilde**4 + \
							- 151.72 * x_tilde**6 )
		
		# Ice surface elevation
		z_s = H[i,:] + bed_L
		
		# Vertical gray line ice front.
		frnt      = np.arange(bed_L[n-1], z_s[n-1], 1)
		l_frnt    = len(frnt)
		frnt_L    = np.empty(l_frnt)
		frnt_L[:] = L[i]
		
		# Ocean
		bed_p = np.where(x_plot > L[i], bed, np.nan)
		sl[:] = 0.0
		sl    = np.where(x_plot > L[i], sl, np.nan)
	
		
		fig = plt.figure(dpi=400) # (5,7)
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)
	
		ax.plot(x_plot, sl, linestyle='-', color='blue', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$sl(x)$') 
		ax.plot(frnt_L, frnt, linestyle='-', color='darkgrey', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$z_s(x)$') 
		ax.plot(x_plot, bed, linestyle='-', color='brown', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$bed(x)$') 
		
		ax.plot(L_plot, z_s, linestyle='-', color='darkgrey', marker='None', \
	  			linewidth=2.0, alpha=1.0, label=r'$z_s(x)$')  
	
		
		# Shade colours.
		ax.fill_between(x_plot, bed_p, 0.0,\
	 						   facecolor='blue', alpha=0.4)
		ax.fill_between(x_plot, bed, -2.5e3,\
	 						   facecolor='brown', alpha=0.4)
		ax.fill_between(L_plot, bed_L, z_s,\
	 						   facecolor='grey', alpha=0.4)
	
	
	
		ax.set_ylabel(r'$z(x) \ (km)$', fontsize=20)
		ax.set_xlabel(r'$x \ (km) $', fontsize=20)
	
		ax.yaxis.label.set_color('black')
	 	
		ax.tick_params(axis='both', which='major', length=4, colors='black')
	
		ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750])
		ax.set_xticklabels(['$0$', '$250$', '$500$', '$750$',\
						  '$1000$', '$1250$','$1500$', '$1750$'], fontsize=15)
		ax.set_yticks([-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000])
		ax.set_yticklabels(['$-1$', '$0$', '$1$',\
						  '$2$', '$3$','$4$','$5$','$6$'], fontsize=15)
	
		ax.set_xlim(0, 1750)
		ax.set_ylim(-1000, 6000)
		
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
		#plt.savefig(path_fig+'flow_line_mismip_exp.1_'+frame+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)
		print('Saved')




#############################################
#############################################
# VARIABLES FRAMES

if save_var_frames == 1:
	
	for i in range(l):
		
		L_plot  = np.linspace(0, L[i], n)
		x_tilde = L_plot / 750.0  
		bed_L   = ( 729.0 - 2184.8 * x_tilde**2 + \
			              + 1031.72 * x_tilde**4 + \
					      - 151.72 * x_tilde**6 )
		
			
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
		 
	
		ax.plot(L_plot, u[i,:], linestyle='-', color='blue', marker='None', \
	 			linewidth=2.0, alpha=1.0, label=r'$u_{b}(x)$') 
		ax5.plot(L_plot, u2[i,:], linestyle='-', color='darkgreen', marker='None', \
	 			linewidth=2.0, alpha=1.0, label=r'$\partial u_{b}/\partial x$')  
		ax3.plot(L_plot, visc[i,:], linestyle='-', color='purple', marker='None', \
	  	 		linewidth=2.0, alpha=1.0, label=r'$\partial H/\partial x$') 
		ax4.plot(L_plot, tau_d[i,:], linestyle='-', color='brown', marker='None', \
	  	 		linewidth=2.0, alpha=1.0, label=r'$\tau_{d} $') 
		ax2.plot(L_plot, H[i,:], linestyle='-', color='black', marker='None', \
	   			linewidth=2.0, alpha=1.0, label=r'$H(x)$')  
		ax6.plot(L_plot, tau_b[i,:], linestyle='-', color='red', marker='None', \
	 			linewidth=2.0, alpha=1.0, label=r'$\tau_{b}(x)$')
		#ax4.plot(L_plot, bed_L, linestyle='-', color='brown', marker='None', \
	 	#		linewidth=2.0, alpha=1.0, label=r'$\tau_{d}(x)$') 
		#ax4.plot(L_plot, C_bed[i,:], linestyle='-', color='brown', marker='None', \
	 	#		linewidth=2.0, alpha=1.0, label=r'$\tau_{d}(x)$') 
	
	
		ax.set_ylabel(r'$u_{b}(x) \ (m/yr)$',fontsize=16)
		ax3.set_ylabel(r'$\eta(x)\ (Pa \cdot s)$',fontsize=16)
		ax4.set_ylabel(r'$\tau_{d} \ (kPa)$',fontsize=16)
		#ax4.set_ylabel(r'$C(x) \ (Pa \ (s/m)^{1/3} )$',fontsize=16)
		ax5.set_ylabel(r'$\partial u_{b}/\partial x $',fontsize=16)
		ax2.set_ylabel(r'$H(x) \ (m)$', fontsize=16)
		ax6.set_ylabel(r'$\tau_{b}(x) \ (kPa)$', fontsize=16)
		ax3.set_xlabel(r'$x \ (km) $',fontsize=16)
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
		#ax.set_ylim(0, 350)
	# 	ax2.set_ylim(100,1100) 
	# 	ax5.set_ylim(-10,10)
		 
		#ax.set_title(r'$i = \ $'+str(i)+r', \ t = \ '+str(round(t_vec[i],1))+r' yr', \
		#			  fontsize=16)
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
		#plt.savefig(path_fig+'flow_line_var_'+frame+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)
		print('Saved')
		
	


#############################################
#############################################
# TEMPERATURE FRAMES

# Number of desired horizontal points.
n_x = 50

theta_avg = np.empty([s[1], n_x])
tens      = np.arange(0, s[2], np.int(s[2]/n_x))

if save_theta == 1:
	
	for i in range(l):
		
		L_plot  = np.linspace(0, L[i], n)
		x_tilde = L_plot / 750.0  
		bed_L   = ( 729.0 - 2184.8 * x_tilde**2 + \
			              + 1031.72 * x_tilde**4 + \
					      - 151.72 * x_tilde**6 )
		
		# Horizontal average every n_x points for visualization.
		for j in range(n_x):
			for k in range(s[1]):
				theta_avg[k,j] = np.mean(theta[i,k,tens[j]:tens[j]+n_x])
		
		theta_avg = np.fliplr(theta_avg)
		
		######################################
		######################################
		# L PLOTS.
		
		fig = plt.figure(dpi=600, figsize=(6,5.5))
		plt.rcParams['text.usetex'] = True
		ax  = fig.add_subplot(111)

		 
		im = ax.imshow(np.rot90(theta_avg,2), cmap='plasma')
		#ax.plot(L_plot, theta[i,0,:], linestyle='-', color='blue', marker='None', \
	 	#		linewidth=2.0, alpha=1.0, label=r'$u_{b}(x)$') 
		#ax.contour(L_plot, theta[i,0,:], linestyle='-', color='blue', marker='None', \
	 	#		linewidth=2.0, alpha=1.0, label=r'$u_{b}(x)$') 
	
		ax.set_ylabel(r'$\theta (x) \ (C)$',fontsize=16)
		#cax   = fig.add_axes([0.95, 0.04, 0.03, 0.92])
		cax   = fig.add_axes([0.99, 0.32, 0.03, 0.40])
		cb    = fig.colorbar(im, cax=cax,extend='neither')

	
		ax.set_title(r'$i = \ $'+str(i)+r'$, \ t =  \ $'+str(np.round(t[i],2))+r'$ \ yr$', fontsize=16)
	 	

			
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
		plt.savefig(path_fig+'flow_line_theta_'+frame+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)
		print('Saved')
		
	


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
		plt.savefig(path_fig+'time_series_gif_'+frame+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)



#############################################
#############################################
 	
# Gaussian filter test:
# sigma = 5.0


# f = np.array([0, 2, 5, 3, 3, 5, 6, 7, 10, 15, 8, 0])

# def gauss(f, sigma):
#  	l = len(f)
#  	summ  = np.zeros(l)
#  	dy = 2.0
 	
#  	A = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
 	
#  	# Weierstrass transform
#  	for i in range(l):
#  	 	x = i * dy
#  	 	for j in range(l):
# 			  y = j * dy
# 			  summ[i] = summ[i] + f[j] * np.exp( - ( (x - y) / sigma )**2  / 2.0 ) * dy
 	
#  	F = A * summ
 	
#  	return F
 	


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# plt.rcParams['text.usetex'] = True

# ax.plot(f, linestyle='-', color='black', marker='None', \
#  	markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

# sigmas = np.array([1.0, 2.0, 3.0, 4.0, 10.0])

# for i in sigmas:
#  	F = gauss(f, i)
#  	ax.plot(F, linestyle='-', color='darkblue', marker='None', \
#  	markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

# ax.set_ylabel(r'$Smooth$',fontsize=18)


# # ax.set_yticks([800, 900, 1000, 1100])
# # ax.set_yticklabels(['$800$', '$900$',\
# # 				  '$1000$', '$1100$'], fontsize=12)


# ax.yaxis.label.set_color('darkblue')


# ax.tick_params(axis='y', which='major', length=4, colors='darkblue')


# ax.grid(axis='x', which='major', alpha=0.85)

# plt.tight_layout()

# plt.show()
# plt.close(fig)


#############################################
#############################################
sec_year = 3.154e7
 	
# Viscosity dependence on T and du/dx
eps  = 1.0e-7
dudx = np.arange(-1.0e-5, 1.0e-5, 1.0e-7)
#eps  = 1.0e-11
#dudx = np.arange(-1.0, 1.0, 1.0e-3)
#dudx = dudx * sec_year

n_gln = 3
n_exp = (1.0 - n_gln) / (2.0 * n_gln)

def f_visc(A, dudx):
	#A = sec_year * A
	B    = A**( -1 / n_gln )
	visc_plot = 0.5 * B * ( (abs(dudx) + eps)**2 )**n_exp
	return visc_plot
 	

fig = plt.figure(dpi=400)
ax = fig.add_subplot(111)

plt.rcParams['text.usetex'] = True

As = np.array([1.0e-24, 1.0e-25, 1.0e-26])
col = ['red', 'darkgreen', 'darkblue']

for i in range(len(As)):
	A_now = As[i]
	visc_plot  = f_visc(As[i], dudx)
	
	ax.plot(dudx, visc_plot, linestyle='-', color=col[i], marker='None', \
		 	markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$A = \ $'+str(A_now)) 


ax.legend(loc='upper right', ncol = 1, frameon = True, \
		  framealpha = 1.0, fontsize = 14, fancybox = True)
	
ax.set_ylabel(r'$\eta(\theta, \partial u / \partial x)$',fontsize=18)
ax.set_xlabel(r'$\partial u / \partial x$',fontsize=18)

ax.yaxis.label.set_color('darkblue')
ax.tick_params(axis='y', which='major', length=4, colors='darkblue')

ax.grid(axis='x', which='major', alpha=0.85)

plt.tight_layout()

plt.show()
plt.close(fig)



 	


