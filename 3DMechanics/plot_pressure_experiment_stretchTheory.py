import numpy as np
import os
import datetime

from scipy.integrate import solve_ivp, simpson

import matplotlib.pyplot as plt
import matplotlib

import plot_utils.PlotLibrary as plotlib


from matplotlib.ticker import (MultipleLocator)

import pandas as pd

cmap = matplotlib.colormaps['plasma']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(60)

# What is the size of this figure relative to the journal specs
xFraction = 1.0
yFraction = 1.0

width = xFraction*figSpecs.singleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Script for plotting data
######################################################################

def compute_pressure_vs_strain(Ri, Ro, C1, C2, strain_array, n_points=200):
	pressures = []
	R_vals = np.linspace(Ri, Ro, n_points)

	for strain in strain_array:
		ri = Ri * (1 + strain)  # deformed inner radius

		# Solve incompressibility ODE: dr/dR = R^2 / r^2
		def dr_dR(R, r): return R**2 / r**2
		sol_r = solve_ivp(
			dr_dR, [Ri, Ro], [ri], dense_output=True,
			rtol=1e-9, atol=1e-12
		)
		r_vals = sol_r.sol(R_vals)[0]

		# Compute stretches
		lambda_theta = r_vals / R_vals
		lambda_r = 1 / lambda_theta**2
		r_eval = r_vals  # deformed radial coordinate

		# Stress difference integrand from Mooney-Rivlin model
		integrand = (
			4 / r_eval * (
				C1 * (lambda_theta**2 - lambda_r**2) -
				C2 * (1 / lambda_theta**2 - 1 / lambda_r**2)
			)
		)

		# Integrate to compute pressure difference (P_in - P_out)
		deltaP = -simpson(integrand[::-1], r_eval[::-1])
		pressures.append(deltaP)

	return np.array(pressures)

E = 4200 

# Import the data from anirban
excelFile = "./Data/time_radius_thickness_strain_pressure_1kPa.xlsx"
#excelFile = "time_radius_thickness_strain_pressure_2kPa.xlsx"
sheets_dict = pd.read_excel(excelFile, sheet_name=None)

flag = True

def rolling_mean_with_edges(data, window_size):
	kernel = np.ones(window_size) / window_size
	mean = np.convolve(data, kernel, mode='same')
	return mean

cnt = 0 
# Compute the inner and outer strains
for sheet_name, df in sheets_dict.items():
	data = df.iloc[1:].to_numpy(dtype=float)  # Skip first row and convert to NumPy
	
	r_lumen = data[:,1]
	r_lumen_std = data[:,2]

	h = data[:,3]
	h_std = data[:,4]	

	P = data[:,11]

	L_lumen = 2*np.pi*r_lumen
	strain_inner = (L_lumen - L_lumen[0]) / L_lumen[0]

	# beta calculation
	beta_min = 0.4
	beta_max = 0.6
	r0 = r_lumen[0]
	h0 = h[0]
	alpha = h0/r0
	beta = h0/(r0+h0)
	print(beta)

	P = rolling_mean_with_edges(P, 5)

	ax.scatter(strain_inner,P,c="darkgrey",s=4)

	cnt+=1

strain_ref = np.linspace(0,1.1,50)

beta = 0.5
mu = E/3
Ri = 30e-6
H = Ri*beta/(1-beta)
C1 = mu/2
C2 = 0
MR_pressures = compute_pressure_vs_strain(Ri=Ri, Ro=Ri+H, C1=C1, C2=C2, strain_array=strain_ref)
plt.plot(strain_ref, MR_pressures, color="k",lw=1, ls=":", zorder=3, label=r"Neo-Hookean thick shell ($E_y=4200$ Pa)")


################
# Theory section
################

d = 10*1e-6		# meters
R0_i = 1.6*d
beta = 0.5
Et0 = 8.25e-3

epsilon = strain_ref

lambda_i = 1 + epsilon
lambda_o = (1 + (1-beta)**3 * (lambda_i**3 - 1) )**(1/3)
lambda_H = lambda_o/beta - (1-beta)*lambda_i/beta

n = 6
alpha_n = np.sqrt( n*np.tan(np.pi/n)/np.pi)

L = (alpha_n/np.sqrt(3)) * (2-beta) * np.sqrt(beta*(3-3*beta+beta**2)/(1-beta)) * (R0_i/d)**1.5


P = Et0/(3*R0_i)* ( (2*lambda_i**(-1) - 2*lambda_i**(-7)) + 
				    (1-beta)*(2*lambda_o**(-1) - 2*lambda_o**(-7)) +  
				  	L*(lambda_H - lambda_H**(-2))*(1/lambda_o**2 - 1/(lambda_i**2 *(1-beta)**2))
				   )

ax.plot(epsilon,P,label="Neo-Hookean shell model",c="cornflowerblue")

Et0_soft = 1.34e-2

Psoft = Et0_soft/(3*R0_i)* ( (2*lambda_i**(-1) - 2*lambda_i**(-7)) + 
				    (1-beta)*(2*lambda_o**(-1) - 2*lambda_o**(-7)) +  
				  	(1/2)*L*(1 - lambda_H**(-2))*(1/lambda_o**2 - 1/(lambda_i**2 *(1-beta)**2))
				   )

ax.plot(epsilon,Psoft,label="Soft-mode shell model",c="firebrick")

print(Et0/2e-7) # 5.7e4 Pa
print(Et0_soft/2e-7) #1.05e5 Pa

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xlabel(r'$\epsilon_{lumen}$')
ax.set_ylabel(r'$P$ [Pa]')

ax.set_xlim([0,1.1])
ax.set_ylim([0,1250])

leg = plotlib.set_legend(ax,pos=4)
ax.xaxis.set_major_locator(MultipleLocator(0.25))

ax.yaxis.set_major_locator(MultipleLocator(250))

plotlib.set_position(ax,x=0.15,y=0.20,width=0.75,height=0.75)


# Save the figure
fig.savefig('./Plots/pressure_versusstrain_compare_Experiment_Cellular_Continuum.pdf',dpi=600)

plt.show()