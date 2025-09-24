import numpy as np
from scipy.integrate import simpson

import matplotlib.pyplot as plt
import matplotlib

import plot_utils.PlotLibrary as plotlib

from matplotlib.ticker import (MultipleLocator)


cmap = matplotlib.colormaps['Reds']
cmap_grey = matplotlib.colormaps['Greys']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('NatMat')
figSpecs.set_figureHeight(60)

# What is the size of this figure relative to the journal specs
xFraction = 5/12
yFraction = 1.0

width = xFraction*figSpecs.doubleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Script for plotting data
######################################################################

kt_range = np.array([0.000005,0.00005,0.0005,0.005,0.05,0.5])

logC_vals = np.log10(kt_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

for i, kt in enumerate(kt_range):
	path = f"./Data/data_kl_1.0_ko_1.0_ka_5e-06_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_dcount9_V001.txt"
	label = kt

	data = np.genfromtxt(path, skip_header=1)

	RO = data[:,1]
	b = data[:,9]	
	RI = RO-b
	P = data[:,8]

	A0 = np.pi*RO[0]**2
	A = np.pi*RO**2
	A_norm = (A - A0)/A0
	dA = A - A0

	RI0 = RI[0]
	RO0 = RO[0]
	H = RO0 - RI0
	A_shell = np.pi*RO0**2 - np.pi*RI0**2 
	RI_ic = np.sqrt(RO**2 - A_shell/np.pi)

	r_norm = (RO - RO[0]) / RO[0]
	epsilon = (RI_ic - RI0) / RI0

#    epsilon = RO/RO[0] - 1
	mask = epsilon < 0.05
	mask[0] = False
	coeffs = np.polyfit(epsilon[mask], P[mask], 1)  # degree 1 for linear fit
	slope, intercept = coeffs

	color = f"C{i:02d}"
	ax.plot(epsilon,P,lw=1,label=label, color=colors[i])


######################################################################
# Script for plotting theory
######################################################################

# parameters
k_s = 1
beta = 0.5
R0_i = 1.6
d = 1

# strain definitions
epsilon_o = np.sqrt(1 + (1-beta)**2*(epsilon**2 + 2*epsilon)) - 1
epsilon_h = (epsilon_o - (1-beta)*epsilon) / beta

# Pressure definitions
P = (k_s/R0_i)*( (epsilon/(1+epsilon)) + (1-beta)*(epsilon_o/(1+epsilon_o)) + ((R0_i**2/d**2)*((beta*(2-beta))/(1-beta)**2)*epsilon_h*( (1-beta)/(1+epsilon_o) - (1/(1+epsilon)) ) ) )
P_soft = (k_s/(R0_i))*( (epsilon/(1+epsilon)) + (1-beta)*(epsilon_o/(1+epsilon_o)) )

ax.plot(epsilon,P,lw=1, zorder=2,c="k",ls="dashed")
ax.plot(epsilon,P_soft,lw=1, zorder=2,c="k",ls="dotted")


custom_lines = [
	matplotlib.lines.Line2D([0], [0], color=cmap(0.5), lw=1, label=r'simulation ($\delta=1$)'),
	matplotlib.lines.Line2D([0], [0], color="k", lw=1.0, ls="dashed", label='model'),
	matplotlib.lines.Line2D([0], [0], color="k", lw=1.0, ls="dashed", label='soft mode model'),

]

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_title(r"$\tilde{k}_b = 0.00005$",pad=-5)
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$P$')

ax.set_xlim([0,1.0])
ax.set_ylim([0,0.6])
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Add legend
leg = plotlib.set_legend(ax,pos=4)
leg.set_title(r"$\tilde{k}_t:$")
ax.add_artist(leg)  # Keep it visible after adding a second legend

# Add second legend
leg2 = ax.legend(handles=custom_lines, loc='upper left', frameon=False,handlelength=1.5  )

plotlib.set_position(ax,x=0.17,y=0.17,width=0.77,height=0.77)

# Save the figure
fig.savefig('./Plots/pressure_strain_ktlowSim_2Dtheory.pdf',transparent=True, dpi=600)

plt.show()
