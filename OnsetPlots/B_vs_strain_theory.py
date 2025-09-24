import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import plot_utils.PlotLibrary as plotlib



cmap = matplotlib.colormaps['Blues']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('NatMat')
figSpecs.set_figureHeight(45)

# What is the size of this figure relative to the journal specs
xFraction = 1/4
yFraction = 1.0

width = xFraction*figSpecs.doubleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Functions
######################################################################

# Onset (reference frame)  (no intrinsic upper bound)
def eps_c_buckling_ktilde(ktilde):
	c = (np.pi**2) * float(ktilde)
	return c

# Amplitude from stationarity (reference frame):
def buckling_amplitude_ktilde(eps_array, ktilde, H0=1.0):
    e = np.asarray(eps_array, float)
    ec = (np.pi**2) * float(ktilde)
    a2 = (4*H0**2) / (np.pi**2 * (1.0 + e)**2) * (e - ec)
    return np.sqrt(np.maximum(a2, 0.0))

# Dimensionless curvature measure B in the reference frame:
def B_from_amplitude(eps_array, a, H0=1.0):
    return (np.pi**4 / 2.0) * (a**2) / (H0**2)

######################################################################
# Script for plotting data
######################################################################

# Parameters
ks, H0 = 1.0, 1.0

eps_min, eps_max = 1e-3, 1e0
epsilon_array = np.logspace(np.log10(eps_min), np.log10(eps_max), 400)
kb_range = np.array([0.000005,0.00005,0.0005,0.005,0.05,0.5])
kb_scaled = (np.log10(kb_range) - (np.log10(np.min(kb_range))-1))/(np.log10(np.max(kb_range)) - (np.log10(np.min(kb_range))-1))
   
for i, kb in enumerate(kb_range):
	ktilde = kb / (ks*H0**2)
	a = buckling_amplitude_ktilde(epsilon_array, ktilde, H0)
	B = B_from_amplitude(epsilon_array, a, H0)
	ax.plot(epsilon_array, B, lw=1, label=f"{kb}", c=cmap(kb_scaled[i]))

	ec = eps_c_buckling_ktilde(ktilde)
	if ec < 1:
		ax.axvline(ec, ls=':', lw=0.9, color="grey", alpha=1.0)

		slope_B = 2*np.pi**2 / (1.0 + ec)**2
		e_loc = np.logspace(np.log10(ec), np.log10(eps_max), 1000)
		B_pred = slope_B * (e_loc - ec)

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xscale("log")
ax.set_yscale("log")
			  

ax.set_xlim([1e-3,1e1])
ax.set_ylim([1e-8,1e2])

leg = plotlib.set_legend(ax,pos=4)
leg.set_title(r"$\tilde{k}_b$")


plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$B$')

plotlib.set_position(ax,x=0.27,y=0.20,width=0.68,height=0.75)

# Save the figure
fig.savefig('./Plots/B_vstrain_theory.pdf',dpi=600,transparent=True)

plt.show()