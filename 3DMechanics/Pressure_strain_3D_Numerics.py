import numpy as np
import os

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

epsilon = np.linspace(0,1,300)

################
# Numerical integral
################

# Parameters
R0_i = 1.6        # inner radius
beta = 0.5        # H0 / (R0_i + H0)
d    = 1.0        # cell size -> V_cell0 = d^3
Et0  = 1.0        # Ey * t0   (surface modulus)
n = 6

S0 = Et0 / 3.0    # mu * t0

prefac = Et0/(R0_i)

# Derived geometry
H0    = beta/(1.0 - beta) * R0_i
R0_o  = R0_i + H0
Vsh0  = (4/3)*np.pi*(R0_o**3 - R0_i**3)
V_cell0 = d**3
Ncell = Vsh0 / V_cell0

# reference areas
A0_i = 4*np.pi*R0_i**2
A0_o = 4*np.pi*R0_o**2

# per-cell circumference factors (n-gon approx)

kappa_n = np.sqrt( n*np.tan(np.pi/n)/np.pi)

# strain-dependent geometry
R_i  = R0_i * (1.0 + epsilon)
R_o  = (R_i**3 + (R0_o**3 - R0_i**3))**(1.0/3.0)
H    = R_o - R_i

# Circumference profile
R0 = np.linspace(R0_i, R0_o, 400)
C0_R0 = 2.0 * kappa_n * np.sqrt((4.0*np.pi*R0**2)/Ncell)

# stretches
lam_i = R_i / R0_i
lam_o = R_o / R0_o

# NH energy densities (per face, incompressible)
def W_sph(lam):  # equi-biaxial sheet
    return 0.5*(2.0*lam**2 + lam**(-4) - 3.0)

def W_lat(lC, lH):  # lateral, two in-plane stretches
    l_t = 1.0/(lC*lH)
    return 0.5*(lC**2 + lH**2 + l_t**2 - 3.0)

# energies
E_in  = A0_i * S0 * W_sph(lam_i)
E_out = A0_o * S0 * W_sph(lam_o)

E_lat = np.zeros(len(epsilon))
for i,_ in enumerate(R_i):
    k_vol = R_i[i]**3 - R0_i**3
    r     = (R0**3 + k_vol)**(1.0/3.0)
    lamC  = r / R0
    lamH  = R0**2 / r**2
    E_lat_cell = S0 * np.trapezoid(C0_R0 * W_lat(lamC, lamH), R0)
    E_lat[i]   = Ncell * E_lat_cell

E_tot = E_in + E_out + E_lat

# pressure from energy derivative wrt inner volume
V_in = (4/3)*np.pi*R_i**3
P    = np.gradient(E_tot, epsilon) / np.gradient(V_in, epsilon)

ax.scatter(epsilon[::10], P[::10]/prefac, color=cmap_grey(0.75),s=8,facecolor="w",label="numerical: neo-Hookean")

# SOFT MODE
def W_lat_soft(lC):  # lateral, two in-plane stretches
    l_t = 1.0/(lC)
    return 0.5*(lC**2 + l_t**2 - 2)

E_lat_soft = np.zeros(len(epsilon))
for i,_ in enumerate(R_i):
    k_vol = R_i[i]**3 - R0_i**3
    r     = (R0**3 + k_vol)**(1.0/3.0)
    lamC  = r / R0
    E_lat_cell = S0 * np.trapezoid(C0_R0 * W_lat_soft(lamC), R0)
    E_lat_soft[i] = Ncell * E_lat_cell

E_tot_soft = E_in + E_out + E_lat_soft
P_soft = np.gradient(E_tot_soft, epsilon) / np.gradient(V_in, epsilon)

ax.scatter(epsilon[::10], P_soft[::10]/prefac, color=cmap_grey(0.25),facecolor="w",s=8,label="numerical: neo-Hookean (soft mode)")

################
# Theory section
################

R0_i = 1.6
beta = 0.5
d = 1

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

ax.plot(epsilon,P/prefac,color=cmap_grey(0.75),lw=1,label="theory: neo-Hookean")

Psoft = Et0/(3*R0_i)* ( (2*lambda_i**(-1) - 2*lambda_i**(-7)) + 
				    (1-beta)*(2*lambda_o**(-1) - 2*lambda_o**(-7)) +  
				  	(1/2)*L*(1 - lambda_H**(-2))*(1/lambda_o**2 - 1/(lambda_i**2 *(1-beta)**2))
				   )

ax.plot(epsilon,Psoft/prefac,color=cmap_grey(0.25),lw=1,label="theory: neo-Hookean (soft mode)")

# Linear expressions
klin = Et0/(3*R0_i)* ( 12 + 12*(1-beta)**4 + L*3*beta*(2-beta)**2/(1-beta) )
klinsoft = Et0/(3*R0_i)* ( 12 + 12*(1-beta)**4  + L*beta*(2-beta)**2/(1-beta) )

#ax.plot(epsilon,klin*epsilon,ls=":",c="cornflowerblue")
#ax.plot(epsilon,klinsoft*epsilon,ls=":",c="firebrick")


######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$\tilde{P}$')

leg = plotlib.set_legend(ax,pos=4)


ax.set_xlim([0,1.0])
ax.set_ylim([0,2])

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(.5))

plotlib.set_position(ax,x=0.17,y=0.17,width=0.77,height=0.77)


# Save the figure
fig.savefig('./Plots/pressure_strain_3Dcellular_numerics.pdf',transparent=True, dpi=600)


plt.show()
