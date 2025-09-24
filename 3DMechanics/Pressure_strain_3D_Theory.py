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
# Theory section
################

R0_i = 1.6
beta = 0.5
d = 1
Et0  = 1.0        # Ey * t0   (surface modulus)
n = 6
prefac = Et0/(R0_i)

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
fig.savefig('./Plots/pressure_strain_3Dcellular_theoryonly.pdf',transparent=True, dpi=600)


plt.show()
