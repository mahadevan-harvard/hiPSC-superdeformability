import numpy as np

from scipy.optimize import brentq

import matplotlib.pyplot as plt
import matplotlib

import plot_utils.PlotLibrary as plotlib

cmap = matplotlib.colormaps['Reds']

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

def rhs(theta, epsilon):
	# (sinθ/θ)*((1+ε)cosθ - 1) / ((1+ε)^2 cos^3θ)
	c = np.cos(theta)
	s = np.sin(theta)
	# avoid θ=0 numerics via sinc-like handling
	r = (s/np.where(theta==0.0, 1.0, theta)) * (((1.0+epsilon)*c - 1.0) / (((1.0+epsilon)**2) * (c**3)))
	return r

def theta_max(eps):
	# θ ∈ [0, arccos(H/H0)] = arccos(1/(1+ε)), capped below π/2
	val = 1.0/(1.0+eps)
	val = np.clip(val, -1.0, 1.0)
	return np.arccos(val)

def solve_thetas(epsilon_array, k_tilde, rhs, theta_max):
	"""
	Solve for tilt angle theta(ε) given balance equation.

	Parameters
	----------
	epsilon_array : array-like
	    Strain values ε.
	k_tilde : float
	    Dimensionless stiffness ratio (kt / (ks * H0^2), possibly × count_factor).
	rhs : callable
	
	    Function rhs(theta, eps).
	theta_max : callable
	    Function theta_max(eps), max admissible angle for given eps.

	Returns
	-------
	thetas : np.ndarray
	    Array of solved tilt angles.
	"""
	thetas = []
	for eps in epsilon_array:
		t_hi = theta_max(eps)
		if t_hi <= 1e-9:
			thetas.append(0.0)
			continue
		a, b = 1e-9, t_hi - 1e-9
		f = lambda th: rhs(th, eps) - k_tilde
		fa, fb = f(a), f(b)
		if fa * fb > 0:
			# coarse scan
			ts = np.linspace(a, b, 200)
			ft = f(ts)
			ix = np.where(np.signbit(ft[:-1]) != np.signbit(ft[1:]))[0]
			if ix.size == 0:
				thetas.append(0.0)
				continue
			a, b = ts[ix[0]], ts[ix[0]+1]
		try:
			th = brentq(f, a, b, maxiter=200, xtol=1e-12)
		except ValueError:
			th = 0.0
		thetas.append(th)
	return np.array(thetas)

def f_eps(eps): return eps/((1+eps)**2)
def A_eps(eps): return (5*eps - 3)/(6*(1+eps)**2)
def fprime_eps(eps): return (1 - eps)/((1+eps)**3)

def eps_c_from_ktilde(k_tilde):
    # solve k_tilde = eps/(1+eps)^2 (valid if k_tilde < 1/4)
    y = k_tilde
    if not (0 < y < 0.25): return np.inf
    return (1 - 2*y - np.sqrt(1 - 4*y)) / (2*y)


######################################################################
# Script for plotting data
######################################################################

# Parameters
ks, H0 = 1.0, 1.0
count_factor = 2.0             # e.g. =2.0 if per-corner vs per-edge (N_c/N_e)

eps_min, eps_max = 1e-3, 1e0
epsilon_array = np.logspace(np.log10(eps_min), np.log10(eps_max), 400)

kt_range = np.array([0.000005,0.00005,0.0005,0.005,0.05,0.5])
kt_scaled = (np.log10(kt_range) - (np.log10(np.min(kt_range))-1))/(np.log10(np.max(kt_range)) - (np.log10(np.min(kt_range))-1))

for i,kt in enumerate(kt_range):

	k_tilde = count_factor * kt / (ks * H0**2)

	thetas = thetas = solve_thetas(epsilon_array, k_tilde, rhs, theta_max)
	ax.plot(epsilon_array, thetas**2, lw=1, label=f"{kt}",c=cmap(kt_scaled[i]))

	# --- inside your plotting loop for each k_tilde curve ---
	ec = eps_c_from_ktilde(k_tilde)
	if np.isfinite(ec):
		# 1) onset line
		plt.axvline(ec, ls=':', lw=0.9, color="grey", alpha=1.0,zorder=0)

# ######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xscale("log")
ax.set_yscale("log")
			  

ax.set_xlim([1e-3,1e1])
ax.set_ylim([1e-8,5e0])

leg = plotlib.set_legend(ax,pos=4)
leg.set_title(r"$\tilde{k}_t$")


plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$ T$')

plotlib.set_position(ax,x=0.27,y=0.20,width=0.68,height=0.75)

# Save the figure
fig.savefig('./Plots/T_vstrain_theory.pdf',dpi=600,transparent=True)

plt.show()