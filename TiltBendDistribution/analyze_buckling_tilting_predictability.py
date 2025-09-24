import numpy as np
import os
import datetime

import matplotlib.pyplot as plt
import matplotlib

import plot_utils.PlotLibrary as plotlib

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
							   AutoMinorLocator, LogLocator)


cmap = matplotlib.colormaps['plasma']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('NatMat')
figSpecs.set_figureHeight(30)

# What is the size of this figure relative to the journal specs
xFraction = 7/24#60/180
yFraction = 1.0

width = xFraction*figSpecs.doubleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

ax2 = ax.twinx()  # Create second y-axis

######################################################################
# Script for plotting data
######################################################################

redcolor = "#e32f27ff"
bluecolor = "#3787c1ff"

def tilt_angle(r, cell_list, Nlateral):
	angles = []

	for cell_nodes in cell_list:
		i_in = cell_nodes[Nlateral]
		i_out = cell_nodes[0]

		R_in = r[i_in]
		R_out = r[i_out]

		# Vector from inner to outer
		v_actual = R_out - R_in
		v_actual /= np.linalg.norm(v_actual)

		# Radial direction at inner node
		radial_dir = R_in / np.linalg.norm(R_in)

		# Compute signed angle
		cross = radial_dir[0] * v_actual[1] - radial_dir[1] * v_actual[0]
		dot = np.dot(radial_dir, v_actual)
		angle = np.arctan2(cross, dot)

		angles.append(angle)

	return np.array(angles)

# Load data from npz file
V_range = np.arange(1,2)#,4,5,6,7,8,9,10])
Nlateral = 20
iterations_to_plot = [78]  # adjust as needed

kt_range = np.array([0.005])
ka = 0.00005


tilt0_pool = {k: [] for k in iterations_to_plot}
tilt_pool = {k: [] for k in iterations_to_plot}
curvature_pool = {k: [] for k in iterations_to_plot}

logC_vals = np.log10(kt_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min()), vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))
markers = ["o", "s", "^", "D", "v", "p", "*", "x"]


for i,kt in enumerate(kt_range):

	for V in V_range:
		name = f"shape_kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_dcount9_V{V:03d}"
		data = np.load(f"Data/{name}.npz")
		shapes = data["data"]  # Shape: (Nnodes, 2, Niterations)
		cell_list = data["cell_list"]
		Niterations = shapes.shape[2]

		for k in iterations_to_plot:
			R0 = shapes[:,:,0]
			R = shapes[:,:,k]
			for j,cell_nodes in enumerate(cell_list):

				# Extract inner and outer node
				inner = cell_nodes[Nlateral]
				outer = cell_nodes[0]
				R_in, R_out = R[inner], R[outer]

				# Get lateral nodes (excluding inner and outer)
				interior_nodes = cell_nodes[1:Nlateral]
				node_indices = [outer] + list(interior_nodes) + [inner]
				points = R[node_indices]

				# Compute arc length with respect to outer node
				s = np.zeros(len(points))
				s[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))

				# Compute tangents with respect to arc length
				dR = np.gradient(points, s, axis=0)					# distance between nearest neighbours
				tangent = dR / np.linalg.norm(dR, axis=1)[:, None]	# unit vectors

				# Compute curvature
				dtangent = np.gradient(tangent, s, axis=0)		# rate of change tangent vector with respect to the arc length
				curvature = np.linalg.norm(dtangent, axis=1)	# Curvature is the magnitude of the rate of change

				# Total arc length
				arc_length = s[-1] - s[0]

				# Arc-length-normalized total bending
				total_bending = arc_length * np.sum(curvature**2 * np.gradient(s))	

				curvature_pool[k].append(total_bending)			
			
			tilt0 = tilt_angle(R0, cell_list, Nlateral)
			tilt = tilt_angle(R, cell_list, Nlateral)
			total_tilt = tilt**2

			for entry in total_tilt:
				tilt_pool[k].append(entry)	

			for entry in tilt0:
				tilt0_pool[k].append(entry)		

	# Example: pick specific iterations
	for k in iterations_to_plot:
		values1 = np.array(curvature_pool[k])
		values2 = np.array(tilt_pool[k])#(tilt_accumulated[:, k]-original_tilt)**2
		tilt_degrees = np.abs(180*np.array(tilt0_pool[k])/np.pi)

		area_growth = k*0.005
		beta = 0.5
		inner_strain = np.sqrt(1 + area_growth/(1-beta)**2) - 1
		ax.scatter(tilt_degrees, values1, label=f'{inner_strain:.2f}',facecolor=bluecolor,edgecolor='none',s=10,alpha=0.5,zorder=0)
		ax2.scatter(tilt_degrees, values2, label=f'{inner_strain:.2f}',facecolor=redcolor,edgecolor='none',s=10,alpha=0.5,zorder=0)

		tilt_lin = np.arange(0,30)
		coeffs1 = np.polyfit(tilt_degrees, values1, 1)
		fit1 = np.poly1d(coeffs1)
		ax.plot(tilt_lin, fit1(tilt_lin), color=bluecolor, linewidth=1, ls=":",zorder=1)

		coeffs2 = np.polyfit(tilt_degrees, values2, 1)
		fit2 = np.poly1d(coeffs2)
		ax2.plot(tilt_lin, fit2(tilt_lin), color=redcolor, linewidth=1, ls=":",zorder=1)

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)
plotlib.set_box(ax2,halfstyle=True,flipy=True)

ax.set_xlabel(r'Initial tilt angle [\textdegree]')
ax.set_ylabel(r'$\langle B \rangle$')

ax.set_xlim(0,30)
ax.set_ylim(0,17)
ax2.set_ylim(0,0.4)


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax2.yaxis.set_major_locator(MultipleLocator(0.2))

plotlib.set_position(ax,x=0.20,y=0.26,width=0.60,height=0.65)
ax.set_title(r"$\tilde{k}_t=0.005$ $\tilde{k}_b=0.00005$",pad=-5, fontsize=6)

ax2.spines['right'].set_color(redcolor)
ax2.tick_params(axis='y', colors=redcolor)
ax2.set_ylabel(r'$\langle T \rangle$', color = redcolor)

# Save the figure
fig.savefig('./Plots/scatter_buckling_tilting_predictability.svg',dpi=600,transparent=True)

plt.show()