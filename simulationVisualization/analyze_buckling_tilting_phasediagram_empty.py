import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib
import plot_utils.PlotLibrary as plotlib

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
							   AutoMinorLocator, LogLocator)


cmap = matplotlib.colormaps['Reds']

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
V_range = np.arange(1,2)
ka_range = np.array([0.000005,0.00005,0.0005,0.005,0.05,0.5])
kt_range = np.array([0.000005,0.00005,0.0005,0.005,0.05,0.5])

Nlateral = 20
iterations_to_plot = [78]  # adjust as needed

results = np.zeros([len(ka_range)*len(kt_range),3])
count = 0
for kt in kt_range:
	for ka in ka_range:

		tilt0_pool = {k: [] for k in iterations_to_plot}
		tilt_pool = {k: [] for k in iterations_to_plot}
		curvature_pool = {k: [] for k in iterations_to_plot}

		for V in V_range:
			name = f"kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_dcount9_V{V:03d}"#shape_kl_1.0_ka_0.01_kt_0.0_al_0.0_Ri_1.6_beta_0.5_N_20_d_0.0_disorder2_check"
			with h5py.File(f"./Data/shape_{name}.h5", "r") as h5:
				shapes = h5["data"][:] 
				cell_list = h5["cell_list"][:]
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

			C = np.mean(values1)/np.mean(values2)
			results[count,0] = kt
			results[count,1] = ka
			results[count,2] = C			
		count +=1

C_vals = results[:,0]
norm = matplotlib.colors.Normalize(vmin=C_vals.min(), vmax=C_vals.max())
colors = cmap(norm(C_vals))

logC_vals = np.log10(kt_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

markers = ["o", "s", "^", "D", "v", "p", "*", "x"]

for i, kt in enumerate(kt_range):
	mask = results[:,0] == kt
	ax.errorbar(results[mask,0],results[mask,1],linewidth=0, ms=3, fmt=f"{markers[i]}-",color=colors[i], label=fr"${kt}$",markerfacecolor="w",alpha=0.255)#,colors=colors)	


ka_range = np.array([0.000005,0.5])
kt_range = np.array([0.000005,0.5])

Nlateral = 20
iterations_to_plot = [78]  # adjust as needed

results = np.zeros([len(ka_range)*len(kt_range),3])
count = 0
for kt in kt_range:
	for ka in ka_range:

		tilt0_pool = {k: [] for k in iterations_to_plot}
		tilt_pool = {k: [] for k in iterations_to_plot}
		curvature_pool = {k: [] for k in iterations_to_plot}

		for V in V_range:
			name = f"kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_dcount9_V{V:03d}"
			with h5py.File(f"./Data/shape_{name}.h5", "r") as h5:
				shapes = h5["data"][:] 
				cell_list = h5["cell_list"][:]
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

			C = np.mean(values1)/np.mean(values2)
			print(C)
			results[count,0] = kt
			results[count,1] = ka
			results[count,2] = C			
		count +=1

C_vals = results[:,0]
norm = matplotlib.colors.Normalize(vmin=C_vals.min(), vmax=C_vals.max())
colors = cmap(norm(C_vals))

logC_vals = np.log10(kt_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

markers = ["o", "p"]

for i, kt in enumerate(kt_range):
	mask = results[:,0] == kt
	ax.errorbar(results[mask,0],results[mask,1],linewidth=0, ms=3, fmt=f"{markers[i]}-",color=colors[i], label=fr"${kt}$")#,colors=colors)	

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xlabel(r'$\tilde{k}_t$')
ax.set_ylabel(r'$\tilde{k}_b$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([5e-8,5e1])
ax.set_ylim([5e-8,5e1])

plotlib.set_position(ax,x=0.17,y=0.17,width=0.77,height=0.77)
ax.yaxis.set_major_locator(LogLocator(base=100.0, subs=[1.0], numticks=10))

ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

# Save the figure
fig.savefig('./Plots/phase_diagram_empty.svg',dpi=600,transparent=True)

plt.show()