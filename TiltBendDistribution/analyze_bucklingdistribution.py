import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib

import plot_utils.PlotLibrary as plotlib

from scipy.stats import gaussian_kde

cmap = matplotlib.colormaps['Greys']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('NatMat')
figSpecs.set_figureHeight(30)

# What is the size of this figure relative to the journal specs
xFraction = 7/24
yFraction = 1.0

width = xFraction*figSpecs.doubleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Script for plotting data
######################################################################

# Load data from npz file
V_range = np.arange(1,2)
Nlateral = 20
iterations_to_plot = [16, 35,78]
bins = np.linspace(0, 75, 26)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Dictionary to accumulate curvature values for each iteration
curvature_pool = {k: [] for k in iterations_to_plot}

for V in V_range:
	name = f"kl_1.0_ko_1.0_ka_5e-05_kt_0.005_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_dcount9_V{V:03d}"
	with h5py.File(f"./Data/shape_{name}.h5", "r") as h5:
		shapes = h5["data"][:] 
		cell_list = h5["cell_list"][:]
		Niterations = shapes.shape[2]

		for k in iterations_to_plot:
			R = shapes[:, :, k]
			for j,cell_nodes in enumerate(cell_list):

				# Extract inner and outer node
				inner = cell_nodes[Nlateral]
				outer = cell_nodes[0]

				# Get lateral nodes (excluding inner and outer)
				interior_nodes = cell_nodes[1:Nlateral]
				node_indices = [outer] + list(interior_nodes) + [inner]
				points = R[node_indices]

				# Compute tangents
				dR = np.gradient(points, axis=0)
				ds = np.linalg.norm(dR, axis=1)
				tangent = dR / ds[:, None]

				# Compute curvatur.e
				dtangent = np.gradient(tangent, axis=0)
				curvature = np.linalg.norm(dtangent, axis=1) / ds

				# Arc length
				arc_length = np.sum(ds)

				# Arc-length-normalized total bending
				total_bending = np.sum(curvature**2 * ds) * arc_length

				curvature_pool[k].append(total_bending)		
	
# Example: pick specific iterations
for k in iterations_to_plot:
	values = np.array(curvature_pool[k])

	kde = gaussian_kde(values)
	x = np.linspace(0, 20, 200)
	y = kde(x)

	area_growth = k*0.005
	beta = 0.5
	inner_strain = np.sqrt(1 + area_growth/(1-beta)**2) - 1
	plt.plot(x,y, label=f'{inner_strain:.2f}',color=cmap(inner_strain/0.65))

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xlim(0,20)
ax.set_ylim(-0.01,2.2)

plt.xlabel(r'$B$')
plt.ylabel('P(*)')
leg = plt.legend(frameon=False)
leg.set_title(r"$\epsilon$")
plotlib.set_position(ax,x=0.20,y=0.26,width=0.75,height=0.65)

ax.set_title(r"$\tilde{k}_t=0.005$ $\tilde{k}_b=0.00005$",pad=-5, fontsize=6)


# Save the figure
fig.savefig('./Plots/histogram_buckling_distribution.svg',dpi=600,transparent=True)

plt.show()