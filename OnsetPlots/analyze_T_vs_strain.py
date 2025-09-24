import numpy as np
import h5py

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
# Script for plotting data
######################################################################

def rotation_angles_from_inner_outer(r, r0, cell_list, Nlateral):
	angles = []

	for cell_nodes in cell_list:
		# Extract indices
		i_in = cell_nodes[Nlateral]
		i_out = cell_nodes[0]

		# Current positions
		R_in = r[i_in]
		R_out = r[i_out]

		# Reference positions
		R0_in = r0[i_in]
		R0_out = r0[i_out]

		# Current norms
		R_out_norm = np.linalg.norm(R_out)
		R_in_norm = np.linalg.norm(R_in)
		R0_out_norm = np.linalg.norm(R0_out)
		R0_in_norm = np.linalg.norm(R0_in)

		# Radial directions
		radial_dir_in_aff = R0_in / R0_in_norm
		radial_dir_out_aff = R0_out / R0_out_norm

		# Affine-transformed positions
		R_in_affine = R_in_norm * radial_dir_in_aff
		R_out_affine = R_out_norm * radial_dir_out_aff

		# Vectors
		v_affine = R_out_affine - R_in_affine
		v_actual = R_out - R_in

		v_affine /= np.linalg.norm(v_affine)
		v_actual /= np.linalg.norm(v_actual)

		# Angle
		cross = v_affine[0] * v_actual[1] - v_affine[1] * v_actual[0]
		dot = np.dot(v_affine, v_actual)
		angle = np.arctan2(cross, dot)  
		angles.append(angle)

	return np.array(angles)

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
# Load data from npz file
V_range = np.array([1]) #arange(1,21)#,4,5,6,7,8,9,10])
kt_range = np.array([0.000005,0.00005,0.0005,0.005,0.05,0.5])
ka = 0.5
Nlateral = 20
iterations_to_plot = np.arange(2,151)  # adjust as needed


logC_vals = np.log10(kt_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

c_count = 0
for kt in kt_range:
	# Dictionary to accumulate curvature values for each iteration
	curvature_pool = {k: [] for k in iterations_to_plot}
	tilt_pool = {k: [] for k in iterations_to_plot}

	for V in V_range:
		name = f"kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_0.0_ddlog9_V{V:03d}"#shape_kl_1.0_ka_0.01_kt_0.0_al_0.0_Ri_1.6_beta_0.5_N_20_d_0.0_disorder2_check"
		with h5py.File(f"./Data/shape_{name}.h5", "r") as h5:
			shapes = h5["data"][:] 
			cell_list = h5["cell_list"][:]
			Niterations = shapes.shape[2]

			data_log = np.genfromtxt(f"Data/data_{name}.txt",skip_header=1) 


			r0 = shapes[:,:,0]
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

					# Compute curvature
					dtangent = np.gradient(tangent, axis=0)
					curvature = np.linalg.norm(dtangent, axis=1) / ds

					# Arc length
					arc_length = np.sum(ds)

					# Arc-length-normalized total bending
					total_bending = np.sum(curvature**2 * ds) * arc_length

					curvature_pool[k].append(total_bending)		

				tilt = tilt_angle(R, cell_list, Nlateral)

				total_tilt = tilt**2
				tilt_pool[k].append(total_tilt)		

	strain = []
	C = []

	for k in iterations_to_plot:
		values1 = np.array(curvature_pool[k])
		values2 = np.array(tilt_pool[k])
		area_growth = data_log[k,0]
		beta = 0.5
		inner_strain = np.sqrt(1 + area_growth / (1 - beta)**2) - 1

		strain.append(inner_strain)
		C.append(np.mean(values2))#/np.mean(values2))

	strain = np.array(strain)
	C = np.array(C)

	# Main
	ax.plot(strain, C,c=colors[c_count],lw=1,ms=2,alpha=1.0,zorder=0,label=f"{kt}")
	c_count +=1

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xscale("log")
ax.set_yscale("log")
			  

ax.set_xlim([1e-3,1e1])
ax.set_ylim([1e-8,5e0])

plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\langle T \rangle$')
leg = plt.legend(frameon=False)
leg.set_title(r"$k_t$")

ax.set_title(fr"$\tilde{{k}}_b={ka}$",pad=-5, fontsize=6)

#ax.set_title(fr"$k_a=0.0$",pad=-5, fontsize=6)


plotlib.set_position(ax,x=0.27,y=0.20,width=0.68,height=0.75)
# Save the figure

fig.savefig('./Plots/T_vstrain_Sim_nodisorder.pdf',dpi=600,transparent=True)

plt.show()