import numpy as np
import h5py


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

def onset_jump_loglog(T, eps, smooth=5, min_dlog=0.3, min_gain=0.3):
	"""
	Onset = ε at max jump in log10(T) vs log10(ε).
	- smooth: odd window for box smoothing in log-space (1 disables)
	- min_dlog: min slope (decades of T per decade of ε)
	- min_gain: required total rise after onset (decades)
	Returns np.nan if no clear onset.
	"""
	eps = np.asarray(eps, float)
	T   = np.asarray(T,   float)

	# keep finite, ε>0, T>0
	m = np.isfinite(eps) & np.isfinite(T) & (eps > 0)
	if m.sum() < 3: return np.nan
	eps, T = eps[m], T[m]
	o = np.argsort(eps)
	eps, T = eps[o], T[o]

	# floors to avoid log(0)
	eps_floor = 1e-12
	T_floor   = max(1e-12, 1e-3*np.nanmax(T))

	x = np.log10(np.maximum(eps, eps_floor))
	y = np.log10(np.maximum(T,   T_floor))

	# optional smoothing (boxcar)
	if smooth > 1:
		k = int(smooth); k = k + 1 - (k % 2)  # make odd
		y = np.convolve(y, np.ones(k)/k, mode='same')

	# gradient and argmax (avoid endpoints)
	dy = np.gradient(y, x)
	if dy.size < 3: return np.nan
	i = 1 + int(np.argmax(dy[1:-1]))

	# guards: significant jump and sustained rise
	if dy[i] < min_dlog: return np.nan
	if (y[-1] - y[i]) < min_gain: return np.nan

	return eps[i]  # return onset in original ε units

######################################################################
# Theory
######################################################################

# Parameters
ks, H0 = 1.0, 1.0

eps_min, eps_max = 1e-3, 1e0
epsilon_array = np.logspace(np.log10(eps_min), np.log10(eps_max), 400)
kb_range = np.logspace(-6,1,100)
epsilon_c = np.zeros(len(kb_range))
for i,kb in enumerate(kb_range):

	k_tilde = kb / (ks * H0**2)

	epsilon_c[i] = k_tilde*(np.pi**2)

ax.plot(kb_range,epsilon_c,color="grey",label="theory")

######################################################################
# Simulation (no disorder)
######################################################################

# Load data from npz file
V_range = np.array([1])
ka_range = np.array([0.0005,0.0008,0.0013,0.002,0.0032,0.005,0.008,0.013,0.02,0.032,0.05,0.08,0.13])
kt = 0.5             
Nlateral = 20
iterations_to_plot = np.arange(2,151)  # adjust as needed

logC_vals = np.log10(kb_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

c_count = 0
for ka in ka_range:
	curvature_pool = {k: [] for k in iterations_to_plot}

	for V in V_range:
		name = f"kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_0.0_ddlog9_V{V:03d}"
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

	strain = []
	C = []

	for k in iterations_to_plot:
		values = np.array(curvature_pool[k])
		area_growth = data_log[k,0]
		beta = 0.5
		inner_strain = np.sqrt(1 + area_growth / (1 - beta)**2) - 1

		strain.append(inner_strain)
		C.append(np.mean(values))

	strain = np.array(strain)
	C = np.array(C)

	onset = onset_jump_loglog(C, strain)
	if np.max(C) < 1e-3:
		onset = None

	if c_count == 0:
		ax.scatter(ka,onset,  edgecolor="k",facecolor="none", marker="o",s=15, label=r"sim. ($\delta=0$)")
	else:
		ax.scatter(ka,onset,  edgecolor="k",facecolor="none", marker="o",s=15)
	c_count +=1

######################################################################
# Simulation (disorder)
######################################################################

# Load data from npz file
V_range = np.array([1])
ka_range = np.array([0.0005,0.0008,0.0013,0.002,0.0032,0.005,0.008,0.013,0.02,0.032,0.05,0.08,0.13])
kt = 0.5
Nlateral = 20
iterations_to_plot = np.arange(2,151)  # adjust as needed

logC_vals = np.log10(kb_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

c_count = 0
for ka in ka_range:
	curvature_pool = {k: [] for k in iterations_to_plot}

	for V in V_range:
		name = f"kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_ddlog9_V{V:03d}"
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

	strain = []
	C = []

	for k in iterations_to_plot:
		values = np.array(curvature_pool[k])
		area_growth = data_log[k,0]
		beta = 0.5
		inner_strain = np.sqrt(1 + area_growth / (1 - beta)**2) - 1

		strain.append(inner_strain)
		C.append(np.mean(values))

	strain = np.array(strain)
	C = np.array(C)

	onset = onset_jump_loglog(C, strain)
	if np.max(C) < 1e-3:
		onset = None

	if c_count == 0:
		ax.scatter(ka,onset, edgecolor="k",facecolor="none",marker="^",s=15, label=r"sim. ($\delta=1$)")
	else:
		ax.scatter(ka,onset, edgecolor="k",facecolor="none",marker="^",s=15)
	c_count +=1


######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)

ax.set_xscale("log")
ax.set_yscale("log")
			  
ax.set_xlim([1e-4,1e0])
ax.set_ylim([1e-3,5e0])

leg = plotlib.set_legend(ax,pos=2)

plt.xlabel(r'$\tilde{k}_b$')
plt.ylabel(r'$\epsilon_c$')

plotlib.set_position(ax,x=0.27,y=0.20,width=0.68,height=0.75)
# Save the figure
fig.savefig('./Plots/onset_vs_kb.pdf',dpi=600,transparent=True)

plt.show()