import numpy as np


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

def eps_c_from_ktilde(k_tilde):
    # solve k_tilde = eps/(1+eps)^2 (valid if k_tilde < 1/4)
    y = k_tilde
    if not (0 < y < 0.25): return np.inf
    return (1 - 2*y - np.sqrt(1 - 4*y)) / (2*y)

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
	if smooth > 1:
		i = smooth + int(np.argmax(dy[smooth:-smooth]))
	else:
		i = 1 + int(np.argmax(dy[1:-1]))

	# guards: significant jump and sustained rise
	if dy[i] < min_dlog: return np.nan
	if (y[-1] - y[i]) < min_gain: return np.nan

	return eps[i]  # return onset in original ε units

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

######################################################################
# THEORY -flat
######################################################################

# Parameters
ks, H0 = 1.0, 1.0
count_factor = 2.0             # e.g. =2.0 if per-corner vs per-edge (N_c/N_e)
beta = 0.5
eps_min, eps_max = 1e-3, 1e0

kt_range = np.logspace(-6,1,100) #np.array([0.00001,0.0001,0.001,0.01,0.1,1.0])
epsilon_c = np.zeros(len(kt_range))
epsilon_c_ring = np.zeros(len(kt_range))

for i,kt in enumerate(kt_range):

	k_tilde = kt / (ks * H0**2)

	epsilon_c[i] = eps_c_from_ktilde(count_factor*k_tilde)
ax.plot(kt_range,epsilon_c, label= "theory",color="grey")


ax.vlines(1/8,1e-5,1e2,color="grey",ls="dashed")

######################################################################
# THEORY - circular
######################################################################



# ######################################################################
# # SIMULATION (no disorder)
# ######################################################################

# Load data from npz file
V_range = np.array([1])
kt_range = np.array([0.0013,0.002,0.0032,0.005,0.008,0.013,0.02,0.032,0.05,0.08,0.13,0.2,0.32])
ka = 0.5
Nlateral = 20
iterations_to_plot = np.arange(2,151)  # adjust as needed

logC_vals = np.log10(kt_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

c_count = 0
for kt in kt_range:
	tilt_pool = {k: [] for k in iterations_to_plot}

	for V in V_range:
		name = f"kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_0.0_ddlog9_V{V:03d}"
		data = np.load(f"Data/shape_{name}.npz")
		shapes = data["data"]
		cell_list = data["cell_list"]
		Niterations = shapes.shape[2]

		data_log = np.genfromtxt(f"Data/data_{name}.txt",skip_header=1) 

		r0 = shapes[:,:,0]
		for k in iterations_to_plot:
			R = shapes[:, :, k]
			tilt = tilt_angle(R, cell_list, Nlateral)
			total_tilt = tilt**2
			tilt_pool[k].append(total_tilt)		

	strain = []
	C = []

	for k in iterations_to_plot:
		values2 = np.array(tilt_pool[k])
		area_growth = data_log[k,0]
		beta = 0.5
		inner_strain = np.sqrt(1 + area_growth / (1 - beta)**2) - 1

		strain.append(inner_strain)
		C.append(np.mean(values2))

	strain = np.array(strain)
	C = np.array(C)

	onset = onset_jump_loglog(C, strain)
	if np.max(C) < 1e-3:
		onset = None

	if c_count == 0:
		ax.scatter(kt,onset, label=r"sim. ($\delta=0$)",edgecolor="k",facecolor="none",marker="o",s=15)
	else:
		ax.scatter(kt,onset,edgecolor="k",facecolor="none",marker="o",s=15)
	c_count +=1

######################################################################
# SIMULATION (disorder)
######################################################################

# Load data from npz file
V_range = np.array([1])
kt_range = np.array([0.02,0.0013,0.002,0.0032,0.005,0.008,0.013,0.02,0.032,0.05,0.08,0.13,0.2,0.32])
ka = 0.5
Nlateral = 20
iterations_to_plot = np.arange(2,151)  # adjust as needed

logC_vals = np.log10(kt_range)
norm = matplotlib.colors.Normalize(vmin=np.floor(logC_vals.min())-1, vmax=np.ceil(logC_vals.max()))
colors = cmap(norm(logC_vals))

c_count = 0
for kt in kt_range:
	tilt_pool = {k: [] for k in iterations_to_plot}

	for V in V_range:
		name = f"kl_1.0_ko_1.0_ka_{ka}_kt_{kt}_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_ddlog9_V{V:03d}"
		data = np.load(f"Data/shape_{name}.npz")
		shapes = data["data"]
		cell_list = data["cell_list"]
		Niterations = shapes.shape[2]

		data_log = np.genfromtxt(f"Data/data_{name}.txt",skip_header=1) 

		r0 = shapes[:,:,0]
		for k in iterations_to_plot:
			R = shapes[:, :, k]
			#tilt = tilt_angle(R, cell_list, Nlateral)
			tilt = rotation_angles_from_inner_outer(R, r0, cell_list, Nlateral)
			total_tilt = tilt**2
			tilt_pool[k].append(total_tilt)		

	strain = []
	C = []

	for k in iterations_to_plot:
		values2 = np.array(tilt_pool[k])
		area_growth = data_log[k,0]
		beta = 0.5
		inner_strain = np.sqrt(1 + area_growth / (1 - beta)**2) - 1

		strain.append(inner_strain)
		C.append(np.mean(values2))

	strain = np.array(strain)
	C = np.array(C)

	onset = onset_jump_loglog(C, strain)
	if np.max(C) < 1e-4:
		onset = None

	if c_count == 0:
		ax.scatter(kt,onset, label=r"sim. ($\delta=1$)",edgecolor="k",facecolor="none",marker="^",s=15)
	else:
		ax.scatter(kt,onset,edgecolor="k",facecolor="none",marker="^",s=15)
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

plt.xlabel(r'$\tilde{k}_t$')
plt.ylabel(r'$\epsilon_c$')


plotlib.set_position(ax,x=0.27,y=0.20,width=0.68,height=0.75)

# Save the figure
fig.savefig('./Plots/onset_vs_kt.pdf',dpi=600,transparent=True)

plt.show()