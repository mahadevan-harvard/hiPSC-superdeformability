import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib

import plot_utils.PlotLibrary as plotlib

cmap = matplotlib.colormaps['Reds']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('NatMat')
figSpecs.set_figureHeight(30)

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


# ######################################################################
# # SIMULATION (no disorder)
# ######################################################################

# Load data from npz file
V_range = np.array([1])
kt_range = np.array([0.005])
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
		with h5py.File(f"./Data/2Dsim_log/shape_{name}.h5", "r") as h5:
			shapes = h5["data"][:] 
			cell_list = h5["cell_list"][:]
			Niterations = shapes.shape[2]

			data_log = np.genfromtxt(f"Data/2Dsim_log/data_{name}.txt",skip_header=1) 

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

	smooth = 5

	eps = np.asarray(strain, float)
	T   = np.asarray(C,   float)

	# keep finite, ε>0, T>0
	m = np.isfinite(eps) & np.isfinite(T) & (eps > 0)
	eps, T = eps[m], T[m]
	o = np.argsort(eps)
	eps, T = eps[o], T[o]

	# floors to avoid log(0)
	eps_floor = 1e-12
	T_floor   = max(1e-12, 1e-3*np.nanmax(T))

	x = np.log10(np.maximum(eps, eps_floor))
	y = np.log10(np.maximum(T,   T_floor))


	x_raw = np.copy(x)
	y_raw = np.copy(y)

	# optional smoothing (boxcar)
	if smooth > 1:
		k = int(smooth); k = k + 1 - (k % 2)  # make odd
		y = np.convolve(y, np.ones(k)/k, mode='same')

	# gradient and argmax (avoid endpoints)
	dy = np.gradient(y, x)

	if smooth > 1:
		i = smooth + int(np.argmax(dy[smooth:-smooth]))
	else:
		i = 1 + int(np.argmax(dy[1:-1]))

	ax.plot(x[smooth:-smooth],dy[smooth:-smooth],color="k",lw=1,label="smoothed data")
	ax.scatter(x[i],dy[i],edgecolor="k",facecolor="none",marker="o",s=15,label="onset")
	c_count +=1

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,halfstyle=True)	  
leg = plotlib.set_legend(ax,pos=1)


plt.xlabel(r'$\log_{10}{\epsilon}$')
plt.ylabel(r'$\frac{d\,\log_{10}{\langle T \rangle}}{d\,\log_{10}{\epsilon}}$')

plotlib.set_position(ax,x=0.25,y=0.30,width=0.68,height=0.65)

# Save the figure
fig.savefig('./Plots/onset_example.pdf',dpi=600,transparent=True)

plt.show()