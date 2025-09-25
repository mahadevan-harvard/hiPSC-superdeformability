import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load data from npz file
name = "kl_1.0_ko_1.0_ka_0.5_kt_0.5_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_dcount9_V001"
with h5py.File(f"./Data/2Dsim_lin/shape_{name}.h5", "r") as h5:
	shapes = h5["data"][:] 
	cell_list = h5["cell_list"][:]
	edge_list = h5["edge_list"][:]
	outer_nodes = h5["outer_nodes"][:]

inner_nodes = ~outer_nodes

# Parameters
Niterations = shapes.shape[2]

# Initialize the plot
fig, ax = plt.subplots(figsize=(1,1))
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0) 
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_aspect('equal')  # Set the aspect ratio to equal
ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks
ax.set_frame_on(False)  # Remove the axis border

box_flag = False
if box_flag:
	box = Rectangle((-0.5, 1.0), 1, 7, linewidth=1.5, edgecolor='grey', facecolor='lightgrey',zorder=1)
	ax.add_patch(box)

magenta_color = "orchid"
green_color = "olivedrab"

inner_plot, = ax.plot([], [], 'o', c="darkorange" , markersize=2, label='Inner nodes',zorder=3,alpha=0)
outer_plot, = ax.plot([], [], 'o', c="royalblue" ,markersize=2, label='Outer nodes',zorder=3,alpha=0)

edges_plot, = ax.plot([], [], '-', c="k",linewidth=0.25)
#edges_plot, = ax.plot([], [], '-', c=green_colr, linewidth=.5)

def update(frame):
	"""Update function for each frame."""
	beta = 0.5
	area_growth = frame*0.005
	inner_strain = np.sqrt(1 + area_growth/(1-beta)**2) - 1

	# Current nodes positions
	nodes = shapes[:, :, frame]
	inner_nodes_coords = nodes[inner_nodes]
	inner_plot.set_data(inner_nodes_coords[:, 0], inner_nodes_coords[:, 1])

	# Outer and junction nodes
	outer_nodes_coords = nodes[outer_nodes]
	outer_plot.set_data(outer_nodes_coords[:, 0], outer_nodes_coords[:, 1])

   # Edges
	x_edges, y_edges = [], []
	for edge in edge_list:
		x_edges.extend([nodes[edge[0], 0], nodes[edge[1], 0], None])  # None for line break
		y_edges.extend([nodes[edge[0], 1], nodes[edge[1], 1], None])
	edges_plot.set_data(x_edges, y_edges)

	return inner_plot, outer_plot, edges_plot

frame = 78  # Choose the frame you want

# --- Parameters ---
R0_inner = 1.6
beta = 0.5
area_growth = frame * 0.005

# --- Derived quantities ---
R0_outer = R0_inner / beta
inner_strain = np.sqrt(1 + area_growth / (1 - beta)**2) - 1
R_inner = R0_inner * (1 + inner_strain)
R_outer = R0_outer * np.sqrt(1 + area_growth)

# --- Area-preserving remap of nodes0 ---
nodes0 = shapes[:, :, 0]
r0 = np.linalg.norm(nodes0, axis=1)
r0_safe = np.where(r0 == 0, 1e-12, r0)
direction = nodes0 / r0_safe[:, None]

r_mapped = np.sqrt(
	R_inner**2 + (r0**2 - R0_inner**2) * ((R_outer**2 - R_inner**2) / (R0_outer**2 - R0_inner**2))
)
mapped_nodes0 = direction * r_mapped[:, None]

# --- Plot mapped edges ---,transparent=True
x0, y0 = [], []
for edge in edge_list:
	x0.extend([mapped_nodes0[edge[0], 0], mapped_nodes0[edge[1], 0], None])
	y0.extend([mapped_nodes0[edge[0], 1], mapped_nodes0[edge[1], 1], None])

update(frame)
plt.savefig(f"./Plots/{name}_frame{frame}.svg", dpi=600)

# Optional: Display the animation

plt.show()