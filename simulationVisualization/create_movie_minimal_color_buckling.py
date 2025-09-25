import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Load data from npz file
name = "kl_1.0_ko_1.0_ka_5e-06_kt_5e-06_al_0.0_Ri_1.6_beta_0.5_N_20_d_1.0_dcount9_V001"
with h5py.File(f"./Data/2Dsim_lin/shape_{name}.h5", "r") as h5:
	shapes = h5["data"][:] 
	cell_list = h5["cell_list"][:]
	edge_list = h5["edge_list"][:]
	outer_nodes = h5["outer_nodes"][:]
# data = np.load(f"Data/{name}.npz")
# shapes = data["data"]  # Shape: (Nnodes, 2, Niterations)
# edge_list = data["edge_list"]  # List of edges, shape: (Nedges, 2)
# outer_nodes = data["outer_nodes"]  # List of outer nodes

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
	ax.set_title(rf"$\epsilon =$ {inner_strain:.2f}",fontsize=6, pad=0)

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

# Create animation
anim = FuncAnimation(fig, update, frames=Niterations, interval=200, blit=True)

# Save as a movie
anim.save(f"./Plots/{name}_1.mp4", writer="ffmpeg",dpi=600, fps=30)

# Optional: Display the animation

plt.show()