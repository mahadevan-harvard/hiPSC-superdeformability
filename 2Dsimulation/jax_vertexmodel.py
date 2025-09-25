import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

import fire2_circular_norot as fire

np.random.seed(0)
key = jax.random.PRNGKey(0)  # You can seed this for reproducibility

################################################
# Functions
################################################

def affine_deformation(nodes, R_current, R_new):
	"""
	Apply affine deformation to a set of nodes such that the sphere expands
	from radius R_current to R_new.
	
	Parameters:
	- nodes (jnp.ndarray): Array of shape (N, 3) containing the (x, y, z) coordinates of nodes.
	- R_current (float): Current radius of the sphere.
	- R_new (float): Desired new radius of the sphere.
	
	Returns:
	- jnp.ndarray: Deformed nodes with updated coordinates.
	"""
	scale_factor = R_new / R_current  # Compute scaling factor
	
	# Center the nodes around the origin, scale, and return to original position
	deformed_nodes = nodes * scale_factor
	
	return deformed_nodes

def area_conserving_affine_deformation(r, constrained, R_outer_new, A_cells):
	"""
	Area-conserving affine deformation for outer and inner nodes.

	Parameters:
	- r: (N, 2) array of node positions
	- constrained: boolean mask for outer nodes
	- R_outer_new: new outer radius
	- A_cells: area between outer and inner rings

	Returns:
	- r_new: updated positions
	"""

	radii = jnp.linalg.norm(r, axis=1)
	R_outer_old = jnp.mean(radii[constrained])
	R_inner_old = jnp.mean(radii[~constrained])

	R_inner_new = jnp.sqrt(R_outer_new**2 - A_cells / jnp.pi)

	scale_outer = R_outer_new / R_outer_old
	scale_inner = R_inner_new / R_inner_old

	scale = jnp.where(constrained, scale_outer, scale_inner)
	scale = scale[:, None]  # for broadcasting

	return r * scale


def compute_radius(positions, outer_nodes):
	"""
	Compute the radius of the outer nodes based on their positions and a mask.

	Args:
		positions: Array of shape [Nnodes, 2] containing the (x, y) coordinates of all nodes.
		outer_nodes: Boolean mask of shape [Nnodes], where True indicates an outer node.

	Returns:
		radius: The average distance of the outer nodes from the origin.
	"""
	# Extract positions of outer nodes
	outer_positions = positions[outer_nodes,:]

	# Compute the distance of each outer node from the origin
	distances = jnp.linalg.norm(outer_positions, axis=1)

	# Compute the average radius
	radius = jnp.mean(distances)

	return radius


def average_thickness(r, R_outer, cell_list, Nlateral):
	b_array = np.zeros(len(cell_list))
	for i, cell_nodes in enumerate(cell_list):
		# Basal (inner) nodes assumed at positions Nlateral and Nlateral+1
		bottom_node_1 = cell_nodes[Nlateral]
		bottom_node_2 = cell_nodes[Nlateral+1]
		midpoint_bottom = 0.5 * (r[bottom_node_1] + r[bottom_node_2])

		R_in = np.linalg.norm(midpoint_bottom)
		b_array[i] = R_outer - R_in

	return np.mean(b_array)

def average_rotation_from_inner_outer(r, r0, cell_list, Nlateral):
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

		# 1. Compute current radii
		R_out_norm = np.linalg.norm(R_out)
		R_in_norm = np.linalg.norm(R_in)
		R0_out_norm = np.linalg.norm(R0_out)
		R0_in_norm = np.linalg.norm(R0_in)

		# 4. Compute radial directions from rest positions
		radial_dir_in_aff = R0_in / R0_in_norm
		radial_dir_out_aff = R0_out / R0_out_norm

		# 5. Get affine-transformed node positions
		R_in_affine = R_in_norm * radial_dir_in_aff
		R_out_affine = R_out_norm * radial_dir_out_aff

		# 6. Compute vectors and normalize
		v_affine = R_out_affine - R_in_affine
		v_actual = R_out - R_in

		v_affine /= np.linalg.norm(v_affine)
		v_actual /= np.linalg.norm(v_actual)

		# 7. Compute angle
		cos_theta = np.clip(np.dot(v_affine, v_actual), -1.0, 1.0)
		angle = np.arccos(cos_theta)
		angles.append(angle)

	return np.mean(angles)

def rms_lateral_deflection(r, cell_list, Nlateral):
	rms_values = []

	for cell_nodes in cell_list:
		i_in = cell_nodes[Nlateral]
		i_out = cell_nodes[0]

		R_in = r[i_in]
		R_out = r[i_out]

		# Affine axis (unit vector)
		axis = R_out - R_in
		L = np.linalg.norm(axis)
		if L < 1e-12:
			rms_values.append(0.0)
			continue
		axis_unit = axis / L

		# Loop over lateral intermediate nodes
		deflections = []
		for j in range(1, Nlateral):
			node_idx = cell_nodes[j]
			p = r[node_idx] - R_in
			proj = np.dot(p, axis_unit) * axis_unit
			perp = p - proj
			deflections.append(np.linalg.norm(perp))

		deflections = np.array(deflections)
		rms = np.sqrt(np.mean(deflections**2))
		rms_values.append(rms)

	return np.mean(rms_values)

def average_excess_length(r, cell_list, Nlateral):
	excess_ratios = []

	for cell_nodes in cell_list:
		i_in = cell_nodes[Nlateral]
		i_out = cell_nodes[0]

		# Path length along lateral edge
		L = 0.0
		for j in range(Nlateral):
			n0 = cell_nodes[j]
			n1 = cell_nodes[j + 1]
			L += np.linalg.norm(r[n1] - r[n0])

		# Direct (affine) distance
		l = np.linalg.norm(r[i_out] - r[i_in])

		if l > 1e-12:
			excess = L / l - 1.0
		else:
			excess = 0.0  # or np.nan

		excess_ratios.append(excess)

	return np.mean(excess_ratios)

################################################
# Cell geometry
################################################
# Reset the area of a cell
def cell_area(r, cell_nodes):
	# Extract positions for the given cell
	R_cell = r[cell_nodes]
	R_next = np.roll(R_cell, 1, axis=0)  # Shift nodes for pairwise calculations
	cross_product = R_next[:, 0] * R_cell[:, 1] - R_next[:, 1] * R_cell[:, 0]
	area = 0.5 * np.abs(jnp.sum(cross_product))
	return area

def cell_angles(R, cell_nodes, mode="zipper"):
	# Extract positions for the relevant nodes in the cell
	R_cell = R[cell_nodes]

	def angle_triplet(i,j,k):
		# Extract indices for triplet ijk
		R_i, R_j, R_k = R_cell[i], R_cell[j], R_cell[k]

		# Compute vectors for the edges ij and jk
		edge_ij = R_j - R_i
		edge_jk = R_k - R_j

		# Compute lengths of edges ij and jk
		l_ij = jnp.linalg.norm(edge_ij)
		l_jk = jnp.linalg.norm(edge_jk)

		# Normalize edge vectors
		unit_ij = edge_ij / l_ij
		unit_jk = edge_jk / l_jk

		# Compute cosine of the angle and stabilize
		cos_theta_raw = jnp.dot(unit_ij, unit_jk)
		cos_theta = jnp.where(cos_theta_raw < -1.0, -1.0, 
								jnp.where(cos_theta_raw > 1.0, 1.0, cos_theta_raw))

		# Compute angle
		theta = jnp.arccos(cos_theta)

		return theta

	# Vectorize over all angle triplets in the cell
	if mode == "zipper":
		angles = np.zeros(len(cell_nodes))
		for n in range(6):
			i = cell_nodes[(n-1)%6]
			j = cell_nodes[n]
			k = cell_nodes[(n+1)%6]
			angles[n] = angle_triplet(i,j,k)
	elif mode == "buckle":
		N = len(cell_nodes)
		angles = np.zeros(N)
		for n in range(N):
			i = (n - 1) % N
			j = n
			k = (n + 1) % N
			angles[n] = angle_triplet(i, j, k)		
	else:
		angles = np.zeros(len(cell_nodes))
		for n in range(4):
			i = cell_nodes[(n-1)%4]
			j = cell_nodes[n]
			k = cell_nodes[(n+1)%4]
			angles[n] = angle_triplet(i,j,k)		
	return angles

################################################
# Physics definitions
################################################

def generate_area_energy_fn(cell_list, k, A0):
	def area_energy_fn(R):
		"""
		Calculate the total area energy for all cells using vectorized operations.
		
		Args:
			R: (Nnodes, 2) Positions of all nodes.
			cell_list: (Ncells, 4) Indices of nodes for each cell.
			k: Area energy coefficient.
			A0: Target area for each cell.
		
		Returns:
			Total area energy.
		"""
		def cell_area(cell_nodes):
			# Extract positions for the given cell
			R_cell = R[cell_nodes]
			R_next = jnp.roll(R_cell, 1, axis=0)  # Shift nodes for pairwise calculations
			cross_product = R_next[:, 0] * R_cell[:, 1] - R_next[:, 1] * R_cell[:, 0]
			area = 0.5 * jnp.abs(jnp.sum(cross_product))
			return area

		# Vectorize the area computation and energy evaluation
		areas = jax.vmap(cell_area)(cell_list)

		# Broadcast A0 if it's a scalar
		A0_broadcast = A0 if isinstance(A0, jnp.ndarray) and A0.ndim == 1 else jnp.full_like(areas, A0)

		total_energy = jnp.sum(0.5 * k * (areas - A0_broadcast) ** 2)

		return total_energy
	
	return area_energy_fn


def generate_tilt_angle_energy_fn(cell_list, cell_edge_list, rest_angles, k_a, Nlateral):
	def tilt_angle_energy_fn(R, l0_list):
		"""
		Calculate the tilt energy based on apical and basal corner angles.

		Args:
			R: (Nnodes, 2) Positions of all nodes.
			cell_list: (Ncells, Nnodes_per_cell) Node indices per cell.
			cell_edge_list: (Ncells, Nedges_per_cell) Edge indices per cell.
			l0_list: (Nedges,) Rest lengths of edges.
			rest_angles_corner: (2,) Desired angles at basal and apical corners.
			k_a: Prefactor for effective tilt resistance.
			Nlateral: Number of segments in lateral edge (implies Nlateral+1 nodes).

		Returns:
			Total tilt energy.
		"""
		Ncells = cell_list.shape[0]

		def cell_tilt_energy(i, all_cell_nodes, all_cell_edges):
			# Extract positions for the relevant nodes in the cell
			#R_cell = R[cell_nodes]
			cell_nodes = jnp.array([
							all_cell_nodes[0],
							all_cell_nodes[Nlateral],
							all_cell_nodes[Nlateral+1],
							all_cell_nodes[-1]
						  ])
			selected_l0 = jnp.array([
							jnp.sum(l0_list[all_cell_edges[:Nlateral]]),
							l0_list[all_cell_edges[Nlateral]],
							jnp.sum(l0_list[all_cell_edges[Nlateral+1:2*Nlateral+1]]),
							l0_list[all_cell_edges[-1]]
						  ])

			# Select correct rest angles
			rest_angles_cell = rest_angles if rest_angles.ndim == 1 else rest_angles[i]

			def angle_triplet_energy(n, rest_angle):
				# Get indices of the angle triplet i, j, k
				i = cell_nodes[jnp.mod(n - 1, len(cell_nodes))]
				j = cell_nodes[n]
				k = cell_nodes[jnp.mod(n + 1, len(cell_nodes))]

				# Get node positions
				R_i, R_j, R_k = R[i], R[j], R[k]

				# Compute vectors for the edges ij and jk
				edge_ij = R_j - R_i
				edge_jk = R_k - R_j

				# Fetch rest lengths for edges ij and jk
				l0_ij = selected_l0[jnp.mod(n - 1, len(selected_l0))]
				l0_jk = selected_l0[n]

				# Compute lengths of edges ij and jk
				l_ij = jnp.linalg.norm(edge_ij)
				l_jk = jnp.linalg.norm(edge_jk)

				# Normalize edge vectors
				unit_ij = edge_ij / l_ij
				unit_jk = edge_jk / l_jk

				# Compute cosine of the angle and stabilize
				cos_theta = jnp.dot(unit_ij, unit_jk)
				sin_theta = jnp.cross(unit_ij, unit_jk)  # In 2D this gives scalar "signed area"
				theta = jnp.arctan2(sin_theta, cos_theta)

				# Compute angle
				theta = jnp.abs(theta)#jnp.arccos(cos_theta)

				# Compute spring stiffness coefficient
				kappa = k_a / (l0_ij + l0_jk)

				# Compute harmonic potential
				energy = 0.5 * kappa * (theta - rest_angle) ** 2
				return energy

			# Vectorize over all angle triplets in the cell
			angle_indices = jnp.arange(len(cell_nodes))
			angle_energies = jax.vmap(angle_triplet_energy)(angle_indices, rest_angles_cell)
			
			return jnp.sum(angle_energies)

		# Compute angle energy for each cell
		cell_indices = jnp.arange(Ncells)
		cell_energies = jax.vmap(cell_tilt_energy)(cell_indices, cell_list, cell_edge_list)

		# Total angle energy
		total_energy = jnp.sum(cell_energies)
		return total_energy
	return tilt_angle_energy_fn

def generate_lateral_angle_energy_fn(cell_list, cell_edge_list, rest_angles, k_a, Nlateral):
	def lateral_angle_energy_fn(R, l0_list):
		"""
		Calculate the tilt energy based on apical and basal corner angles.

		Args:
			R: (Nnodes, 2) Positions of all nodes.
			cell_list: (Ncells, Nnodes_per_cell) Node indices per cell.
			cell_edge_list: (Ncells, Nedges_per_cell) Edge indices per cell.
			l0_list: (Nedges,) Rest lengths of edges.
			rest_angles_corner: (2,) Desired angles at basal and apical corners.
			k_a: Prefactor for effective tilt resistance.
			Nlateral: Number of segments in lateral edge (implies Nlateral+1 nodes).

		Returns:
			Total tilt energy.
		"""
		Ncells = cell_list.shape[0]

		def cell_lateral_energy(i, all_cell_nodes, all_cell_edges):
			# Extract positions for the relevant nodes in the cell
			#R_cell = R[cell_nodes]
			cell_nodes = all_cell_nodes[:Nlateral+1]
			selected_l0 = l0_list[all_cell_edges[:Nlateral]]

			# Select correct rest angles
			rest_angles_cell = rest_angles[i,1:Nlateral]

			def angle_triplet_energy(n, rest_angle):
				# Get indices of the angle triplet i, j, k
				i = cell_nodes[n-1]
				j = cell_nodes[n]
				k = cell_nodes[n+1]

				# Get node positions
				R_i, R_j, R_k = R[i], R[j], R[k]

				# Compute vectors for the edges ij and jk
				edge_ij = R_j - R_i
				edge_jk = R_k - R_j

				# Fetch rest lengths for edges ij and jk
				l0_ij = selected_l0[n-1]
				l0_jk = selected_l0[n]

				# Compute lengths of edges ij and jk
				l_ij = jnp.linalg.norm(edge_ij)
				l_jk = jnp.linalg.norm(edge_jk)

				# Normalize edge vectors
				unit_ij = edge_ij / l_ij
				unit_jk = edge_jk / l_jk

				# Compute cosine of the angle and stabilize
				cos_theta = jnp.dot(unit_ij, unit_jk)
				sin_theta = jnp.cross(unit_ij, unit_jk)  # In 2D this gives scalar "signed area"
				theta = jnp.arctan2(sin_theta, cos_theta)

				# Compute angle
				theta = jnp.abs(theta)

				# Compute spring stiffness coefficient
				kappa = 2*k_a / (l0_ij + l0_jk) # @@@ 2 to account for lateral edge of the current and neighbouring cell

				# Compute harmonic potential
				energy = 0.5 * kappa * (theta - rest_angle) ** 2
				return energy

			# Vectorize over all angle triplets in the cell
			angle_indices = jnp.arange(1,Nlateral)#len(cell_nodes))
			angle_energies = jax.vmap(angle_triplet_energy)(angle_indices, rest_angles_cell)
			
			return jnp.sum(angle_energies)

		# Compute angle energy for each cell
		cell_indices = jnp.arange(Ncells)
		cell_energies = jax.vmap(cell_lateral_energy)(cell_indices, cell_list, cell_edge_list)

		# Total angle energy
		total_energy = jnp.sum(cell_energies)
		return total_energy
	return lateral_angle_energy_fn

def generate_spring_energy_fn(edge_list, k_list):
	def spring_energy_fn(R, l0_list):
		"""
		Calculate the total harmonic spring energy for a set of edges.

		Args:
			R: (Nnodes, 2) Positions of all nodes.
			edge_list: (Nedges, 2) List of node index pairs defining the springs.
			k_list: (Nedges,) Spring stiffness coefficients.
			l0_list: (Nedges,) Rest lengths of the springs.

		Returns:
			Total harmonic spring energy.
		"""
		def spring_energy(edge, k, l0):
			# Extract node indices for the spring
			node1, node2 = edge
			R1, R2 = R[node1], R[node2]

			# Compute the distance between the nodes
			distance = jnp.linalg.norm(R2 - R1)

			# Harmonic spring energy
			energy = 0.5 * k / l0 * (distance - l0) ** 2
			return energy

		# Compute spring energy for each edge
		spring_energies = jax.vmap(spring_energy)(edge_list, k_list, l0_list)

		# Total harmonic spring energy
		total_energy = jnp.sum(spring_energies)

		return total_energy
	return spring_energy_fn

def generate_lateral_spring_energy_fn(edge_list, k_list):
	def lateral_spring_energy_fn(R, lateral_mask,  l0_list):
		"""
		Calculate the total harmonic spring energy for a set of edges.

		Args:
			R: (Nnodes, 2) Positions of all nodes.
			edge_list: (Nedges, 2) List of node index pairs defining the springs.
			k_list: (Nedges,) Spring stiffness coefficients.
			l0_list: (Nedges,) Rest lengths of the springs.

		Returns:
			Total harmonic spring energy.
		"""
		def spring_energy(edge, k, l0):
			# Extract node indices for the spring
			node1, node2 = edge
			R1, R2 = R[node1], R[node2]

			# Compute the distance between the nodes
			distance = jnp.linalg.norm(R2 - R1)

			# Harmonic spring energy
			energy = 0.5 * k / l0 * (distance - l0) ** 2
			return energy

		# Compute spring energy for each edge
		spring_energies = jax.vmap(spring_energy)(edge_list, k_list, l0_list)
		spring_energies = jnp.where(lateral_mask,spring_energies,0.0)

		# Total harmonic spring energy
		total_energy = jnp.sum(spring_energies)

		return total_energy
	return lateral_spring_energy_fn

###################################
# Minimization core
###################################

# Initialize FIRE minimization
def run_minimization(force_fn, R_init, l0_list, invmass, constraint, i_fix, radius, max_frms_thresh = 1e-12, max_num_steps=1000000, **kwargs):

	# FIRE parameters
	dt_start, dt_min, dt_max = 0.001, 0.001, 0.01

	# Initialize FIRE descent
	init, apply = fire.fire_descent(force_fn, dt_start=dt_start, dt_min=dt_min, dt_max=dt_max)
	apply = jax.jit(apply)
	
	@jax.jit
	def cond_fn(val):
		state, i = val
		return jnp.logical_and(state.Frms > max_frms_thresh, i<max_num_steps)

	@jax.jit
	def body_fn(val):
		state, i = val
		state = apply(state, l0_list=l0_list)				
		return state, i+1

	state = init(R_init, invmass, constraint, i_fix, radius, l0_list=l0_list)
	state, num_iterations = jax.lax.while_loop(cond_fn, body_fn, (state, 0))

	Frms = state.Frms
	positions = state.position

	return positions, Frms, num_iterations