"""
- Create a random array by displacing the top and bottom nodes up to 0.5*the average cell width (using the alpha)
- Make sure rest-lengths are accessed individually (probably there)
- Make sure rest-areas are accessed individualy
- Make sure rest-angles are accessed individually
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax_vertexmodel as jv

# Configure JAX
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_platform_name", "cpu")

# Seed
np.random.seed(0)
key = jax.random.PRNGKey(0)  # You can seed this for reproducibility

################################################
# Parameter setup
################################################
#shape_kl_1.0_ka_1e-05_kt_0.0_al_0.01_Ri_1.6_beta_0.5_N_20_d_1.0_disorder2

# kt: 5e-4 until 5e-1
# kb: 5e-4 until 5e-1
# === Sweep parameters ===
k_theta_range = np.array([0.5])#0.00001 # 0.000008,0.000013,0.00002,0.000032,0.0008,0.0013,0.002,0.0032,
k_tilt_range = np.array([0.08,0.13,0.2,0.32])     	# angular spring constant #2.2e-4,4.6e-4,2.2e-3,4.6e-3])#])#
alpha_lateral = 0.0     		# reinforcement
V_range = np.array([1])#,2,3])#,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
disorder = 1.0

# === Fixed geometric parameters ===
d = 1.0                 # Cell size
R0_in = 1.6             # Inner radius
beta = 0.5              # Aspect ratio control

# === Lateral edge discretization ===
Nlateral = 20  # Number of segments between inner and outer edge

# === Elastic parameters ===
k_A = 100.0
k_actin_outer = 1.0
k_actin_inner = 1.0
k_lateral = 1.0

# === Growth settings ===
growth = 0.75
growth_steps = 150
Niterations = growth_steps + 1

# === Solver settings ===
max_iter = 1000000
max_Frms = 1e-9

################################################
# Derived quantities
################################################

A0 = d**2                              # Preferred cell area
H0 = R0_in * beta / (1 - beta)         # Effective height of shell
Area = np.pi * (H0**2 + 2*R0_in*H0)    # Shell area
Ncells = int(np.round(Area / A0))      # Number of cells

# Radius of the cortinoid
Acells = Ncells*A0						# Total area of the cells
A0_in = np.pi*(R0_in**2)				# Initial lumen area
Atot = A0_in + Acells					# Total area of the cortinoid
R0 = np.sqrt(Atot/np.pi)            	# Filament radius based on cell + junction size
dalpha = np.pi / (Ncells)		    	# Half width of a cell along the curve in radians

################################################
# Main loop
################################################

for V in V_range:

	################################################
	# Initial conditions
	################################################

	# Generate array to store positions
	Nnodes_assigned_percell = (Nlateral + 1)
	Nedges_assigned_percell = (Nlateral + 2)

	Nnodes = Ncells*Nnodes_assigned_percell
	Nedges = Ncells*Nedges_assigned_percell

	Nnodes_percell = 2*Nlateral+2
	Nedges_percell = 2*Nlateral+2

	np_r = np.zeros((Nnodes,2))
	np_constrained = np.zeros(Nnodes, dtype=bool)
	np_inner_nodes = np.zeros(Nnodes, dtype=bool)
	np_cell_list = np.zeros((Ncells,Nnodes_percell),dtype=int) # Which nodes belongs to which cells
	np_cell_edge_list = np.zeros((Ncells,Nedges_percell),dtype=int) # Which edges belongs to which cells

	# Generate arrays for springs
	np_k = np.zeros(Nedges)
	np_l0 = np.zeros(Nedges)
	np_edge_list = np.zeros([Nedges,2],dtype=int)
	np_lateral_edge_list = np.zeros(Nedges,dtype=bool)

	# ASSIGN NODES TO CELLS
	# Calculate midpoints based on alpha. Assign position based on fractional arclengh
	# The total arclength is related to both Ncells*d and l0*d, so a single cell covers a fraction 
	disalpha_inner = dalpha + dalpha*np.random.uniform(-disorder, disorder, size=Ncells)
	disalpha_outer = dalpha + dalpha*np.random.uniform(-disorder, disorder, size=Ncells)

	edge_ctr = 0 
	for i in range(Ncells):
		np_cell_list[i] = np.arange(Nnodes_assigned_percell*i,Nnodes_assigned_percell*i+Nnodes_percell,1,dtype=int) % Nnodes
		for j in range(Nlateral + 1):
			np_cell_list[i, j] = Nnodes_assigned_percell*i + Nnodes_assigned_percell - 1 - j 
		if i == Ncells-1:
			for j in range(Nlateral+1):
				np_cell_list[i, Nlateral+1 + j] = j			

		alpha_inner = (2 * np.pi) * (i / Ncells) - disalpha_inner[i]
		alpha_outer = (2 * np.pi) * (i / Ncells) - disalpha_outer[i]

		# Compute endpoints
		inner_pos = np.array([
			R0_in * np.sin(alpha_inner),
			R0_in * np.cos(alpha_inner)
		])
		outer_pos = np.array([
			R0 * np.sin(alpha_outer),
			R0 * np.cos(alpha_outer)
		])

		# Interpolate between inner and outer positions of the lateral edge
		for j in range(Nlateral + 1):
			node_idx = i * (Nlateral + 1) + j
			frac = j / Nlateral
			pos = (1 - frac) * inner_pos + frac * outer_pos

			np_r[node_idx] = pos
			np_constrained[node_idx] = (j == Nlateral)			

		# ASSIGN EDGES
				
		# IR to IL
		np_edge_list[edge_ctr] = np.array([np_cell_list[i,Nlateral],np_cell_list[i,Nlateral+1]])
		np_k[edge_ctr] = k_actin_inner
		np_l0[edge_ctr] = d
		edge_ctr += 1

		for j in range(Nlateral):
			n0 = i*(Nlateral+1) + j
			n1 = i*(Nlateral+1) + j+1
			np_edge_list[edge_ctr] = [n0, n1]
			np_k[edge_ctr] = 2*k_lateral 							# To account from contributions from both the left and right cell
			np_l0[edge_ctr] = np.linalg.norm(np_r[n1] - np_r[n0])
			edge_ctr += 1

		# O to O
		np_edge_list[edge_ctr] = np.array([np_cell_list[i,-1],np_cell_list[i,0]])
		np_k[edge_ctr] = k_actin_outer
		np_l0[edge_ctr] = d
		edge_ctr += 1

		k = 0
		if i == 0:
			start_edge = Nedges_assigned_percell*(Ncells-1) + Nlateral
		else:
			start_edge = Nedges_assigned_percell*(i-1) + Nlateral
		for j in range(Nlateral):
			np_cell_edge_list[i, k] = start_edge - j
			k+=1
		start_edge = Nedges_assigned_percell*i	
		for j in range(Nedges_assigned_percell):
			np_cell_edge_list[i, k] = Nedges_assigned_percell*i + j
			k+=1

	# Reset the rest area based on current positions
	np_A0_list = np.zeros(Ncells)
	for i in range(Ncells):
		np_A0_list[i] = jv.cell_area(np_r, np_cell_list[i])
	
	# Reset rest-lengths based on current positions
	np_l0 = np.sqrt(np.sum(np.power(np_r[np_edge_list[:,1]] - np_r[np_edge_list[:,0]],2),axis=1))

	# ASSIGN ANGLES
	np_all_cell_rest_angles = np.zeros((Ncells, Nnodes_percell))
	np_all_corner_rest_angles = np.zeros((Ncells, 4))

	for i in range(Ncells):
		cell_nodes = np_cell_list[i]
		np_all_cell_rest_angles[i] = jv.cell_angles(np_r, cell_nodes, mode="buckle")

		corner_select = np.array([
			cell_nodes[0],
			cell_nodes[Nlateral],
			cell_nodes[Nlateral+1],
			cell_nodes[-1]
		])
		np_all_corner_rest_angles[i] = jv.cell_angles(np_r, corner_select, mode="buckle")


	for k_tilt in k_tilt_range:
		for k_theta in k_theta_range:

			#k_theta = k_tilt
			print(f"\nRunning simulation for k_theta = {k_theta}, k_tilt = {k_tilt}. V = {V}")

			# Convert to jax numpy
			r = jnp.copy(np_r)
			r0 = jnp.copy(np_r)
			constrained = jnp.copy(np_constrained)
			cell_list = jnp.copy(np_cell_list)
			cell_edge_list = jnp.copy(np_cell_edge_list)
			invmass = jnp.ones(Nnodes)
			free = jnp.zeros(Nnodes,dtype=bool)
			edge_list = jnp.copy(np_edge_list)
			lateral_edge_list = jnp.copy(np_lateral_edge_list)		
			k_list = jnp.copy(np_k)
			l0_list = jnp.copy(np_l0)
			A0_list = jnp.copy(np_A0_list)
			all_cell_rest_angles = jnp.copy(np_all_cell_rest_angles)
			all_corner_rest_angles = jnp.copy(np_all_corner_rest_angles)

			i_fix = jnp.where(constrained)[0][0]

			################################################
			# Physics definitions
			################################################

			# --- Energy terms ---
			area_energy_fn 	= jv.generate_area_energy_fn(cell_list, k_A, A0_list)
			spring_energy_fn = jv.generate_spring_energy_fn(edge_list, k_list)
			angle_energy_fn = jv.generate_lateral_angle_energy_fn(cell_list, cell_edge_list, all_cell_rest_angles, k_theta, Nlateral) # @@@ 2 times ktheta to account for both sides
			tilt_energy_fn = jv.generate_tilt_angle_energy_fn(cell_list, cell_edge_list, all_corner_rest_angles, k_tilt, Nlateral)

			lateral_spring_energy_fn = jv.generate_lateral_spring_energy_fn(edge_list, k_list)

			# --- Combined energy ---
			def combined_energy_fn(R, l0_list=l0_list):
				return ( 	
					area_energy_fn(R) 
					+ spring_energy_fn(R, l0_list=l0_list) 
					+ angle_energy_fn(R, l0_list=l0_list)
					+ tilt_energy_fn(R, l0_list=l0_list) 
				)

			print('\n Creating initial config...')
			print('\tTotal energy of the system, U = {:f}'.format(combined_energy_fn(r)))
			print('\tArea energy of the system, U = {:f}'.format(area_energy_fn(r)))
			print('\tSpring energy of the system, U = {:f}'.format(spring_energy_fn(r, l0_list=l0_list)))
			print('\tAngular energy of the system, U = {:f}'.format(angle_energy_fn(r, l0_list=l0_list)))
			print('\tTilt energy of the system, U = {:f}'.format(tilt_energy_fn(r, l0_list=l0_list)))		
			print('\tLateral deflection energy of the system, U = {:f}'.format(0.0))		

			# --- Forces (neg. gradients) ---
			area_force = jax.grad(jax.checkpoint(area_energy_fn))
			spring_force = jax.grad(jax.checkpoint(spring_energy_fn))
			angular_force = jax.grad(jax.checkpoint(angle_energy_fn))
			tilt_force = jax.grad(jax.checkpoint(tilt_energy_fn))

			@jax.jit
			def combined_force_fn(R, l0_list):
				return (
					-area_force(R) 
					-spring_force(R, l0_list=l0_list) 
					-angular_force(R, l0_list=l0_list)
					-tilt_force(R, l0_list=l0_list)  
				)

			####################################
			# Run simulation
			####################################

			print("\nRunning main loop...")

			# --- Initialize organoid state ---
			Rorganoid = R0
			A0_organoid = jnp.pi*Rorganoid**2
			shapes = np.zeros([Nnodes,2,Niterations+1])
			shapes[:,:,0] = np.copy(r)

			factor = 0.0
			word = "log"

			factor_range = np.logspace(-3,np.log10(growth),Niterations)

			# --- Open output file ---
			filename = f"Data/data_kl_{k_lateral}_ko_{k_actin_outer}_ka_{k_theta}_kt_{k_tilt}_al_{alpha_lateral}_Ri_{R0_in}_beta_{beta}_N_{Nlateral}_d_{disorder}_{word}_V{V:03d}.txt"
			with open(filename, "w") as file:
				# Write header
				file.write("factor\tRO\tRI\tEtot\tEA\tEI\tEL\tEN\tP\tb\trot\tex_len\trms_lat\n")

				# --- Main inflation loop ---
				for gstep in range(Niterations):

					# apply affine transformation based on factor
					Rnew = jnp.sqrt(A0_organoid*(1+factor) / jnp.pi)
					r  = jv.affine_deformation(r, Rorganoid, Rnew)
					Rorganoid = Rnew

					# add random displacement
					key, subkey = jax.random.split(key)  # Update the key
					random_displacement = 0.001 * (2 * jax.random.uniform(subkey, shape=r.shape) - 1)
					r = jnp.where(constrained[:, None], r, r + random_displacement)

					# Minimize energy_V{V:03d}
					r, Frms, niters = jv.run_minimization(combined_force_fn, r, l0_list, invmass, constrained, i_fix, Rorganoid, max_frms_thresh=max_Frms, max_num_steps=max_iter)
					print(f'\t{factor:.4f}/{growth:.2f}:\t Frms = {Frms:.3g} in {niters} steps.')

					shapes[:,:,gstep+1] = np.copy(r)

					# --- Geometric measurements ---
					b = jv.average_thickness(r, Rorganoid, cell_list, Nlateral)
					rot = jv.average_rotation_from_inner_outer(r, r0, cell_list, Nlateral)
					rms_lat = jv.rms_lateral_deflection(r, cell_list, Nlateral)
					ex_len = jv.average_excess_length(r, cell_list, Nlateral)

					# --- Pressure calculation ---
					P = 0 
					
					# --- Energies and radii ---
					Etot = combined_energy_fn(r, l0_list=l0_list)
					EA = area_energy_fn(r)
					EI = spring_energy_fn(r, l0_list=l0_list)
					EL = lateral_spring_energy_fn(r, lateral_edge_list,l0_list=l0_list)
					ED = 0.0		
					RO = jv.compute_radius(r,constrained)
					RI = jv.compute_radius(r,~constrained)

					# --- Save to file ---
					file.write(f"{factor:.5f}\t{RO:.5f}\t{RI:.5f}\t{Etot:.5g}\t{EA:.5g}\t{EI:.5g}\t{EL:.5g}\t{ED:.5g}\t{P:.5g}\t{b:.5g}\t{rot:.5g}\t{ex_len:.5g}\t{rms_lat:.5g}\n")

					# --- Inflation update ---
					factor = factor_range[gstep]
					
			print("clear cache")
			jax.clear_caches()

			################################################
			# Save result
			################################################

			np.savez_compressed(f"Data/shape_kl_{k_lateral}_ko_{k_actin_outer}_ka_{k_theta}_kt_{k_tilt}_Ri_{R0_in}_beta_{beta}_N_{Nlateral}_d_{disorder}_{word}_V{V:03d}.npz",data=shapes, edge_list=np_edge_list, cell_list=np_cell_list, outer_nodes=np_constrained)