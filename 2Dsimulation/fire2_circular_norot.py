"""
This version includes rigid bodies

It uses the Fire 2.0 protocol as described in https://www.sciencedirect.com/science/article/pii/S0927025620300756
"""

from collections import namedtuple

from typing import TypeVar, Callable, Tuple, Union, Any

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce

from jax_md import dataclasses
from jax_md import util
from jax_md import quantity


# Types
PyTree = Any
Array = util.Array
f32 = util.f32
f64 = util.f64
i32 = util.i32
T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Minimizer = Tuple[InitFn, ApplyFn]


# FireDescentState Dataclass
@dataclasses.dataclass
class FireDescentState:
	"""A dataclass containing state information for the FIRE Descent minimizer."""
	position: Array
	velocity: Array
	force: Array
	invmass: Array
	constraint: Array
	i_fix: int
	radius: float
	dt: float
	alpha: float
	Ncount: int
	Frms: float


# FIRE Descent Function
def fire_descent(energy_or_force: Callable[..., Array],
				 dt_start: float = 0.1,
				 dt_min: float = 0.002, # 0.02*DT recommened
				 dt_max: float = 1.0, # 10*DT recommended
				 Nmin: int = 5,
				 f_inc: float = 1.1,
				 f_dec: float = 0.5,
				 alpha_start: float = 0.1,
				 f_alpha: float = 0.99) -> Minimizer[FireDescentState]:
	"""
	FIRE minimization based on jaxMD, but with custom implementation.
	Which, amongst others, uses the velocity, and allows special constraints.
	Args:
		force_fn: A function that computes forces given positions.
		dt_start: Initial timestep.
		dt_max: Maximum timestep.
		Nmin: Minimum steps in the correct direction before updating `dt` and `alpha`.
		f_inc: Step size increment factor.
		f_dec: Step size reduction factor.
		alpha_start: Initial mixing parameter.
		f_alpha: Mixing parameter reduction factor.

	Returns:
		A tuple (init_fn, apply_fn) where:
			- init_fn: Initializes the FireDescentState.
			- apply_fn: Applies one step of the FIRE minimization.
	"""

	dt_start, dt_min, dt_max, Nmin, f_inc, f_dec, alpha_start, f_alpha = util.static_cast(
	dt_start, dt_min, dt_max, Nmin, f_inc, f_dec, alpha_start, f_alpha)

	force = quantity.canonicalize_force(energy_or_force)

	#  def init_fn(R: PyTree, mass: Array=1.0, free: Array=True, **kwargs) -> FireDescentState:
	def init_fn(R: PyTree, invmass: Array, constraint: Array, i_fix: i32, radius: f64, **kwargs) -> FireDescentState:    
		V = tree_map(lambda x: jnp.zeros_like(x), R)
		Ncount = jnp.zeros((), jnp.int32)
		Frms = jnp.ones((), f64)
		F = force(R, **kwargs)
		state = FireDescentState(R, V, F, invmass, constraint, i_fix, radius, dt_min, alpha_start, Ncount, Frms)  # pytype: disable=wrong-arg-count
		return state

	def apply_fn(state: FireDescentState, **kwargs) -> FireDescentState:

		state = integrate_EI(state, force, **kwargs)
		R, V, F, invmass, constraint, i_fix, radius, dt, alpha, Ncount, Frms = dataclasses.unpack(state)

		# Measure FIRE
		Frms, F_dot_V = measure_FIRE(F, V, constraint)

		Ncount = jnp.where(F_dot_V >= 0, Ncount + 1, 0)	
		dt_choice = jnp.array([dt * f_inc, dt_max])

    	# if power > 0.0 and if Ncount> Nmin
		dt = jnp.where(F_dot_V > 0,
					jnp.where(Ncount > Nmin,
								jnp.min(dt_choice),
								dt),
					dt)
		alpha = jnp.where(F_dot_V > 0,
						jnp.where(Ncount > Nmin,
									alpha * f_alpha,
									alpha),
						alpha)

	    # if power < 0.0
		dt_choice = jnp.array([dt * f_dec, dt_min])
		dt = jnp.where(F_dot_V < 0, jnp.max(dt_choice), dt)
		alpha = jnp.where(F_dot_V < 0, alpha_start, alpha)

		# Halfstep back implementation
		#R = jnp.where(F_dot_V < 0, R - 0.5 * dt * V, R)
		V = jnp.where(F_dot_V < 0, jnp.zeros_like(V), V)	# Reset velocities

		return FireDescentState(R, V, F, invmass, constraint, i_fix, radius, dt, alpha, Ncount, Frms)  # pytype: disable=wrong-arg-count
	return init_fn, apply_fn


def integrate_EI(state: FireDescentState, force_fn: Callable, **kwargs) -> FireDescentState:
	"""
	Integrate by the Euler Implicit algorithm, using the state object.
	
	step 1: $v(t+dt)=v(t)+f(t))/m *dt$
	apply mixing step
	step 2: $x(t+dt) = x(t) + v(t)*dt$
	step 3: calculate forces at t+dt

	Args:
		state: FireDescentState containing position, velocity, force, etc.
		force_fn: Function to compute forces given positions.
	
	Returns:
		Updated FireDescentState after integration.
	"""
	# Extract state variables
	r, V, F, invmass, constraint, i_fix, radius, dt, alpha = (
		state.position, state.velocity, state.force, 
		state.invmass, state.constraint, state.i_fix, state.radius, state.dt, state.alpha
	)

	# Broadcast arrays for constrained and free nodes
	constrained = jnp.broadcast_to(constraint[:, None], V.shape)
	unconstrained = ~constrained
	invmass_broadcast = jnp.broadcast_to(invmass[:, None], F.shape)

	# Step 1a: Update velocities for unconstrained nodes
	V = V + jnp.where(unconstrained, F * invmass_broadcast * dt, 0.0)

    # Step 1b: Update angular velocity and angle for constrained nodes
	angles = jnp.where(
		jnp.logical_and(r[:, 0] == 0.0, r[:, 1] == 0.0), 
		0.0, 
		jnp.arctan2(r[:, 1], r[:, 0])
	)

	torque = r[:, 0] * F[:, 1] - r[:, 1] * F[:, 0]  # Torque is r x F (z-component for 2D)
	angular_velocity = (r[:, 0] * V[:, 1] - r[:, 1] * V[:, 0]) / radius**2  # Tangential velocity converted to angular velocity
	angular_acceleration = torque / (invmass * radius**2)  # α = τ / I
	angular_velocity = angular_velocity + jnp.where(constraint, angular_acceleration * dt, 0.0)

	# Zero out for fixed node
	torque = torque.at[i_fix].set(0.0)
	angular_velocity = angular_velocity.at[i_fix].set(0.0)

	# Mixing step
	angular_velocity = update_angularvelocity(alpha, angular_velocity, torque)
	V = update_vel(alpha, V, F)

	# Step 2a: Update positions for unconstrained nodes
	r = r + jnp.where(unconstrained, V * dt, 0.0)

    # Step 2b: Update angles and positions for constrained nodes
	angles += angular_velocity * dt  # Update angle for constrained nodes
	r = jnp.where(
        constrained,
        jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1),
        r
    )

	# Step 3: Recalculate forces
	F = force_fn(r, **kwargs)

	# Remove normal components for constrained nodes
	tangential_direction = jnp.stack([-r[:, 1], r[:, 0]], axis=1) / radius  # Tangential unit vector
	tangential_velocity = angular_velocity[:, None] * radius * tangential_direction  # Projected tangential velocity
	F_tangential = jnp.sum(F * tangential_direction, axis=1, keepdims=True) * tangential_direction

	V_unconstrained = jnp.where(unconstrained, V, 0.0)
	V_constrained = jnp.where(constrained, tangential_velocity, 0.0)
	V = V_unconstrained + V_constrained	

	F_unconstrained = jnp.where(unconstrained, F, 0.0)
	F_constrained = jnp.where(constrained, F_tangential, 0.0)
	F = F_unconstrained + F_constrained

	# Zero V and F at the fixed node index
	V = V.at[i_fix].set(jnp.array([0.0, 0.0]))
	F = F.at[i_fix].set(jnp.array([0.0, 0.0]))

	state = state.set(position=r, force=F, velocity=V)

	# Return updated state
	return state


def update_vel(alpha, vel, force):
	"""
	Update velocities according to FIRE-rule.
	"""

	mod2f = jnp.sum(force**2,axis=1)
	invmodf = jnp.where(mod2f > 0.0, 1.0 / jnp.sqrt(mod2f), 0.0)

	mod2v = jnp.sum(vel**2, axis=1)
	modv = jnp.where(mod2v > 0.0, jnp.sqrt(mod2v), 0.0)

	vel = (1.0 - alpha) * vel + alpha * modv[:, None] * force * invmodf[:, None]

	return vel

def update_angularvelocity(alpha, angvel, torque):
    """
    Update angular velocities according to FIRE-rule.

    Args:
        alpha: Mixing parameter.
        angvel: Angular velocity array.
        torque: Torque array.

    Returns:
        Updated angular velocity array.
    """
    mod2t = torque**2
    invmodt = jnp.where(mod2t > 0.0, 1.0 / jnp.sqrt(mod2t), 0.0)

    mod2v = angvel**2
    modv = jnp.where(mod2v > 0.0, jnp.sqrt(mod2v), 0.0)

    angvel = (1.0 - alpha) * angvel + alpha * modv * torque * invmodt
    return angvel

def measure_FIRE(F, V, constraint):
	"""
	Measure FIRE metrics: RMS force and power for the FIRE algorithm.

	Args:
		F: Forces on all nodes, shape [Nnodes, spatial_dim].
		V: Velocities of all nodes, shape [Nnodes, spatial_dim].
		constraint: Boolean mask indicating which nodes are constrained.

	Returns:
		Frms: Root mean square of the force over free degrees of freedom.
		F_dot_V: Power (dot product of force and velocity).
	"""
	# Calculate the number of degrees of freedom
	num_dofs = 2 * len(F) - jnp.sum(constraint) - 1

	# Compute RMS force (Frms)
	Frms = jnp.sqrt(jnp.sum(F ** 2) / num_dofs)

	# Compute Power (F_dot_V)
	F_dot_V = jnp.sum(F * V)

	return Frms, F_dot_V