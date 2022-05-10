"""

Path tracking simulation with pure pursuit steering and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""
from jax import jit
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial
from rich.traceback import install
install(show_locals=False)

class FinishError(Exception):
    pass

# Parameters
k = 0.1   # look forward gain
Lfc = 2.0 # [m] look-ahead distance
Kp = 1.0  # speed proportional gain
b = 1.    # [m] wheel base of vehicle
m = 1.

BASE_MAX   = 10.
MASS_MAX   = 10.
STEER_MAX  = np.deg2rad(57.5) # Yeah why not
POSS_STEER = np.pi
V_MAX      = 1.
FORCE_MAX  = 1.
PARAMS = np.array([b/BASE_MAX, m/MASS_MAX, STEER_MAX/POSS_STEER, FORCE_MAX])

@jit
def dynamics(t, x, u):
    w, f = jnp.split(u, 2, axis=-1)
    # Basically we assume tanh & scaling is applied to inputs by a nn fairie
    x, y, rear_x, rear_y, yaw, sin_yaw, cos_yaw, v = jnp.split(x, 8, axis=-1)
    # Update dynamics
    x   = x + v * jnp.cos(yaw) * t
    y   = y + v * jnp.sin(yaw) * t
    yaw += v / b * jnp.tan(w) * t # The most sus :)
    v   = v + (f/m) * t           # Not too sus
    v   = jnp.minimum(v, V_MAX)
    yaw = jnp.maximum(yaw, -POSS_STEER) # Keep between -pi, pi
    yaw = jnp.minimum(yaw, POSS_STEER)
    # Repack vectors
    rear_x = x - ((b / 2) * jnp.cos(yaw))
    rear_y = y - ((b / 2) * jnp.sin(yaw))
    next_state = jnp.concatenate((x, y, rear_x, rear_y, yaw,
                                  jnp.sin(yaw), jnp.cos(yaw), v))
    return next_state

def initial_state(x=0.0, y=0.0, yaw=0.0, v=0.0):
    rear_x = x - ((b / 2) * jnp.cos(yaw))
    rear_y = y - ((b / 2) * jnp.sin(yaw))
    return jnp.array([x, y, rear_x, rear_y, yaw, jnp.sin(yaw), jnp.cos(yaw), v, 0., 0.])

def apply_noise(state, rng, state_noise):
    state = state.at[:4].set(state[:4] + rng.normal(size=(4,), scale=state_noise))

# @partial(jit, static_argnums=0)
def update(state, f, delta, dt):
    f     = jnp.tanh(f) * FORCE_MAX
    delta = jnp.tanh(delta) * STEER_MAX
    state = state.at[-2:].set(jnp.array([delta, f]))
    x_new = dynamics(dt, state[:8], state[-2:])
    state = state.at[:8].set(x_new)

def calc_distance(state, point_x, point_y):
    dx = state[2] - point_x
    dy = state[3] - point_y
    return jnp.hypot(dx, dy)

# @partial(jit, static_argnums=0)
def append_state(states, i, state, ai, di):
    states = states.at[i, :].set(state)

def vectorize(states, size=100.0):
    states_vec = np.asarray(states).copy()
    states_vec[:, :4] /= size
    states_vec[:, 4]  /= POSS_STEER
    states_vec[:, 5]  = np.sin(states_vec[:, 4])
    states_vec[:, 6]  = np.cos(states_vec[:, 5])
    states_vec[:, 7]  /= V_MAX
    # jnp.array([jnp.tanh(di), jnp.tanh(ai)]))
    # assert np.max(np.abs(states_vec[:, :4])) < 1.5
    assert np.max(np.abs(states_vec[:, 4:])) < 1.00001
    return states_vec

def proportional_control(target, current):
    a = Kp * (target - current)
    return a

def search_target_index(state, cx, cy, old_i):
    if old_i is None:
        # search nearest point index
        d = jnp.hypot(state[2] - cx, state[3] - cy)
        ind = jnp.argmin(d)
        old_i = ind
    else:
        ind = old_i
        distance_this_index = calc_distance(state, cx[ind], cy[ind])
        while True:
            distance_next_index = calc_distance(state, cx[ind + 1], cy[ind + 1])
            if distance_this_index < distance_next_index:
                break
            ind = ind + 1 if (ind + 1) < len(cx) else ind
            distance_this_index = distance_next_index
        old_i = ind

    Lf = k * state[7] + Lfc  # update look ahead distance

    # search look ahead target point index
    while Lf > calc_distance(state, cx[ind], cy[ind]):
        if (ind + 1) >= len(cx):
            break  # not exceed goal
        ind += 1

    return ind, Lf, old_i


def pure_pursuit_steer_control(state, cx, cy, old_i, pind):
    ind, Lf, old_i = search_target_index(state, cx, cy, old_i)

    if pind >= ind:
        ind = pind

    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:  # toward goal
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    alpha = jnp.arctan2(ty - state[3], tx - state[2]) - state[4]
    delta = jnp.arctan2(2.0 * b * math.sin(alpha) / Lf, 1.0)

    return delta, ind, old_i


from numpy.random import default_rng

def pure_pursuit(cx, cy, target_speed=10.0/3.6, x0=0, y0=--3.0, yaw0=0.0, v0=0.0,
        t_max=128.0, show_animation=False, size=100.0, dt=0.5,
        state_noise=None,
        act_noise=0.,
        seed=2022):

    # initial state
    state = initial_state(x=x0, y=y0, yaw=yaw0, v=v0)
    rng = default_rng(seed)
    if state_noise is None:
        state_noise = rng.uniform(low=0, high=2e-1) # Either no noise or max 2e-1

    lastIndex = len(cx) - 1
    time = 0.0
    i = 0
    i_max = int(t_max / dt)
    states = jnp.zeros((i_max, 10))
    append_state(states, i, state, 0., 0.)
    target_ind, _, old_i = search_target_index(state, cx, cy, None)

    while t_max >= time and lastIndex > target_ind:
        apply_noise(state, rng, state_noise)
        # Calc control input
        ai = proportional_control(target_speed, state[7])
        ai += rng.normal(loc=0.0, scale=act_noise)
        try:
            di, target_ind, old_i = pure_pursuit_steer_control(
                state, cx, cy, old_i, target_ind)
            # di += rng.normal(loc=0.0, scale=act_noise)
        except IndexError: # If spline path is less than max time
            break

        update(state, ai, di, dt)  # Control vehicle

        time += dt
        i += 1
        append_state(states, i, state, 0., 0.)

    print('Total time: ')
    print(time)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    return vectorize(states)

if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    #  target course
    cx = np.arange(0, 50, 0.5)
    cy = np.array([math.sin(ix / 5.0) * ix / 2.0 for ix in cx])

    target_speed = 10.0 / 3.6  # [m/s]

    traj = pure_pursuit(cx, cy, target_speed, show_animation=True, dt=0.5)
    print(traj.shape)
    print(traj)
    1/0
