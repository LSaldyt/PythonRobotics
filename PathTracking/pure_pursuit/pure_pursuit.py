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
from rich import print
from rich.pretty import pprint

# from jax.config import config
# config.update('jax_disable_jit', True)

class FinishError(Exception):
    pass

# Parameters
k = 0.1   # look forward gain
Lfc = 2.0 # [m] look-ahead distance
Kp  = 1.0  # speed proportional gain
b   = 1.    # [m] wheel base of vehicle
m   = 20.

BASE_MAX   = 100.
MASS_MAX   = 100.
STEER_MAX  = 1. # one radian max steering np.deg2rad(57.5) # Yeah why not
POSS_STEER = np.pi
V_MAX      = 2.
FORCE_MAX  = 1.
PARAMS = np.array([b/BASE_MAX, m/MASS_MAX, STEER_MAX/POSS_STEER, FORCE_MAX])

@jit
def dynamics(t, state, action):
    # Basically we assume tanh & scaling is applied to inputs by a nn fairie
    w, f = jnp.split(action, 2, axis=-1)
    x, y, rear_x, rear_y, yaw, sin_yaw, cos_yaw, v, go = jnp.split(state, 9, axis=-1)
    # Update dynamics
    x   = x + v * jnp.cos(yaw) * t
    y   = y + v * jnp.sin(yaw) * t
    yaw += v / b * jnp.tan(w) * t
    v   = v + (f/m) * t
    v   = jnp.minimum(v, V_MAX)
    # Repack vectors
    rear_x = x - ((b / 2) * jnp.cos(yaw))
    rear_y = y - ((b / 2) * jnp.sin(yaw))
    next_state = jnp.concatenate((x, y, rear_x, rear_y, yaw,
                                  jnp.sin(yaw), jnp.cos(yaw), v))
    return (go * next_state) + ((1 - go) * state[:-1]) # Use go to gate :)

@jit
def initial_state(x=0.0, y=0.0, yaw=0.0, v=0.0):
    rear_x = x - ((b / 2) * jnp.cos(yaw))
    rear_y = y - ((b / 2) * jnp.sin(yaw))
    return jnp.array([x, y, rear_x, rear_y, yaw, jnp.sin(yaw), jnp.cos(yaw), v, 0., 0.])

@jit
def apply_noise(state, key, state_noise):
    _, key = jr.split(key)
    return key, state.at[:5].set(state[:5] + jr.normal(key, (5,)) * state_noise)

@jit
def update(state, f, delta, dt):
    state = state.at[-2:].set(jnp.array([delta, f]))
    x_new = dynamics(dt, jnp.concatenate((state[:8], jnp.array([1]))), state[-2:])
    state = state.at[:8].set(x_new)
    return state

@jit
def calc_distance(state, point_x, point_y):
    dx = state[2] - point_x
    dy = state[3] - point_y
    return jnp.hypot(dx, dy)

@jit
def append_state(states, i, state, ai, di):
    state = state.at[-2:].set(jnp.array([di, ai]))
    states = states.at[i, :].set(state)
    return states

@jit
def proportional_control(target, current):
    a = Kp * (target - current)
    return a

@jit
def search_target_index(state, cx, cy, old_i):
    l = len(cx)
    dists = calc_distance(state, cx, cy)
    ind   = jnp.argmin(dists)
    old_i = ind

    Lf = k * state[7] + Lfc  # update look ahead distance

    # Elegance.
    cond = jnp.where(Lf > dists, 1, 0)
    alt  = cond + jnp.where(ind <= jnp.arange(l), cond, 1)
    ind  = jnp.argmin(alt)
    return ind, Lf, old_i

@jit
def pure_pursuit_steer_control(state, cx, cy, old_i, pind):
    ind, Lf, old_i = search_target_index(state, cx, cy, old_i)
    ind = jnp.minimum(jnp.maximum(pind, ind), len(cx) - 1)

    alpha = jnp.arctan2(cy[ind] - state[3], cx[ind] - state[2]) - state[4]
    delta = jnp.arctan2(2.0 * b * jnp.sin(alpha) / Lf, 1.0)
    return delta, ind, old_i

import jax.random as jr

# @partial(jit, static_argnames=('dt', 'seed',))
def pure_pursuit(cx, cy, x0=0, y0=0.0, yaw0=0.0, v0=0.0,
        t_max=128.0, size=100.0, dt=0.5, seed=2022,
        state_noise=None):
    key = jr.PRNGKey(seed)
    _, key = jr.split(key)
    yaw0 = jr.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
    _, key = jr.split(key)
    v0 = jr.uniform(key, minval=0, maxval=0.5)
    state = initial_state(x=x0, y=y0, yaw=yaw0, v=v0)
    _, key = jr.split(key)
    state_noise = 5e-2

    lastIndex = len(cx) - 1
    time = 0.0; i = 0
    i_max = int(t_max / dt)
    states = jnp.zeros((i_max, 10))
    states = append_state(states, i, state, 0., 0.)
    i += 1
    target_ind, _, old_i = search_target_index(state, cx, cy, 0)

    while t_max >= time and lastIndex > target_ind:
        key, state = apply_noise(state, key, state_noise)
        # Calc control input
        ai = proportional_control(V_MAX, state[7])
        di, target_ind, old_i = pure_pursuit_steer_control(
            state, cx, cy, old_i, target_ind)
        state = update(state, ai, di, dt)
        states = append_state(states, i, state, ai, di)
        time += dt; i += 1
        # print(f'Time: {time} / {t_max} ({i})')
    # print('Finished pure pursuit!')
    # print(f'Time: {time} / {t_max}')
    # print('indices', lastIndex, target_ind, len(cx))
    # Time: 142.0 / 1024.0
    # indices 383 383
    # (105, 10)

    return vectorize(states)

def vectorize(states, size=100.0):
    states_vec = np.asarray(states).copy()
    states_vec[:, :4] /= size
    # states_vec[:, 4]  = 0. # Don't report yaw :)
    states_vec[:, 7]  /= V_MAX
    states_vec[:, 8]  /= STEER_MAX
    states_vec[:, 9]  /= V_MAX
    # assert np.max(np.abs(states_vec[:, :4])) < 1.5
    if np.max(np.abs(states_vec[:, 5:])) > 1.00001:
        print(np.max(np.abs(states_vec), axis=0))
        raise ValueError('Pure pursuit must return a normalized array')
    mask   = np.minimum(np.sum(states_vec, axis=1), 1e-5)
    finish = np.argmin(mask, axis=0) + 1
    states_vec = states_vec[:finish, :]
    return states_vec


import plotly.express as px
import plotly.graph_objects as go

if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    #  target course
    cx = jnp.arange(0, 50, 0.5)
    cy = jnp.sin(cx / 5.0) * cx / 2.0

    traj = pure_pursuit(cx, cy, dt=1.0, t_max=256)
    # print(traj.shape)
    # print(traj)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cx/100., y=cy/100., name='spline'))
    fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], name='pure pursuit'))
    fig.show()
