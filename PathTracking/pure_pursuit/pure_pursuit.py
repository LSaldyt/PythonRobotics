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

# Parameters
k = 0.1   # look forward gain
Lfc = 2.0 # [m] look-ahead distance
Kp = 1.0  # speed proportional gain
b = 1.    # [m] wheel base of vehicle
m = 1.

BASE_MAX   = 10.
MASS_MAX   = 100.
STEER_MAX  = np.deg2rad(40.)
POSS_STEER = np.deg2rad(180.)
YAW_MAX    = 10.
V_MAX      = 3.
FORCE_MAX  = 1.0
PARAMS = np.array([b/BASE_MAX, m/MASS_MAX, STEER_MAX/POSS_STEER, FORCE_MAX])

@jit
def dynamics(t, x, u):
    w, f = jnp.split(u, 2, axis=-1)
    # Basically we assume tanh & scaling is applied to inputs by a nn fairie
    # f = jnp.tanh(f) * FORCE_MAX
    # w = jnp.tanh(w) * STEER_MAX
    # f = f * FORCE_MAX
    # w = w * STEER_MAX
    x, y, rear_x, rear_y, yaw, sin_yaw, cos_yaw, v = jnp.split(x, 8, axis=-1)
    # v   = v * V_MAX
    # Update dynamics
    x   = x + v * jnp.cos(yaw) * t
    y   = y + v * jnp.sin(yaw) * t
    yaw += v / b * jnp.tan(w) * t
    v   = v + (f/m) * t
    v   = jnp.tanh(v) * V_MAX
    # Repack vectors
    rear_x = x - ((b / 2) * jnp.cos(yaw))
    rear_y = y - ((b / 2) * jnp.sin(yaw))
    next_state = jnp.concatenate((x, y, rear_x, rear_y, yaw,
                                  jnp.sin(yaw), jnp.cos(yaw), v))
    return next_state

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((b / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((b / 2) * math.sin(self.yaw))

    def update(self, f, delta, dt):
        f     = math.tanh(f) * FORCE_MAX
        delta = math.tanh(delta) * STEER_MAX
        u = jnp.array([delta, f])
        x = jnp.array([self.x, self.y, self.rear_x, self.rear_y, self.yaw,
                       jnp.sin(self.yaw), jnp.cos(self.yaw), self.v])
        x_new = dynamics(dt, x, u)
        x, y, rear_x, rear_y, yaw, sin_yaw, cos_yaw, v = jnp.split(x_new, 8, axis=-1)
        self.x      = float(x)
        self.y      = float(y)
        self.rear_x = float(rear_x)
        self.rear_y = float(rear_y)
        self.yaw    = float(yaw)
        self.v      = float(v)
        # self.x += self.v * math.cos(self.yaw) * dt
        # self.y += self.v * math.sin(self.yaw) * dt
        # self.yaw += self.v / b * math.tan(delta) * dt
        # self.v += (f/m) * dt
        # self.v = math.tanh(self.v) * V_MAX
        # self.rear_x = self.x - ((b / 2) * math.cos(self.yaw))
        # self.rear_y = self.y - ((b / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []
        self.rear_x = []
        self.rear_y = []
        self.ai     = []
        self.di     = []

    def append(self, t, state, ai, di):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)
        self.rear_x.append(state.rear_x)
        self.rear_y.append(state.rear_y)
        self.ai.append(ai)
        self.di.append(di)

    def vectorize(self, size=100.0):
        x      = np.array(self.x)
        y      = np.array(self.y)
        yaw    = np.array(self.yaw)
        v      = np.array(self.v)
        rear_x = np.array(self.rear_x)
        rear_y = np.array(self.rear_y)
        di     = np.array(self.di)
        ai     = np.array(self.ai)

        return np.concatenate((np.expand_dims(x,   -1) / size,
                               np.expand_dims(y,   -1) / size,
                               np.expand_dims(rear_x,   -1) / size,
                               np.expand_dims(rear_y,   -1) / size,
                               np.expand_dims(yaw, -1) / YAW_MAX,
                               np.expand_dims(np.sin(yaw), -1),
                               np.expand_dims(np.cos(yaw), -1),
                               np.expand_dims(v,   -1) / V_MAX,
                               np.expand_dims(di, -1) / STEER_MAX,
                               np.expand_dims(ai, -1) / FORCE_MAX,
                               ), axis=1)


def proportional_control(target, current):
    a = Kp * (target - current)
    return a

class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
    delta = math.atan2(2.0 * b * math.sin(alpha) / Lf, 1.0)

    return delta, ind


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def pure_pursuit(cx, cy, target_speed=10.0/3.6, x0=0, y0=--3.0, yaw0=0.0, v0=0.0, t_max=128.0, show_animation=False, size=100.0, dt=0.25):
    # initial state
    state = State(x=x0, y=y0, yaw=yaw0, v=v0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state, 0., 0.)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while t_max >= time and lastIndex > target_ind:

        # Calc control input
        ai = proportional_control(target_speed, state.v)
        try:
            di, target_ind = pure_pursuit_steer_control(
                state, target_course, target_ind)
        except IndexError:
            break

        state.update(ai, di, dt)  # Control vehicle

        time += dt
        states.append(time, state, ai, di)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()

    return states.vectorize(size)

if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    #  target course
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    target_speed = 10.0 / 3.6  # [m/s]

    traj = pure_pursuit(cx, cy, target_speed, show_animation=True, dt=0.5)
    print(traj.shape)
    print(traj)
    1/0
