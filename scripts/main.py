from typing import Literal

import numpy as np
from numpy.typing import NDArray

control_mode: Literal["pid", "mpc"] = "pid"

is_video_saved: bool = False

simulation_time: int = 35
simulation_rk4_time_step: float = 0.002

kmh2ms: float = 1 / 3.6
vel_ref: float = 30 * kmh2ms

# Vehicle parameters
tau: float = 0.27  # steering delay dynamics: 1d-approximated time constants
wheelbase: float = 2.7
steer_limit: float = np.deg2rad(30)
max_vel: float = 10
min_vel: float = -5

input_delay: float = 0.24  # [s]
control_dt: float = 0.03  # [s]
measurement_noise_std = [0.1, 0.1, np.deg2rad(1.0), np.deg2rad(0.5)]
steering_steady_state_error = np.deg2rad(1.0)

# Initial position (x, y, yaw, delta)
x0: list[float] = [0.0, 0.5, 0.0, 0.0]

ts = 0.0
dt = simulation_rk4_time_step
tf = simulation_time
t = np.arange(ts, tf, dt)

# Reference path
path_filename: str = "assets/path.npz"
path_data = np.load(path_filename)

px_path: NDArray = path_data["px_path"]
py_path: NDArray = path_data["py_path"]
yaws: NDArray = path_data["yaws"]

reference: NDArray = np.zeros([px_path.shape[0], 6])
reference[:, 0] = px_path
reference[:, 1] = py_path
reference[:, 2] = yaws
reference[:, 3] = vel_ref

# Insert curvature to reference
path = np.array([px_path, py_path]).T
p1 = path[:, :-2]
p2 = path[:, 1:-1]
p3 = path[:, 2:]
A = (
    (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
    - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
) / 2
reference[1:-1, 4] = (
    4
    * A
    / np.linalg.norm(p2 - p1, axis=0)
    / np.linalg.norm(p3 - p2, axis=0)
    / np.linalg.norm(p3 - p1, axis=0)
)

# Insert velocity reference
px_diff = np.diff(px_path)
py_diff = np.diff(py_path)
distances = np.sqrt(px_diff**2 + py_diff**2)
dt_ref = distances / vel_ref
t_ref = np.cumsum(dt_ref)
reference[:-1, 5] = np.diff(t_ref)
reference[-1, 5] = reference[-2, 5] + dt_ref[-1]
