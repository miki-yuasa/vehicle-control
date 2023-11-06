from typing import Literal

import numpy as np

control_mode: Literal["pid", "mpc"] = "pid"

is_video_saved: bool = False

simulation_time: int = 35
simulation_rk4_time_step: float = 0.002

kmh2ms: float = 1 / 3.6
vel_ref: float = 30 * kmh2ms

# Vehicle parameters
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
