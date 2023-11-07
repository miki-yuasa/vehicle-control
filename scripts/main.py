from typing import Literal

import numpy as np
from numpy.typing import NDArray
from vehicle_control.kinematics import kinematic_model
from vehicle_control.simulate import simulate_rk4

from vehicle_control.typing import (
    ControllerParameters,
    ReferenceDict,
    VehicleParameters,
)
from vehicle_control.controller.pid import pid_controller

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

# PID parameters
kp: float = 0.3
kd: float = 1.5

# Initial position (x, y, yaw, delta)
x0: list[float] = [0.0, 0.5, 0.0, 0.0]

ts: float = 0.0
dt: float = simulation_rk4_time_step
tf: int = simulation_time
t: NDArray = np.arange(ts, tf, dt)

# Reference path
path_size_scale: float = 15.0
path_filename: str = "assets/path.npz"
path_data = np.load(path_filename)

px_path: NDArray = path_data["px_path"] * path_size_scale
py_path: NDArray = path_data["py_path"] * path_size_scale
yaws: NDArray = path_data["yaws"]

reference: NDArray = np.zeros([px_path.shape[0], 6])
reference[:, 0] = px_path
reference[:, 1] = py_path
reference[:, 2] = yaws
reference[:, 3] = vel_ref

# Insert curvature to reference
path = np.array([px_path, py_path]).T
p1 = path[:-2]
p2 = path[1:-1]
p3 = path[2:]
A: NDArray = (
    (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
    - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
) / 2
reference[1:-1, 4] = (
    4
    * A
    / np.linalg.norm(p2 - p1, axis=1)
    / np.linalg.norm(p3 - p2, axis=1)
    / np.linalg.norm(p3 - p1, axis=1)
)

# Insert time reference
px_diff: NDArray = np.diff(px_path)
py_diff: NDArray = np.diff(py_path)
distances: NDArray = np.sqrt(px_diff**2 + py_diff**2)
dt_ref: NDArray = distances / vel_ref
t_ref: NDArray = np.cumsum(dt_ref)
reference[:-1, 5] = t_ref
reference[-1, 5] = reference[-2, 5] + dt_ref[-1]

ref_dict: ReferenceDict = {
    "px": reference[:, 0],
    "py": reference[:, 1],
    "yaw": reference[:, 2],
    "v": reference[:, 3],
    "curvature": reference[:, 4],
    "t": reference[:, 5],
}

controller_params: ControllerParameters = {
    "input_delay": input_delay,
    "control_dt": control_dt,
    "measurement_noise_std": measurement_noise_std,
    "kp": kp,
    "kd": kd,
}

veh_params: VehicleParameters = {
    "tau": tau,
    "wheelbase": wheelbase,
    "steer_limit": steer_limit,
    "max_vel": max_vel,
    "min_vel": min_vel,
    "steering_steady_state_error": steering_steady_state_error,
}

match control_mode:
    case "pid":
        X, U, debug = simulate_rk4(
            kinematic_model,
            pid_controller,
            x0,
            ref_dict,
            ts,
            dt,
            tf,
            veh_params,
            controller_params,
        )
    case _:
        raise NotImplementedError(f"Control mode {control_mode} is not implemented.")

print("Simulation finished")
