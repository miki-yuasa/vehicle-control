import time
from typing import Literal
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from vehicle_control.kinematics import kinematic_model
from vehicle_control.simulate import simulate_rk4
from vehicle_control.typing import (
    ControllerParameters,
    ReferenceDict,
    VehicleParameters,
)
from vehicle_control.controller.pid import pid_controller
from vehicle_control.plot import plot_trajectory

control_mode: Literal["pid", "mpc"] = "pid"

is_video_saved: bool = False

simulation_time: int = 35
simulation_rk4_time_step: float = 0.01

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
kp: float = 0.1
kd: float = 3

# Initial position (x, y, yaw, delta)
x0: list[float] = [0.0, 0.0, 0.0, 0.0]

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

# Plot the trajectory

total_timesteps: int = X.shape[0]

sns.set_theme()
fig: Figure
ax: Axes
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(ref_dict["px"], ref_dict["py"], "g--", label="reference path")
ax.plot(X[:, 0], X[:, 1], "b", label="trajectory")
plt.show()

# for i in range(total_timesteps):
#     state = X[i, :]
#     tracked_traj = X[: i + 1, :2]

#     rear_x: float = state[0]
#     rear_y: float = state[1]
#     yaw: float = state[2]
#     delta: float = state[3]
#     L: float = veh_params["wheelbase"]
#     rear_length = 1
#     front_length = 1
#     side_width = 0.9
#     front_x: float = rear_x + L
#     front_y: float = rear_y

#     tracked = ax.plot(tracked_traj[:, 0], tracked_traj[:, 1], "r--", label="tracked")
#     ax.plot(ref_dict["px"], ref_dict["py"], "g--", label="reference path")

#     rear_origin: NDArray = np.array([rear_x, rear_y, 0])
#     front_origin: NDArray = np.array(
#         [rear_x + L * np.cos(yaw), rear_y + L * np.sin(yaw), 0]
#     )

#     # Rotate rear and front tires and body
#     z_axis: NDArray = np.array([0, 0, 1])
#     rear_rot = transforms.Affine2D().rotate_around(rear_origin[0], rear_origin[1], yaw)
#     front_rot = transforms.Affine2D().rotate_around(
#         front_origin[0], front_origin[1], delta
#     )

#     # Plot the rotated rear and front tires and body
#     (rear_tire,) = ax.plot(
#         rear_x,
#         rear_y,
#         "ro",
#         label="rear wheel",
#     )  # transform=rear_rot)
#     (front_tire,) = ax.plot(
#         front_x,
#         front_y,
#         "bo",
#         label="front wheel",  # transform=front_rot
#     )
#     (body,) = ax.plot(
#         [
#             rear_x - rear_length,
#             front_x + front_length,
#             front_x + front_length,
#             rear_x - rear_length,
#             rear_x - rear_length,
#         ],
#         [
#             rear_y - side_width,
#             front_y - side_width,
#             front_y + side_width,
#             rear_y + side_width,
#             rear_y - side_width,
#         ],
#         "k",
#         label="car body",
#         # transform=rear_rot,
#     )

#     # Plot title
#     ax.set_title(f"Time: {i * controller_params['control_dt']:.2f} s")

#     # Plot legend
#     # ax.legend()

#     ax.set_xlim(-10, 120)
#     ax.set_ylim(-10, 70)

#     plt.pause(0.00001)

#     rear_tire.remove()
#     front_tire.remove()
#     body.remove()
