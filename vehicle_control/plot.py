import time
from typing import Literal, Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
from numpy.typing import NDArray

from vehicle_control.typing import (
    VehicleParameters,
    ControllerParameters,
    ReferenceDict,
)


def plot_trajectory(
    state: NDArray,
    tracked_traj: NDArray,
    time_count: int,
    veh_params: VehicleParameters,
    control_params: ControllerParameters,
    ref_dict: ReferenceDict,
    debug: dict[str, Any],
    display_mode: Literal["human", "rgb"],
) -> NDArray:
    """
    Plot the trajectory of the vehicle.

    Parameters
    ----------
    state : NDArray
        State of the vehicle. [x, y, yaw, delta]
    time_count : int
        Time count.
    veh_params : VehicleParameters
        Vehicle parameters.
    control_params : ControllerParameters
        Controller parameters.
    display_mode : Literal['human', 'rgb']
        Display mode.

    Returns
    -------
    img: NDArray
        Image of the trajectory.
    """
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(8, 8))
    rear_x: float = state[0]
    rear_y: float = state[1]
    yaw: float = state[2]
    delta: float = state[3]
    L: float = veh_params["wheelbase"]
    rear_length = 1
    front_length = 1
    side_width = 0.9
    front_x: float = rear_x + L
    front_y: float = rear_y

    tracked = ax.plot(tracked_traj[:, 0], tracked_traj[:, 1], "r--", label="tracked")
    ax.plot(ref_dict["px"], ref_dict["py"], "g--", label="reference path")
    # rear_tire = ax.plot(rear_x, rear_y, "ro", label="rear wheel")
    # front_tire = ax.plot(front_x, front_y, "bo", label="front wheel")
    # body = ax.plot(
    #     [
    #         rear_x - rear_length,
    #         front_x + front_length,
    #         front_x + front_length,
    #         rear_x - rear_length,
    #         rear_x - rear_length,
    #     ],
    #     [
    #         rear_y - side_width,
    #         front_y - side_width,
    #         front_y + side_width,
    #         rear_y + side_width,
    #         rear_y - side_width,
    #     ],
    #     "k",
    #     label="car body",
    # )

    rear_origin: NDArray = np.array([rear_x, rear_y, 0])
    front_origin: NDArray = np.array(
        [rear_x + L * np.cos(yaw), rear_y + L * np.sin(yaw), 0]
    )

    # Rotate rear and front tires and body
    z_axis: NDArray = np.array([0, 0, 1])
    rear_rot = transforms.Affine2D().rotate_around(rear_origin[0], rear_origin[1], yaw)
    front_rot = transforms.Affine2D().rotate_around(
        front_origin[0], front_origin[1], delta
    )

    # Plot the rotated rear and front tires and body
    rear_tire = ax.plot(rear_x, rear_y, "ro", label="rear wheel", transform=rear_rot)
    front_tire = ax.plot(
        front_x, front_y, "bo", label="front wheel", transform=front_rot
    )
    body = ax.plot(
        [
            rear_x - rear_length,
            front_x + front_length,
            front_x + front_length,
            rear_x - rear_length,
            rear_x - rear_length,
        ],
        [
            rear_y - side_width,
            front_y - side_width,
            front_y + side_width,
            rear_y + side_width,
            rear_y - side_width,
        ],
        "k",
        label="car body",
        transform=rear_rot,
    )

    # Plot title
    ax.set_title(f"Time: {time_count * control_params['control_dt']} s")

    # Plot legend
    ax.legend()

    ax.set_xlim(-10, 120)
    ax.set_ylim(-10, 120)

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)  # 0.1秒だけ開ける
    plt.show()
