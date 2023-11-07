from typing import Literal, Any
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from vehicle_control.typing import VehicleParameters, ControllerParameters


def plot_trajectory(
    state: NDArray,
    tracked_traj: NDArray,
    veh_params: VehicleParameters,
    control_params: ControllerParameters,
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

    fig, ax = plt.subplots(figsize=(8, 8))
    rear_x: float = state[0]
    rear_y: float = state[1]
    yaw: float = state[2]
    delta: float = state[3]
    L: float = veh_params["wheelbase"]
    front_x: float = rear_x + L
    front_y: float = rear_y
    tracked = ax.plot(tracked_traj[:, 0], tracked_traj[:, 1], "r--", label="tracked")
