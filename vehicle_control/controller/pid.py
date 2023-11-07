from typing import Any

import numpy as np
from numpy.typing import NDArray

from vehicle_control.typing import (
    ControllerParameters,
    ReferenceDict,
    VehicleParameters,
)


def pid_controller(
    state: NDArray,
    t: float,
    ref: ReferenceDict,
    veh_param: VehicleParameters,
    controller_params: ControllerParameters,
) -> tuple[NDArray, dict[str, Any]]:
    """
    PID controller for lateral and longitudinal control.

    Parameters
    ----------
    state : NDArray
        Current state of the vehicle. [x, y, yaw, delta]
    t : float
        Current time.
    ref : ReferenceDict
        Reference trajectory. [x, y, yaw, v, curvature, t]
    params : dict[str, Any]
        Parameters for the controller.
    """

    yaw: float = state[2]

    distance: NDArray = np.sqrt(
        (ref["px"] - state[0]) ** 2 + (ref["py"] - state[1]) ** 2
    )
    min_index: int = np.argmin(distance)

    v_des: float = ref["v"][min_index]

    # Feedforward input calculation
    ff_curvature = np.arctan(veh_param.wheelbase * ref["curvature"][min_index])

    # Coordinate transformation to the body frame
    T = np.array(
        [
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)],
        ]
    )
    error_xy = np.array(
        [ref["px"][min_index] - state[0], ref["py"][min_index] - state[1]]
    ).T
    error_lat_long = T @ error_xy
    error_yaw = (yaw - ref["yaw"][min_index]) % (2 * np.pi)

    error_yaw = error_yaw if error_yaw < np.pi else error_yaw - 2 * np.pi

    LON: int = 0
    LAT: int = 1

    # PID control
    delta_des = -kp * error_lat_long[LAT] - kd * error_yaw + ff_curvature
    fb_lat = -kp * error_lat_long[LAT]
    fb_yaw = -kd * error_yaw

    u = np.array([v_des, delta_des])

    debug_info = {
        "error_lat": error_lat_long[LAT],
        "error_yaw": error_yaw,
        "ff_curvature": ff_curvature,
        "fb_lat": fb_lat,
        "fb_yaw": fb_yaw,
        "min_index": min_index,
        "min_x": ref["px"][min_index],
        "min_y": ref["py"][min_index],
        "min_yaw": ref["yaw"][min_index],
    }

    return u, debug_info
