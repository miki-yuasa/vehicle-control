from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from vehicle_control.typing import (
    ControllerParameters,
    ReferenceDict,
    VehicleParameters,
)


def simulate_rk4(
    model: Callable[[NDArray, NDArray, VehicleParameters], NDArray],
    controller: Callable[
        [NDArray, float, ReferenceDict, VehicleParameters, ControllerParameters],
        tuple[NDArray, dict[str, Any]],
    ],
    x0: list[float],
    ref_dict: ReferenceDict,
    ts: float,
    dt: float,
    tf: int,
    veh_params: VehicleParameters,
    controller_params: ControllerParameters,
):
    x: NDArray = np.array(x0)

    t_vec: NDArray = np.arange(ts, tf, dt)

    state_log: NDArray = np.zeros([t_vec.shape[0], x.shape[0]])

    u_tmp, u_tmp_debug = controller(x, ts, ref_dict, veh_params, controller_params)
    input_log: NDArray = np.zeros([t_vec.shape[0], u_tmp.shape[0]])
    debug_info: list[dict[str, Any]] = []

    input_delay: float = controller_params["input_delay"]
    delay_count: int = np.round(input_delay / dt)
    input_buffer: NDArray = np.zeros([delay_count, u_tmp.shape[0]])
    u = np.zeros(u_tmp.shape)  # initial input
    u_debug = u_tmp_debug.copy()

    control_dt: float = controller_params["control_dt"]
    control_count: int = np.round(control_dt / dt)

    for i, t in enumerate(t_vec):
        if i % control_count == 0:
            # add noise to measurement
            x_noised = x + np.random.normal(
                0, controller_params["measurement_noise_std"]
            )
            u, u_debug = controller(
                x_noised, t, ref_dict, veh_params, controller_params
            )

        # add input delay
        input_buffer = np.vstack([u, input_buffer[:-1, :]])
        u_delayed = input_buffer[-1, :]

        k1 = model(x, u_delayed, veh_params)
        k2 = model(x + dt / 2 * k1, u_delayed, veh_params)
        k3 = model(x + dt / 2 * k2, u_delayed, veh_params)
        k4 = model(x + dt * k3, u_delayed, veh_params)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        state_log[i, :] = x
        input_log[i, :] = u
        debug_info.append(u_debug)

        print(f"Simulation progress: {i/t_vec.shape[0]*100:.2f}%", end="\r")

    return state_log, input_log, debug_info
