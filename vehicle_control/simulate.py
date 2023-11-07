from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from vehicle_control.typing import ReferenceDict, VehicleParameters


def simulate_rk4(
    model: Callable[[NDArray, NDArray, VehicleParameters], NDArray],
    controller: Callable[
        [NDArray, float, ReferenceDict, dict[str, Any]],
        tuple[NDArray, dict[str, Any]],
    ],
    x0: list[float],
    ref_dict: ReferenceDict,
    ts: float,
    dt: float,
    tf: int,
    veh_param: VehicleParameters,
):
    x: NDArray = np.array(x0)

    t_vec: NDArray = np.arange(ts, tf, dt)

    state_log: NDArray = np.zeros([t_vec.shape[0], x.shape[0]])
