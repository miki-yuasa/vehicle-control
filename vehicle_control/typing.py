from typing import NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray


class VehicleParameters(NamedTuple):
    tau: float  # steering delay dynamics: 1d-approximated time constants
    wheelbase: float
    steer_limit: float
    max_vel: float
    min_vel: float
    steering_steady_state_error: float


class ReferenceDict(TypedDict):
    px: NDArray
    py: NDArray
    yaw: NDArray
    v: NDArray
    curvature: NDArray
    t: NDArray
