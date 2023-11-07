from typing import NamedTuple

import numpy as np


class VehicleParameters(NamedTuple):
    tau: float  # steering delay dynamics: 1d-approximated time constants
    wheelbase: float
    steer_limit: float
    max_vel: float
    min_vel: float
    steering_steady_state_error: float
