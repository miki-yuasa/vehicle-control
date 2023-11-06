import json
from typing import TypedDict

import scipy.interpolate as interp
import numpy as np
from numpy.typing import NDArray


def create_path(
    path_points_filename: str, path_savename: str, param_step_size: float = 0.01
):
    with open(path_points_filename) as f:
        points_dict = json.load(f)

    points: list[list[float]] = points_dict["points"]
    points_np: NDArray[np.float64] = np.array(points)

    s = np.arange(0, len(points), 1)

    px_spline = interp.CubicSpline(s, points_np[:, 0])
    py_spline = interp.CubicSpline(s, points_np[:, 1])

    s_path = np.arange(0, len(points), param_step_size)

    px_path = px_spline(s_path)
    py_path = py_spline(s_path)

    # Calculate yaws
    yaws: NDArray = np.zeros(len(px_path))

    for i in range(1, (len(px_path) - 1)):
        yaws[i] = np.arctan2(
            py_path[i + 1] - py_path[i - 1], px_path[i + 1] - px_path[i - 1]
        )

    yaws[0] = yaws[1]
    yaws[-1] = yaws[-2]

    np.savez(path_savename, px_path=px_path, py_path=py_path, yaws=yaws)

    return px_path, py_path, yaws
