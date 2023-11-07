import numpy as np

from vehicle_control.typing import VehicleParameters


def kinematic_model(state, inputs, veh_param: VehicleParameters):
    v_des: float = inputs[0]
    delta_des: float = inputs[1]

    # limit
    delta_des = np.clip(delta_des, -veh_param.steer_limit, veh_param.steer_limit)
    v_des = np.clip(v_des, veh_param.min_vel, veh_param.max_vel)

    # state
    yaw: float = state[2]
    delta: float = state[3]

    v: float = v_des

    # update
    d_x: float = v * np.cos(yaw)
    d_y: float = v * np.sin(yaw)
    d_yaw: float = v / veh_param.wheelbase * np.tan(delta)
    d_delta: float = 1 / veh_param.tau * (delta_des - delta)

    # add steady state error caused by friction
    if np.abs(delta - delta_des) < veh_param.steering_steady_state_error:
        d_delta = 0
    else:
        pass

    d_state = np.array([d_x, d_y, d_yaw, d_delta])

    return d_state
