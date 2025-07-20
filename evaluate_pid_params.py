# evaluate_pid_params.py

import mujoco
import numpy as np
from mujoco_controller import compute_control_signals, create_env

def evaluate_pid_params(params):
    """
    Evaluate a given PID parameter set [kp, ki, kd]
    by simulating the palm movement task and returning total position error.
    """
    kp, ki, kd = params
    model, data = create_env()
    mujoco.mj_resetData(model, data)

    # Initialize variables
    total_error = 0.0
    max_steps = 300
    error_threshold = 0.005

    # Set initial and target position
    palm_id = model.body('palm').id
    target_pos = np.asarray([0.0, 0.0, -0.045])  
    prev_error = np.zeros(3)
    integral_error = np.zeros(3)

    for _ in range(max_steps):
        mujoco.mj_step(model, data)
        current_pos = data.body(palm_id).xpos
        error = target_pos - current_pos
        control = compute_control_signals(kp, ki, kd, error, prev_error, integral_error)
        data.ctrl[0:3] = control

        total_error += np.linalg.norm(error)

        if np.linalg.norm(error) < error_threshold:
            break

        prev_error[:] = error

    return total_error
