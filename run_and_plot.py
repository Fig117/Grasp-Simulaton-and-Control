# run_and_plot.py

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mujoco_controller import compute_control_signals, create_env

def run_simulation(kp, ki, kd, plot=True):
    model, data = create_env()
    mujoco.mj_resetData(model, data)

    palm_id = model.body('palm').id
    target_pos = np.asarray([0.0, 0.0, -0.045]) 
    max_steps = 1000

    prev_error = np.zeros(3)
    integral_error = np.zeros(3)
    error_norms = []

    for _ in range(max_steps):
        current_pos = data.body(palm_id).xpos
        error = target_pos - current_pos
        control = compute_control_signals(kp, ki, kd, error, prev_error, integral_error)
        data.ctrl[0:3] = control

        mujoco.mj_step(model, data)
        error_norms.append(np.linalg.norm(error))
        prev_error[:] = error

        if np.linalg.norm(error) < 0.005:
            break

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(error_norms, label=f'Kp={kp}, Ki={ki}, Kd={kd}')
        plt.xlabel("Step")
        plt.ylabel("Position Error Norm")
        plt.title("PID Control Error over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return error_norms

if __name__ == "__main__":
    run_simulation(kp=0.1, ki=0.02, kd=0.001)
