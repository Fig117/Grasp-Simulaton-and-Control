# mujoco_controller.py

import mujoco
import numpy as np

xml_path = "Scene/wonik_allegro/Scene_right_changed.xml"

def create_env():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data

def compute_control_signals(kp, ki, kd, error, prev_error, integral_error):
    """
    Basic PID control computation
    """
    p = kp * error
    integral_error += error
    i = ki * integral_error
    d = kd * (error - prev_error)
    return p + i + d
