import time
import mujoco
import mujoco.viewer
import numpy as np

# %%
xml_path = "Scene/wonik_allegro/scene_right.xml"

# Load model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.mj_kinematics(model, data)
renderer = mujoco.Renderer(model)

# PID controller parameters
kp = 0.2
ki = 0.001
kd = 0.01
integral_error = np.zeros(3)
prev_error = np.zeros(3)

error = 0.000001
hand_initial_pos = data.body('palm').xpos
target_initial_pos = data.body('cylinder_object').xpos
distance_to_target = target_initial_pos - hand_initial_pos

init_flag = False
grasp_flag = False
z_reached = False  # 标记 z=0.2 是否完成
ff_flag = False
mf_flag = False
rf_flag = False
th_flag = False

mujoco.mj_resetData(model, data)

# Jacobians
ff_tip_jacp = np.zeros((3, model.nv))
ff_tip_jacr = np.zeros((3, model.nv))
mf_tip_jacp = np.zeros((3, model.nv))
mf_tip_jacr = np.zeros((3, model.nv))
rf_tip_jacp = np.zeros((3, model.nv))
rf_tip_jacr = np.zeros((3, model.nv))
th_tip_jacp = np.zeros((3, model.nv))
th_tip_jacr = np.zeros((3, model.nv))

# Target fingertip positions
ff_tip_target = np.asarray([0.01164853, -0.03474731, 0.06874375])
mf_tip_target = np.asarray([0.01088803, -0.03492336, 0.02110088])
rf_tip_target = np.asarray([0.01188256, -0.03492185, 0])
th_tip_target = np.asarray([0.02939237, 0.09479632, 0.03315931])

# Get indexes
ff_tip_idx = model.body('ff_tip').id
mf_tip_idx = model.body('mf_tip').id
rf_tip_idx = model.body('rf_tip').id
th_tip_idx = model.body('th_tip').id
cylinder_object_idx = model.body('cylinder_object').id

def compute_control_signals(kp, ki, kd, error):
    global integral_error, prev_error
    p = kp * error
    integral_error += error
    i = ki * integral_error
    d = kd * (error - prev_error)
    prev_error = error
    return p + i + d

error_fingers = 0.025

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 1000:
        step_start = time.time()

        if np.abs(distance_to_target[0]) > error and np.abs(distance_to_target[1]) > error and np.abs(distance_to_target[2]) > error and not init_flag:
            hand_initial_pos = data.body('palm').xpos
            target_initial_pos = data.body('cylinder_object').xpos
            distance_to_target = target_initial_pos - hand_initial_pos + [-0.095, 0.01, 0]
            data.ctrl[0:3] = compute_control_signals(0.2, 0.001, 0.01, distance_to_target)

        elif not grasp_flag:
            init_flag = True
            ff_tip_current_pos = data.body('ff_tip').xpos
            mf_tip_current_pos = data.body('mf_tip').xpos
            rf_tip_current_pos = data.body('rf_tip').xpos
            th_tip_current_pos = data.body('th_tip').xpos

            ff_tip_distance_to_target = ff_tip_target - ff_tip_current_pos
            mf_tip_distance_to_target = mf_tip_target - mf_tip_current_pos
            rf_tip_distance_to_target = rf_tip_target - rf_tip_current_pos
            th_tip_distance_to_target = th_tip_target - th_tip_current_pos

            ff_flag = np.all(np.abs(ff_tip_distance_to_target) < error_fingers)
            mf_flag = np.all(np.abs(mf_tip_distance_to_target) < error_fingers)
            rf_flag = np.all(np.abs(rf_tip_distance_to_target) < error_fingers)
            th_flag = np.all(np.abs(th_tip_distance_to_target) < error_fingers + 0.01)

            if ff_flag and mf_flag and rf_flag and th_flag:
                grasp_flag = True
                print("grasped done")

            """ PID for Hard finger
            ff_tip_control_signals = compute_control_signals(3, 0.2, 0.02, ff_tip_distance_to_target)
            mf_tip_control_signals = compute_control_signals(3, 0.2, 0.01, mf_tip_distance_to_target)
            rf_tip_control_signals = compute_control_signals(3, 0.2, 0.01, rf_tip_distance_to_target)
            th_tip_control_signals = compute_control_signals(0.9, 0.03, 0.01, th_tip_distance_to_target)
            """
            """PID for Soft finger
            ff_tip_control_signals = compute_control_signals(1.5, 0.2, 0.02, ff_tip_distance_to_target)
            mf_tip_control_signals = compute_control_signals(1.5, 0.2, 0.01, mf_tip_distance_to_target)
            rf_tip_control_signals = compute_control_signals(1.4, 0.2, 0.01, rf_tip_distance_to_target)
            th_tip_control_signals = compute_control_signals(0.9, 0.03, 0.01, th_tip_distance_to_target)
            """
            # PID for Frinctionless finger
            ff_tip_control_signals = compute_control_signals(3, 0.2, 0.01, ff_tip_distance_to_target)
            mf_tip_control_signals = compute_control_signals(3, 0.2, 0.01, mf_tip_distance_to_target)
            rf_tip_control_signals = compute_control_signals(3, 0.2, 0.01, rf_tip_distance_to_target)
            th_tip_control_signals = compute_control_signals(0.9, 0.03, 0.01, th_tip_distance_to_target)
            

            mujoco.mj_jac(model, data, ff_tip_jacp, ff_tip_jacr, ff_tip_current_pos, ff_tip_idx)
            mujoco.mj_jac(model, data, mf_tip_jacp, mf_tip_jacr, mf_tip_current_pos, mf_tip_idx)
            mujoco.mj_jac(model, data, rf_tip_jacp, rf_tip_jacr, rf_tip_current_pos, rf_tip_idx)
            mujoco.mj_jac(model, data, th_tip_jacp, th_tip_jacr, th_tip_current_pos, th_tip_idx)

            ff_tip_jacp = ff_tip_jacp.reshape((3, model.nv))
            mf_tip_jacp = mf_tip_jacp.reshape((3, model.nv))
            rf_tip_jacp = rf_tip_jacp.reshape((3, model.nv))
            th_tip_jacp = th_tip_jacp.reshape((3, model.nv))

            ff_tip_joint = np.linalg.pinv(ff_tip_jacp) @ ff_tip_control_signals
            mf_tip_joint = np.linalg.pinv(mf_tip_jacp) @ mf_tip_control_signals
            rf_tip_joint = np.linalg.pinv(rf_tip_jacp) @ rf_tip_control_signals
            th_tip_joint = np.linalg.pinv(th_tip_jacp) @ th_tip_control_signals

            control_signals_joint_space = np.concatenate((
                ff_tip_joint[3:7],
                mf_tip_joint[7:11],
                rf_tip_joint[11:15],
                th_tip_joint[15:19]
            ))
            data.ctrl[3:] = control_signals_joint_space

        # 抓稳之后上升到 z = 0.2
        if grasp_flag and not z_reached:
            current_z = data.qpos[2]
            z_error = 0.2 - current_z
            if abs(z_error) > 0.001:
                z_signal = compute_control_signals(2.0, 0.0, 0.01, np.array([0, 0, z_error]))[2]
                data.ctrl[2] = z_signal
            else:
                data.ctrl[2] = 0
                z_reached = True
                print("z = 0.2 reached")

        mujoco.mj_step(model, data)
        renderer.update_scene(data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
