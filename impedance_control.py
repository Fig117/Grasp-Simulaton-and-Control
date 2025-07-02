# impedance_control.py

import mujoco
import mujoco.viewer
import numpy as np

# --- Load model ---
model = mujoco.MjModel.from_xml_path("/home/guan/Desktop/mujoco_menagerie-main/wonik_allegro/scene_left.xml")
data = mujoco.MjData(model)

# --- Impedance control parameters ---
Kp = np.array([50.0] * 16)   # TODO：可调节刚度
Kd = np.array([2.0] * 16)    # TODO：可调节阻尼

# --- Joint mapping ---
joint_names = ["ffj0", "ffj1", "ffj2", "ffj3", 
               "mfj0", "mfj1", "mfj2", "mfj3",
               "rfj0", "rfj1", "rfj2", "rfj3",
               "thj0", "thj1", "thj2", "thj3"] 
joint_ids = [model.joint(name).qposadr for name in joint_names]

# --- Status record variable ---
object_pos_prev = data.body("object").xpos.copy()
stability_counter = 0

# --- Calculate the target joint angle ---
def compute_q_desired_from_end_effector_targets(model, data, joint_ids):
    q_desired = np.zeros(16)
    finger_tips = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
    tip_targets = {
        "ff_tip": np.array([0.02, -0.05, 0.1]),  # TODO：替换为小球表面对应接触点
        "mf_tip": np.array([0.02,  0.00, 0.1]),
        "rf_tip": np.array([0.02,  0.05, 0.1]),
        "th_tip": np.array([-0.02, 0.00, 0.1])
    }

    for i, tip in enumerate(finger_tips):
        try:
            current_pos = data.body(tip).xpos.copy()
            delta_x = tip_targets[tip] - current_pos
            J_pos = np.zeros((3, model.nv))
            mujoco.mj_jacBodyCom(model, data, J_pos, None, model.body(tip).id)
            J_finger = J_pos[:, joint_ids[i*4:(i+1)*4]]
            dq = J_finger.T @ delta_x
            q_desired[i*4:(i+1)*4] = data.qpos[joint_ids[i*4:(i+1)*4]] + dq
        except Exception as e:
            print(f"Error computing q_desired for {tip}: {e}")

    return q_desired

# --- Launch viewer ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(1000):
        mujoco.mj_step(model, data)

        # Compute updated q_desired each step
        q_desired = compute_q_desired_from_end_effector_targets(model, data, joint_ids)

        q = data.qpos[joint_ids]
        qd = data.qvel[joint_ids]
        torque = Kp * (q_desired - q) - Kd * qd
        data.ctrl[:16] = torque

        object_pos = data.body("object").xpos.copy()
        object_vel = data.body("object").xvelp.copy()
        print(f"[{step}] Object position: {object_pos}, velocity norm: {np.linalg.norm(object_vel):.4f}")

        contact_with_object = False
        for i in range(data.ncon):
            c = data.contact[i]
            body1 = model.geom_bodyid[c.geom1]
            body2 = model.geom_bodyid[c.geom2]
            name1 = model.body(body1).name
            name2 = model.body(body2).name
            if "object" in [name1, name2]:
                contact_with_object = True
                break

        if contact_with_object:
            print("✅ Contact detected with object.")

        pos_diff = np.linalg.norm(object_pos - object_pos_prev)
        object_pos_prev = object_pos.copy()

        if pos_diff < 1e-4 and contact_with_object:
            stability_counter += 1
        else:
            stability_counter = 0

        if stability_counter > 10:
            print("✅ Object is stably grasped.")
            # TODO：此处可记录状态或保存 keyframe
            break

        viewer.sync()
