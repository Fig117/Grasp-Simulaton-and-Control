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
joint_qpos_ids = [model.joint(name).qposadr for name in joint_names]
joint_dof_ids = [model.joint(name).dofadr for name in joint_names]

# --- Status record variable ---
object_pos_prev = data.body("object").xpos.copy()
stability_counter = 0

# --- Calculate the target joint angle ---
# --- Calculate the target joint angle ---
def compute_q_desired_from_end_effector_targets(model, data, joint_qpos_ids, joint_dof_ids):
    q_desired = np.zeros(16)
    finger_tips = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
    tip_targets = {
        "ff_tip": np.array([0.02, -0.05, 0.1]),
        "mf_tip": np.array([0.02,  0.00, 0.1]),
        "rf_tip": np.array([0.02,  0.05, 0.1]),
        "th_tip": np.array([-0.02, 0.00, 0.1])
    }

    for i, tip in enumerate(finger_tips):
        try:
            current_pos = data.body(tip).xpos.copy()
            delta_x = tip_targets[tip] - current_pos

            # Compute full Jacobian for center of mass of the body
            J_pos = np.zeros((3, model.nv))
            mujoco.mj_jacBodyCom(model, data, J_pos, None, model.body(tip).id)

            # Extract only the relevant DOFs for this finger (4 DOFs per finger)
            dof_ids = joint_dof_ids[i*4:(i+1)*4]
            J_finger = J_pos[:, dof_ids]  # shape (3, 4)

            # Compute joint delta via Jacobian transpose
            dq = (J_finger.T @ delta_x).flatten()  # shape (4,)

            for j in range(4):
                q_id = joint_qpos_ids[i*4 + j]
                q_desired[i*4 + j] = data.qpos[q_id] + dq[j]

        except Exception as e:
            print(f"Error computing q_desired for {tip}: {e}")

    return q_desired

# --- Launch viewer ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(1000):
        mujoco.mj_step(model, data)

        # Compute updated q_desired each step
        q_desired = compute_q_desired_from_end_effector_targets(model, data, joint_qpos_ids, joint_dof_ids)
        q = np.zeros(16)
        qd = np.zeros(16)
        for i in range(16):
            q[i] = data.qpos[joint_qpos_ids[i]]
            qd[i] = data.qvel[joint_qpos_ids[i]]
        error = q_desired - q
        d_error = -qd
        torque = Kp * error + Kd * d_error
        data.ctrl[:16] = torque

        object_pos = data.body("object").xpos.copy()
        object_vel = data.cvel[model.body("object").id][:3]

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
