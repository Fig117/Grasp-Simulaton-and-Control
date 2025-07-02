# hybrid_control.py

import mujoco
import mujoco.viewer
import numpy as np

# --- Load model ---
model = mujoco.MjModel.from_xml_path("/home/guan/Desktop/mujoco_menagerie-main/wonik_allegro/scene_left.xml")
data = mujoco.MjData(model)

# --- Control gains ---
Kp_joint = np.array([50.0] * 16)  # Joint space proportional gains
Kd_joint = np.array([2.0] * 16)   # Joint space derivative gains
Kf = 5.0  # Force gain for hybrid control

# --- Joint setup ---
joint_names = ["ffj0", "ffj1", "ffj2", "ffj3", 
               "mfj0", "mfj1", "mfj2", "mfj3",
               "rfj0", "rfj1", "rfj2", "rfj3",
               "thj0", "thj1", "thj2", "thj3"]
joint_ids = [model.joint(name).qposadr for name in joint_names]

# --- Initialize desired joint angles ---
q_desired = np.array([
    0.2, 0.4, 0.6, 0.8,     # ff
    0.2, 0.4, 0.6, 0.8,     # mf
    0.2, 0.4, 0.6, 0.8,     # rf
    0.4, 0.6, 0.5, 0.3      # th
])

# --- Define end-effector force directions and targets ---
f_desired = 1.0  # TODO：接触法向上的期望力大小
force_dirs = {
    "ff_tip": np.array([1.0, 0.0, 0.0]),
    "mf_tip": np.array([1.0, 0.0, 0.0]),
    "rf_tip": np.array([1.0, 0.0, 0.0]),
    "th_tip": np.array([-1.0, 0.0, 0.0])
}
fingers = list(force_dirs.keys())

# --- State tracking ---
object_pos_prev = data.body("object").xpos.copy()
stability_counter = 0

# --- Launch viewer and start simulation ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(1000):
        mujoco.mj_step(model, data)

        # --- Joint control part ---
        q = data.qpos[joint_ids]
        qd = data.qvel[joint_ids]
        joint_torque = Kp_joint * (q_desired - q) - Kd_joint * qd

        # --- Force control part ---
        torque_force = np.zeros(16)
        for i, name in enumerate(fingers):
            try:
                J_pos = np.zeros((3, model.nv))
                mujoco.mj_jacBodyCom(model, data, J_pos, None, model.body(name).id)
                J_finger = J_pos[:, joint_ids[i*4:(i+1)*4]]
                f_tip = f_desired * force_dirs[name]
                torque_force[i*4:(i+1)*4] += J_finger.T @ f_tip
            except Exception as e:
                print(f"Jacobian error at {name}: {e}")

        # --- Total torque ---
        total_torque = joint_torque + Kf * torque_force
        data.ctrl[:16] = total_torque

        # --- Monitor object state and stability ---
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
            break

        viewer.sync()