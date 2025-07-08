import numpy as np
import mujoco
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("scene_right.xml")
data = mujoco.MjData(model)

# actuator name1
actuator_names = [
    "ffa0", "ffa1", "ffa2", "ffa3",
    "mfa0", "mfa1", "mfa2", "mfa3",
    "rfa0", "rfa1", "rfa2", "rfa3",
    "tha0", "tha1", "tha2", "tha3"
]

actuator_ids = [model.actuator(name).id for name in actuator_names]

# 目标角度（手指闭合）
target_positions = np.array([
    0.0,   0.879, 0.946, 1.33,   # ffa0 ~ ffa3（First finger）
    0.0,   0.752, 0.88,  0.622,  # mfa0 ~ mfa3（Middle finger）
    0.0,   1.06,  0.589, 1.32,   # rfa0 ~ rfa3（Ring finger）
    1.38,  0.592, 0.407, 0.788   # tha0 ~ tha3（Thumb）
])


# 初始化控制值
data.ctrl[:] = 0.0
speed = 0.01  # 收拢速度

def check_grasp(model, data):
    object_geom_name = "object_geom"
    fingertip_geom_names = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]

    try:
        object_gid = model.geom(object_geom_name)
    except:
        print(f"⚠️ 找不到目标物体 geom 名：{object_geom_name}")
        return False

    fingertip_gids = []
    for name in fingertip_geom_names:
        try:
            fingertip_gids.append(model.geom(name))
        except:
            print(f"⚠️ 找不到手指 geom 名：{name}")

    contact_count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        print("🟡 发生接触：", g1_name, "<-->", g2_name)

        if (g1 == object_gid and g2 in fingertip_gids) or (g2 == object_gid and g1 in fingertip_gids):
            contact_count += 1

    print("✅ 指尖接触数:", contact_count)
    return contact_count >= 2

print("模型中 actuator 数:", model.nu)

# 启动 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 平滑逼近目标位置
        for i, act_id in enumerate(actuator_ids):
            delta = target_positions[i] - data.ctrl[act_id]
            data.ctrl[act_id] += np.clip(delta, -speed, speed)

        mujoco.mj_step(model, data)

        if data.time * model.opt.timestep % 10 < 1e-5:
            if check_grasp(model, data):
                print("✅ 成功抓住球！")
                np.savez("grasp_snapshot_right.npz",
                         qpos=data.qpos.copy(),
                         qvel=data.qvel.copy(),
                         ctrl=data.ctrl.copy(),
                         time=data.time,
                         xpos=data.xpos.copy(),
                         xquat=data.xquat.copy())
                print("💾 仿真数据已保存为 grasp_snapshot_right.npz")
                break

        viewer.sync()
