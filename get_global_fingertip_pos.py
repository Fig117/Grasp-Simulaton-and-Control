import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/noetic/.mujoco/mujoco-3.3.3/bin/mjmodel.xml")
data = mujoco.MjData(model)

# 指尖站点名称列表
fingertip_sites = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        # 获取并打印所有指尖位置
        for tip in fingertip_sites:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip)
            pos = data.site_xpos[site_id]
            print(f"{tip} position: {pos}")
        
        viewer.sync()