import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

def grasp(model, data, close_angle=0.8):
    """控制手指关节闭合实现抓握"""
    for joint_name in   ["ffj0", "ffj1", "ffj2", "ffj3",
    			 "mfj0", "mfj1", "mfj2", "mfj3",
    			 "rfj0", "rfj1", "rfj2", "rfj3",
    	   		 "thj0", "thj1", "thj2", "thj3"]:
 # 拇指
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            # 设置目标角度（根据关节类型调整）
            if '0' in joint_name:  # 基关节
                data.ctrl[joint_id] = close_angle * 0.5
            else:
                data.ctrl[joint_id] = close_angle

def get_contact_points(model, data, hand_body_name="palm"):
    """获取手与物体之间的接触点世界坐标"""
    contact_points = []
    hand_geom_ids = [i for i in range(model.ngeom) 
                    if hand_body_name in model.geom(model.body(model.geom_bodyid[i]).name)]
    
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2
        
        # 检查是否涉及手的几何体
        if (geom1 in hand_geom_ids) or (geom2 in hand_geom_ids):
            pos = contact.pos  # 接触点世界坐标
            contact_points.append(pos)
    
    return np.array(contact_points) if contact_points else None

def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path("/home/noetic/mujoco_menagerie/wonik_allegro/scene_left.xml")
    data = mujoco.MjData(model)
    
    # 初始化控制器
    for joint_id in range(model.nu):
        data.ctrl[joint_id] = 0
    
    # 运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        is_grasping = False
        while viewer.is_running():
            # 1. 控制整体下落
            hand_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
            if data.time < 2.0:  # 前2秒下落
                data.qpos[hand_root_id*7 + 2] -= 0.001  # z轴下落
            
            # 2. 检测接触后开始抓握
            contacts = get_contact_points(model, data)
            if not is_grasping and contacts is not None:
                print(f"接触点坐标:\n{contacts}")
                grasp(model, data, close_angle=1.2)
                is_grasping = True
            
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
