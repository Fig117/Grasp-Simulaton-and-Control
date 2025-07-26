# **Allegro Hand Grasping and Lifting Project**

### **Team Members**
- **Tian Qin**
- **Chenhao Guan**
- **Shuowen Li**

---

## **Overview**

This project demonstrates object grasping and manipulation using the **Allegro right hand** in the **MuJoCo physics engine**. The system integrates:

- **A physics-based simulation scene**,  
- **A biomimetic hand model**, and  
- **A Python-based control system** implementing **PID** and **Jacobian-based inverse kinematics**.

The final outcome shows that the robotic hand can successfully:
1. **Move toward a given object** in 3D space;
2. **Perform a Medium Wrap grasp** using all four fingers;
3. **Lift the object** to a target height while maintaining grasp stability.

---
## **Project Structure**
main/
├── PID_final.py # Main grasp-and-lift controller
├── scene_right.xml # Scene file loading the Allegro hand and object
├── Scene/
│ └── wonik_allegro/
│ ├── right_hand.xml # Allegro hand definition
│ ├── scene_right.xml # Alternate scene file (also loads hand + object)
│ ├── Right_hand_set/
│ │ ├── Soft.xml # Soft finger configuration
│ │ ├── Hard.xml # Hard finger configuration
│ │ └── Frictionless.xml # Frictionless finger configuration
│ └── assets/ # Meshes and textures
│ └── allegro_hand.png # Model image

---

## **Features**

- **PID + IK Grasping**: Fingertips are driven to targets using PID control in Cartesian space, mapped to joints via Jacobian pseudoinverse.  
- **Grasp Detection**: Once all fingers reach within threshold, grasp is marked successful.  
- **Post-Grasp Lifting**: After grasp, the hand lifts the object vertically to a fixed height.  
- **Real-Time Visualization**: MuJoCo viewer renders the motion; contact visualization is supported.  

---

## **How to Run**
1. Make sure [MuJoCo](https://mujoco.org/) and `mujoco-py` or `mujoco` bindings are properly installed.

2. From the `main/` directory, run the following command:

```bash
python PID_final.py
Selecting Finger Type (Soft / Hard / Frictionless)
Finger models are located in:
Scene/wonik_allegro/Right_hand_set/
Make sure the selection in Python matches the finger model in the scene file.
