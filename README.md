# **Allegro Hand Grasping and Lifting Project**

### **Team Members**
- **Tian Qing**
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
2. **Perform a coordinated grasp** using all four fingers;
3. **Lift the object** to a target height while maintaining grasp stability.

---


---

## **Features**

- **PID + IK Grasping**: Fingertips are driven to targets using PID control in Cartesian space, mapped to joints via Jacobian pseudoinverse.  
- **Grasp Detection**: Once all fingers reach within threshold, grasp is marked successful.  
- **Post-Grasp Lifting**: After grasp, the hand lifts the object vertically to a fixed height.  
- **Real-Time Visualization**: MuJoCo viewer renders the motion; contact visualization is supported.  

---

## **How to Run**

1. Place all files under proper structure and make sure **MuJoCo** is installed correctly.  
2. Confirm that `Soft.xml` is included in your scene (`scene_right.xml`).  
3. Run:

```bash
python PID_final.py
# or
python finalproject.py

