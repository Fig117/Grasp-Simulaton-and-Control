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

## **File Structure**
Project/
├── Sceneproject.txt # Main scene including hand and movable object
├── Righthandproject.txt # Full Allegro hand model (right)
├── Soft.xml # Soft-finger configuration used in the simulation
├── PID_final.py # Grasping and lifting controller (PID + Jacobian)
├── finalproject.py # Extended grasp-and-lift controller with contact analysis

- **Sceneproject.txt**: Main MuJoCo XML scene file, referencing `right_hand.xml` to include the full Allegro model and a red box as the grasp target.  
- **Righthandproject.txt**: Detailed model of the **right Allegro hand**, including all finger joints, actuators, and sensors.  
- **Soft.xml**: One of the three finger types (**Soft / Hard / Frictionless**); this is the one used in `scene_right.xml`.  
- **PID_final.py**: Controls the hand’s palm motion and finger grasping using **PID** and **inverse kinematics**.  
- **finalproject.py**: An extended version with **contact force monitoring** and **object lifting** after grasp.  

> 💡 **Note**: We demonstrate **one Allegro hand** in the scene, but the framework supports multiple hands (not included here).

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

