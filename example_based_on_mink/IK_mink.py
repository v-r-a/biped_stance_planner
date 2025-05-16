import os
import sys

import numpy as np
import mujoco

import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

# Global variables
model = None
data = None

# Mocap body IDs list
all_mids = [0,1,2]
curr_mid = all_mids[0]
nmids = 3

def keyboard_func(keycode):
    """
    Keyboard callback to toggle simulation pause.
    Spacebar toggles the pause state.
    """
    global all_mids, nmids, curr_mid, data, model

    incr_pos = 0.01 # metre
    incr_ang = 0.01 # radian

    # print(f"Key pressed: {keycode}")

    # Toggle between three sites using spacebar
    if keycode == 32:
        # Increment the current_mid index
        curr_mid = (curr_mid + 1) % nmids
        print(f"Current mid: {curr_mid}")
    elif keycode == 265:
        # X++
        data.mocap_pos[all_mids[curr_mid]][0] += incr_pos
    elif keycode == 264:
        # X--
        data.mocap_pos[all_mids[curr_mid]][0] -= incr_pos
    elif keycode == 263:
        # Y--
        data.mocap_pos[all_mids[curr_mid]][1] -= incr_pos
    elif keycode == 262:
        # Y++
        data.mocap_pos[all_mids[curr_mid]][1] += incr_pos
    elif keycode == 266:
        # Z++
        data.mocap_pos[all_mids[curr_mid]][2] += incr_pos
    elif keycode == 267:
        # Z--
        data.mocap_pos[all_mids[curr_mid]][2] -= incr_pos
    elif keycode == 49:
        # Pitch++
        dquat = np.array([np.cos(incr_ang/2), np.sin(incr_ang/2), 0., 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 50:
        # Pitch--
        dquat = np.array([np.cos(-incr_ang/2), np.sin(-incr_ang/2), 0., 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 51:
        # Roll++
        dquat = np.array([np.cos(incr_ang/2), 0., np.sin(incr_ang/2), 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 52:
        # Roll--
        dquat = np.array([np.cos(-incr_ang/2), 0., np.sin(-incr_ang/2), 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 53:
        # Yaw++
        dquat = np.array([np.cos(incr_ang/2), 0., 0., np.sin(incr_ang/2)])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 54:
        # Yaw--
        dquat = np.array([np.cos(-incr_ang/2), 0., 0., np.sin(-incr_ang/2)])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)

def main():
    # Read the robot model
    if len(sys.argv) != 2:
        print("Usage: python3 my_IK_program.py <path_to_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    print(f"Using model at: {model_path}")
    global model
    model = mujoco.MjModel.from_xml_path(model_path)

    # Print names of all the sites
    print("Printing all site names")
    for i in range(model.nsite):
        name = model.site(i).name
        print(f"ID: {i} name: {name}")
    # Ask user for site IDs
    values = list(map(int, input("Enter three site IDs [hip site, left foot site, right foot site] separated by space:\n").split()))
    # Foot sites name list
    site_names_list = [model.site(i).name for i in values]
    print(f"Choosing hip site: {site_names_list[0]},\n left foot site: {site_names_list[1]},\n right foot site: {site_names_list[2]}.")
    print("\n")

    # Print names of all the mocap bodies
    print("Printing all mocap body names")
    for i in range(model.nbody):
        # Skip is regular body
        if model.body_mocapid[i] == -1:
            continue
        else:
            name = model.body(i).name
            print(f"ID: {i}  Name: {name}")
    values = list(map(int, input("Enter four mocap body IDs [hip, CoM, left foot, right foot] separated by space: ").split()))
    mocap_names_list = [model.body(i).name for i in values]
    mocap_IDs_list = [model.body_mocapid[value] for value in values]
    print(f"Choosing hip mocap body: {mocap_names_list[0]},\n CoM mocap body: {mocap_names_list[1]},\n left foot: {mocap_names_list[2]},\n right foot: {mocap_names_list[3]}")

    # Choose keyframe to load
    # Print names of all the keyframes
    for i in range(model.nkey):
        name = model.keyframe(i).name
        print(f"ID: {i} name: {name}")
    key_ID = int(input("Enter the keyframe ID to load: "))
    key_name = model.keyframe(key_ID).name
    print(f"Keyframe {key_name} is chosen.")
        
    # IK tasks
    # Closeness to the current joint angles (posture)
    posture_task = mink.PostureTask(model, cost=1e-1)
    # Torso orientation
    base_orientation_task = mink.FrameTask(frame_name=site_names_list[0],
                                            frame_type="site",
                                            position_cost=0.0,
                                            orientation_cost=5.0,
                                            lm_damping=1.0)
    # Maintain COM at the said position
    com_task = mink.ComTask(cost=10.0)
    # Maintain left foot position and orientation
    task_lf = mink.FrameTask(
            frame_name=site_names_list[1],
            frame_type="site",
            position_cost=10.0,
            orientation_cost=5.0,
            lm_damping=1.0,
        )
    # Maintain right foot position and orientation    
    task_rf = mink.FrameTask(
            frame_name=site_names_list[2],
            frame_type="site",
            position_cost=10.0,
            orientation_cost=5.0,
            lm_damping=1.0,
        )
    # List of all the task objects    
    tasks_list = [base_orientation_task,
                posture_task,
                com_task,
                task_lf,
                task_rf]

    # IDs of the mocap bodies


    # Initialise mink configuration class object
    brs1_config = mink.Configuration(model)
    model = brs1_config.model
    global data
    data = brs1_config.data

    with mujoco.viewer.launch_passive(model=model, data=data, key_callback=keyboard_func,
    show_left_ui=True, show_right_ui=True) as viewer:

        # Initialize to the home keyframe.
        brs1_config.update_from_keyframe(key_name)
        posture_task.set_target_from_configuration(brs1_config)
        base_orientation_task.set_target_from_configuration(brs1_config)

        # Initialize position of mocap bodies at sites.
        mink.move_mocap_to_frame(model, data, mocap_names_list[0], site_names_list[0], "site") # Torso
        data.mocap_pos[mocap_IDs_list[1]] = data.subtree_com[1]                              # CoM
        mink.move_mocap_to_frame(model, data, mocap_names_list[2], site_names_list[1], "site") # LF
        mink.move_mocap_to_frame(model, data, mocap_names_list[3], site_names_list[2], "site") # RF

        rate = RateLimiter(frequency=60.0, warn=False)
        while viewer.is_running():

            # Update task targets based on keyboard inputs / mouse drag
            base_orientation_task.set_target(mink.SE3.from_mocap_id(data, mocap_IDs_list[0]))            
            com_task.set_target(data.mocap_pos[mocap_IDs_list[1]])
            task_lf.set_target(mink.SE3.from_mocap_id(data, mocap_IDs_list[2]))
            task_rf.set_target(mink.SE3.from_mocap_id(data, mocap_IDs_list[3]))
            
            vel = mink.solve_ik(brs1_config, tasks_list, rate.dt, "quadprog", 1e-1)

            brs1_config.integrate_inplace(vel, rate.dt)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()

    

