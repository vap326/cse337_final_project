import torch
import random
import numpy as np
import os
import time
import mujoco
import mujoco_viewer
import glfw

# IMPORTANT: Ensure this matches your training filename
from dqn_with_graphs import CricketOverEnv, DQNAgent, ACTION_MAP

class TrainingMetrics:
    pass

# ==============================================================================
# 1. TUNED PHYSICS XML (Stable & Timed)
# ==============================================================================

CRICKET_XML = """
<mujoco model="cricket_simple">
    <option gravity="0 0 -9.81" timestep="0.002"/>
    
    <visual>
        <global offwidth="1920" offheight="1080"/>
        <map fogstart="3" fogend="10" force="0.1" znear="0.1"/>
        <quality shadowsize="2048"/>
    </visual>

    <worldbody>
        <light diffuse=".6 .6 .6" pos="0 0 4" dir="0 0 -1" castshadow="true"/>
        <geom name="ground" type="plane" size="10 10 0.1" rgba=".3 .8 .3 1"/>
        <geom name="pitch" type="box" pos="0 2 0.01" size="1 4 0.01" rgba=".8 .7 .5 1"/>

        <body name="wickets" pos="0 0 0">
            <geom name="stump1" type="cylinder" size="0.04 0.7" pos="-0.1 0 0.7" rgba="0.6 0.4 0.2 1"/>
            <geom name="stump2" type="cylinder" size="0.04 0.7" pos="0 0 0.7"    rgba="0.6 0.4 0.2 1"/>
            <geom name="stump3" type="cylinder" size="0.04 0.7" pos="0.1 0 0.7"   rgba="0.6 0.4 0.2 1"/>
        </body>

        <body name="bat" pos="0 0.5 0.8" euler="90 0 0">
            <joint name="bat_hinge" type="hinge" axis="1 0 0" range="-90 90" damping="1.0"/>
            <geom name="bat_handle" type="cylinder" size="0.025 0.2" pos="0 0 0.3" rgba="0 0 0 1" mass="0.5"/>
            <geom name="bat_blade" type="box" size="0.08 0.03 0.4" pos="0 0 -0.2" rgba="0.6 0.4 0.2 1" mass="0.5"/>
        </body>

        <body name="ball" pos="0 8 1.0">
            <joint name="ball_joint" type="free"/>
            <geom name="ball_geom" type="sphere" size="0.05" rgba="0.8 0.1 0.1 1" mass="0.16"/>
        </body>
    </worldbody>
    
    <actuator>
        <position name="swing_servo" joint="bat_hinge" kp="300" kv="40" ctrlrange="-2.0 2.0"/>
    </actuator>
</mujoco>
"""

# ==============================================================================
# 2. VISUALIZER CLASS
# ==============================================================================

class CricketVisualizer:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(CRICKET_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # IDs
        self.bat_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "swing_servo")
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.bat_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bat_hinge")
        
        self.ball_qpos_adr = self.model.jnt_qposadr[self.ball_joint_id]
        self.ball_dof_adr = self.model.jnt_dofadr[self.ball_joint_id]
        self.bat_qpos_adr = self.model.jnt_qposadr[self.bat_joint_id]
        self.bat_dof_adr = self.model.jnt_dofadr[self.bat_joint_id]

        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 11.0
        self.viewer.cam.lookat[1] = 2.0

    def reset_physics(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. Reset Ball
        self.data.qpos[self.ball_qpos_adr + 1] = 8.0 
        self.data.qpos[self.ball_qpos_adr + 2] = 1.0 
        
        # 2. Reset Bat (Cocked Back)
        start_angle = -1.2 
        self.data.qpos[self.bat_qpos_adr] = start_angle
        self.data.qvel[self.bat_dof_adr] = 0.0 # Stop any movement
        self.data.ctrl[self.bat_actuator_id] = start_angle
        
        # 3. Warm Up
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def visualize_shot(self, action_name, outcome, ball, total_runs):
        self.reset_physics()
        
        # Throw Ball (-9.0 speed)
        self.data.qvel[self.ball_dof_adr + 1] = -33.0 
        
        impact_happened = False
        steps = 0
        swing_triggered = False

        status_text = f"Ball: {ball}/6 | Shot: {action_name} | Outcome: {outcome} | Total Runs: {total_runs}"
        
        # Try to set the title if the viewer has a window
        if hasattr(self.viewer, 'window'):
            glfw.set_window_title(self.viewer.window, status_text)
        
        while steps < 300:
            ball_y = self.data.qpos[self.ball_qpos_adr + 1]

            
            # --- SWING TIMING FIX ---
            # Wait until ball is at 2.0m (much closer)
            # You can tweak this number: 
            #   < 2.5 : Swings Earlier
            #   < 1.5 : Swings Later
            if ball_y < 1.0 and not swing_triggered:
                self.data.ctrl[self.bat_actuator_id] = 1.2 # Swing!
                swing_triggered = True
            
            # --- IMPACT LOGIC ---
            if ball_y < 0.6 and not impact_happened:
                impact_happened = True
                if outcome == 'Wicket':
                    self.data.qvel[self.ball_dof_adr] = 0
                    self.data.qvel[self.ball_dof_adr+1] = -2
                    self.data.qvel[self.ball_dof_adr+2] = 2
                else:
                    fx, fy, fz = self._get_shot_vector(action_name)
                    self.data.qvel[self.ball_dof_adr] = fx
                    self.data.qvel[self.ball_dof_adr+1] = fy
                    self.data.qvel[self.ball_dof_adr+2] = fz

            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            steps += 1
            
            time.sleep(0.002) 
        
        time.sleep(0.2)

    def _get_shot_vector(self, shot_name):
        # (X=Left/Right, Y=Fwd/Back, Z=Up
        vectors = {
            'Straight Drive': (0, 15, 5),
            'Cover Drive':    (10, 10, 3),
            'Hook':           (-10, -5, 10),
            'Pull':           (-12, 6, 6),
            'Cut':            (12, 0, 4),
            'Sweep':          (-8, 3, 2),
            'Defensive':      (0, 2, -2),
            'Flick':          (-8, 8, 4)
        }
        return vectors.get(shot_name, (0, 5, 5))

    def close(self):
        self.viewer.close()

# ==============================================================================
# 3. MAIN
# ==============================================================================

def run_simulation(model_path='cricket_agent_final.pth', num_balls=6):
    print(f"LOADING AGENT FROM: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    env = CricketOverEnv()
    agent = DQNAgent(env)
    agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()

    visualizer = CricketVisualizer()
    
    try:
        print("\nStarting Simulation...")
        totalRuns = 0
        for i in range(num_balls):
            obs = env.reset()
            action_idx = agent.select_action(obs, training=False)
            shot_name = ACTION_MAP[action_idx]
            
            _, reward, done, info = env.step(action_idx)
            outcome = info['outcome']
            if outcome != 'Wicket':
                totalRuns += int(outcome)
            
            print(f"Ball {i+1}: {shot_name:15s} -> Outcome: {outcome}")
            
            visualizer.visualize_shot(shot_name, outcome, i+1, totalRuns)
            
            if done: 
                 print("  (Episode End - Wicket or Max Balls)")
                 break

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        visualizer.close()

if __name__ == '__main__':
    run_simulation()


