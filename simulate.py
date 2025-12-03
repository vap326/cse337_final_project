import torch
import random
import numpy as np
import os
import time
import mujoco
import mujoco_viewer

# Import RL logic
from rl_agent import CricketEnv, DQNAgent, ACTION_MAP

# ==============================================================================
# 1. MUJOCO XML (With Position Servo)
# ==============================================================================

CRICKET_XML = """
<mujoco model="cricket_simple">
    <option gravity="0 0 -9.81" timestep="0.005"/>
    
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
            <geom name="bat_blade" type="box" size="0.08 0.03 0.4" pos="0 0 -0.2" rgba="0.6 0.4 0.2 1" mass="1.0"/>
        </body>

        <body name="ball" pos="0 8 1.0">
            <joint name="ball_joint" type="free"/>
            <geom name="ball_geom" type="sphere" size="0.05" rgba="0.8 0.1 0.1 1" mass="0.16"/>
        </body>
    </worldbody>
    
    <actuator>
        <position name="swing_servo" joint="bat_hinge" kp="500" kv="50" ctrlrange="-1.5 1.5"/>
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
        self.ball_qpos_adr = self.model.jnt_qposadr[self.ball_joint_id]
        self.ball_dof_adr = self.model.jnt_dofadr[self.ball_joint_id]

        # Camera
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 11.0
        self.viewer.cam.lookat[1] = 2.0

    def reset_physics(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset ball position (Way down the pitch)
        self.data.qpos[self.ball_qpos_adr + 1] = 8.0 # Y
        self.data.qpos[self.ball_qpos_adr + 2] = 1.0 # Z
        
        # Reset Bat Position (Cocked back: -1.0 radian)
        self.data.ctrl[self.bat_actuator_id] = -1.0 
        
        mujoco.mj_forward(self.model, self.data)

    def visualize_shot(self, action_name, outcome):
        self.reset_physics()
        
        # Throw the ball (Velocity towards bat)
        self.data.qvel[self.ball_dof_adr + 1] = -12.0 
        
        impact_happened = False
        steps = 0
        swing_triggered = False
        
        while steps < 250:
            # Get Ball Y Position
            ball_y = self.data.qpos[self.ball_qpos_adr + 1]
            
            # --- SWING LOGIC (SERVO) ---
            # When ball is 3.5m away, set servo target to forward (+1.2 rad)
            if ball_y < 3.5 and not swing_triggered:
                self.data.ctrl[self.bat_actuator_id] = 1.2 # Swing forward!
                swing_triggered = True
            
            # --- IMPACT LOGIC ---
            # Override physics for visual clarity when ball reaches hitting zone
            if ball_y < 0.6 and not impact_happened:
                impact_happened = True
                
                if outcome == 'Wicket':
                    # Kill side velocity, ensure it hits stumps
                    self.data.qvel[self.ball_dof_adr] = 0
                    self.data.qvel[self.ball_dof_adr+1] = -5 # Keep going back slowly
                    self.data.qvel[self.ball_dof_adr+2] = 2  # Pop up slightly
                else:
                    # Apply fake force vector based on shot type
                    fx, fy, fz = self._get_shot_vector(action_name)
                    self.data.qvel[self.ball_dof_adr] = fx
                    self.data.qvel[self.ball_dof_adr+1] = fy
                    self.data.qvel[self.ball_dof_adr+2] = fz

            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            steps += 1
        
        # Small pause between balls
        time.sleep(0.3)

    def _get_shot_vector(self, shot_name):
        # (X=Left/Right, Y=Fwd/Back, Z=Up)
        vectors = {
            'Straight Drive': (0, 18, 5),
            'Cover Drive':    (12, 12, 3),
            'Edge':           (3, -8, 2),
            'Hook':           (-12, -5, 12),
            'Pull':           (-15, 8, 8),
            'Cut':            (14, 0, 4),
            'Sweep':          (-10, 4, 2),
            'Defensive':      (0, 3, -2), # Hit down
            'Flick':          (-10, 10, 4)
        }
        return vectors.get(shot_name, (0, 5, 5))

    def close(self):
        self.viewer.close()

# ==============================================================================
# 3. MAIN
# ==============================================================================

def run_simulation(model_path='cricket_dqn_final.pth', num_balls=20):
    print(f"LOADING AGENT FROM: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    # Load Agent
    env = CricketEnv()
    agent = DQNAgent(env)
    agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()

    visualizer = CricketVisualizer()
    
    try:
        print("\nStarting Simulation...")
        for i in range(num_balls):
            obs = env.reset()
            action_idx = agent.select_action(obs, training=False)
            shot_name = ACTION_MAP[action_idx]
            
            _, reward, done, info = env.step(action_idx)
            outcome = info['outcome']
            
            # Print to console instead of screen overlay
            print(f"Ball {i+1}: {shot_name:15s} -> Outcome: {outcome}")
            
            visualizer.visualize_shot(shot_name, outcome)
            
            if done: 
                 print("  (Episode End - Wicket or Max Balls)")

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        visualizer.close()

if __name__ == '__main__':
    run_simulation()

'''import torch
import random
import numpy as np
import os
import time
import mujoco
import mujoco_viewer

# We ONLY import the RL logic from your training script
# Make sure your training file is named 'cricket_rl.py'
from rl_agent import CricketEnv, DQNAgent, ACTION_MAP

# ==============================================================================
# 1. MUJOCO XML DEFINITION (Lives here now)
# ==============================================================================

CRICKET_XML = """
<mujoco model="cricket_simple">
    <option gravity="0 0 -9.81" timestep="0.005"/>
    
    <visual>
        <global offwidth="1920" offheight="1080"/>
        <map fogstart="3" fogend="10" force="0.1" znear="0.1"/>
    </visual>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="10 10 0.1" rgba=".3 .8 .3 1"/>
        <geom name="pitch" type="box" pos="0 2 0.01" size="1 4 0.01" rgba=".8 .7 .5 1"/>

        <body name="wickets" pos="0 0 0">
            <geom name="stump1" type="cylinder" size="0.04 0.7" pos="-0.1 0 0.7" rgba="0.6 0.4 0.2 1"/>
            <geom name="stump2" type="cylinder" size="0.04 0.7" pos="0 0 0.7"    rgba="0.6 0.4 0.2 1"/>
            <geom name="stump3" type="cylinder" size="0.04 0.7" pos="0.1 0 0.7"   rgba="0.6 0.4 0.2 1"/>
        </body>

        <body name="bat" pos="0 0.5 0.8" euler="90 0 0">
            <joint name="bat_hinge" type="hinge" axis="1 0 0" range="-90 90" damping="0.1"/>
            <geom name="bat_handle" type="cylinder" size="0.025 0.2" pos="0 0 0.3" rgba="0 0 0 1"/>
            <geom name="bat_blade" type="box" size="0.08 0.03 0.4" pos="0 0 -0.2" rgba="0.6 0.4 0.2 1"/>
        </body>

        <body name="ball" pos="0 8 1.0">
            <joint name="ball_joint" type="free"/>
            <geom name="ball_geom" type="sphere" size="0.05" rgba="0.8 0.1 0.1 1" mass="0.16"/>
        </body>
    </worldbody>
    
    <actuator>
        <motor name="swing" joint="bat_hinge" gear="100"/>
    </actuator>
</mujoco>
"""

# ==============================================================================
# 2. VISUALIZER CLASS (Lives here now)
# ==============================================================================

class CricketVisualizer:
    def __init__(self):
        # We load the XML directly from the string above
        self.model = mujoco.MjModel.from_xml_string(CRICKET_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # Physics IDs
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.bat_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "swing")
        
        # Camera Settings
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 12.0
        self.viewer.cam.lookat[1] = 2.0

    def reset_physics(self):
        mujoco.mj_resetData(self.model, self.data)
        # Reset ball position
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.data.qpos[self.model.jnt_qposadr[ball_joint_id] + 1] = 8.0 # Y
        self.data.qpos[self.model.jnt_qposadr[ball_joint_id] + 2] = 1.0 # Z
        mujoco.mj_forward(self.model, self.data)

    def visualize_shot(self, action_name, outcome):
        self.reset_physics()
        
        ball_dof_adr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")]
        
        # Throw the ball
        self.data.qvel[ball_dof_adr + 1] = -12.0 
        
        impact_happened = False
        steps = 0
        
        while steps < 300:
            ball_y = self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")] + 1]
            
            # Swing logic
            if ball_y < 2.5 and not impact_happened:
                self.data.ctrl[self.bat_actuator_id] = 5.0
            
            # Impact logic
            if ball_y < 0.6 and not impact_happened:
                impact_happened = True
                
                if outcome == 'Wicket':
                    self.data.qvel[ball_dof_adr] = 0
                    self.data.qvel[ball_dof_adr+1] = -10
                else:
                    fx, fy, fz = self._get_shot_vector(action_name)
                    self.data.qvel[ball_dof_adr] = fx
                    self.data.qvel[ball_dof_adr+1] = fy
                    self.data.qvel[ball_dof_adr+2] = fz
            
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            steps += 1
        
        time.sleep(0.5)

    def _get_shot_vector(self, shot_name):
        vectors = {
            'Straight Drive': (0, 15, 5),
            'Cover Drive':    (10, 10, 2),
            'Edge':           (2, -8, 1),
            'Hook':           (-10, -5, 10),
            'Pull':           (-12, 5, 5),
            'Cut':            (12, 0, 2),
            'Sweep':          (-8, 2, 1),
            'Defensive':      (0, 2, -2),
            'Flick':          (-8, 8, 2)
        }
        return vectors.get(shot_name, (0, 5, 5))

    def close(self):
        self.viewer.close()

# ==============================================================================
# 3. MAIN SIMULATION LOOP
# ==============================================================================

def run_simulation(model_path='cricket_dqn_final.pth', num_balls=20):
    print(f"LOADING AGENT FROM: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train first!")
        return

    # Initialize Env and Agent from cricket_rl.py
    env = CricketEnv()
    agent = DQNAgent(env)
    
    # Load Weights
    agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()

    # Initialize Visualizer (Defined locally above)
    visualizer = CricketVisualizer()
    
    try:
        print("\nStarting Simulation...")
        for i in range(num_balls):
            obs = env.reset()
            action_idx = agent.select_action(obs, training=False)
            shot_name = ACTION_MAP[action_idx]
            
            _, reward, done, info = env.step(action_idx)
            outcome = info['outcome']
            
            print(f"Ball {i+1}: {shot_name} -> {outcome}")
            visualizer.visualize_shot(shot_name, outcome)
            
            if done: 
                print("  (Episode End)")

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        visualizer.close()

if __name__ == '__main__':
    run_simulation()
'''