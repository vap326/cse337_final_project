import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import joblib
from collections import deque

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATH = 'cricket_predictor_model.joblib'
ENCODER_PATH = 'feature_encoder.joblib'
DATA_PATH = 'cricket_shot_selection_updated.csv'

# Actions (shots the agent can choose)
ACTION_MAP = {
    0: 'Straight Drive',
    1: 'Hook',
    2: 'Cut',
    3: 'Cover Drive',
    4: 'Sweep',
    5: 'Pull',
    6: 'Defensive',
    7: 'Flick'
}
N_ACTIONS = len(ACTION_MAP)
N_OBSERVATIONS = 14  # State encoding size

# DQN Hyperparameters
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 0.0005
REPLAY_CAPACITY = 10000
BATCH_SIZE = 64
EPSILON_START = 1.0     
EPSILON_END = 0.01      
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 1000
BALLS_PER_OVER = 6      # Episode length = 6 balls

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# ENVIRONMENT: Cricket Over (6 Balls)
# ==============================================================================

class CricketOverEnv:
    """
    Cricket Over Environment
    
    Episode = One over (6 balls maximum)
    Goal: Maximize runs without getting out
    
    State: Delivery features (bowler type, length, speed, etc.)
    Action: Shot type (chosen by RL agent)
    Reward: Runs scored (0-6) or -100 for wicket
    """
    
    def __init__(self):
        # Load predictive model (reward function)
        self.predictor = joblib.load(MODEL_PATH)
        self.rl_encoder = joblib.load(ENCODER_PATH)
        self.df = pd.read_csv(DATA_PATH)
        
        # State columns
        self.state_cols = [
            'Bowler Type', 'Ball Length', 'Ball Line', 
            'Speed (km/h)', 'Field Placement', 'Angle', 'Bounce (cm)'
        ]
        self.state_df = self.df[self.state_cols].copy()
        
        # Episode tracking
        self.ball_count = 0
        self.total_runs = 0
        self.got_out = False
        self.current_state = None

    def _get_observation(self, state_dict):
        """Encode state as observation vector"""
        state_df = pd.DataFrame([state_dict])[self.state_cols]
        obs_array = self.rl_encoder.transform(state_df).flatten()
        
        if obs_array.shape[0] != N_OBSERVATIONS:
            raise RuntimeError(f"Observation size mismatch: {obs_array.shape[0]} vs {N_OBSERVATIONS}")
        
        return torch.tensor(obs_array, dtype=torch.float32, device=DEVICE)

    def _sample_delivery(self):
        """Sample a random delivery"""
        raw_sample = self.state_df.sample(1).iloc[0]
        return {col: raw_sample[col] for col in self.state_cols}

    def reset(self):
        """Start a new over"""
        self.ball_count = 0
        self.total_runs = 0
        self.got_out = False
        self.current_state = self._sample_delivery()
        
        return self._get_observation(self.current_state)

    def step(self, action_int):
        """
        Agent plays a shot (action)
        
        Args:
            action_int: Shot type chosen by agent (0-8)
            
        Returns:
            next_observation, reward, done, info
        """
        shot_type = ACTION_MAP[action_int]
        
        # 1. Create input for predictive model: (State, Action)
        pred_input = self.current_state.copy()
        pred_input['Shot Type'] = shot_type
        pred_input_df = pd.DataFrame([pred_input])
        
        # 2. Get outcome probabilities from predictive model
        probs = self.predictor.predict_proba(pred_input_df).flatten()
        
        # 3. Sample outcome (stochastic environment)
        outcome = np.random.choice(self.predictor.classes_, p=probs)
        
        # 4. Calculate reward
        if outcome == 'Wicket':
            reward = -20
            self.got_out = True
            done = True
        else:
            reward = int(outcome)
            self.total_runs += reward
            self.ball_count += 1
            done = (self.ball_count >= BALLS_PER_OVER)
        
        # 5. Sample next delivery (if over continues)
        if not done:
            self.current_state = self._sample_delivery()
        
        next_observation = self._get_observation(self.current_state)
        
        info = {
            'ball': self.ball_count,
            'total_runs': self.total_runs,
            'got_out': self.got_out,
            'outcome': outcome,
            'shot': shot_type
        }
        
        return next_observation, reward, done, info

# ==============================================================================
# DQN COMPONENTS
# ==============================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.stack(state).to(DEVICE),
            torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(DEVICE),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.stack(next_state).to(DEVICE),
            torch.tensor(done, dtype=torch.bool).unsqueeze(1).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network: Maps state to Q-values for each action"""
    
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.policy_net = QNetwork(N_OBSERVATIONS, N_ACTIONS).to(DEVICE)
        self.target_net = QNetwork(N_OBSERVATIONS, N_ACTIONS).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_CAPACITY)
        self.epsilon = EPSILON_START

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection
        
        IMPORTANT: Agent CHOOSES the action, not random!
        Random only happens during exploration phase.
        """
        if training and random.random() < self.epsilon:
            # EXPLORATION: Try random shot
            return random.randrange(N_ACTIONS)
        else:
            # EXPLOITATION: Choose best shot based on Q-values
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        
        # Current Q-values
        current_q = self.policy_net(state).gather(1, action)
        
        # Target Q-values
        with torch.no_grad():
            next_q = torch.zeros(BATCH_SIZE, 1, device=DEVICE)
            non_final = ~done.squeeze()
            if non_final.any():
                next_q[non_final] = self.target_net(next_state[non_final]).max(1)[0].unsqueeze(1)
        
        target_q = reward + (GAMMA * next_q)
        
        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path='cricket_agent.pth'):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

# ==============================================================================
# TRAINING
# ==============================================================================

def train():
    print("="*80)
    print("CRICKET SHOT SELECTION RL - 6 BALL OVER")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Goal: Maximize runs over 6 balls without getting out")
    print(f"Episodes: {NUM_EPISODES}")
    print("="*80 + "\n")
    
    env = CricketOverEnv()
    agent = DQNAgent(env)
    
    # Statistics
    episode_rewards = []
    episode_runs = []
    episode_balls_survived = []
    episode_wickets = []
    
    for episode in range(1, NUM_EPISODES + 1):
        observation = env.reset()
        total_reward = 0
        
        for ball in range(BALLS_PER_OVER):
            # Agent CHOOSES shot (not random!)
            action = agent.select_action(observation)
            
            # Environment responds
            next_observation, reward, done, info = env.step(action)
            
            # Store experience and learn
            agent.memory.push(observation, action, reward, next_observation, done)
            agent.optimize_model()
            
            observation = next_observation
            total_reward += reward
            
            if done:
                break
        
        # Record statistics
        episode_rewards.append(total_reward)
        episode_runs.append(info['total_runs'])
        episode_balls_survived.append(info['ball'])
        episode_wickets.append(1 if info['got_out'] else 0)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
        
        # Logging
        if episode % 10 == 0:
            last_10_runs = episode_runs[-10:]
            last_10_balls = episode_balls_survived[-10:]
            last_10_wickets = episode_wickets[-10:]
            
            avg_reward_50 = np.mean(episode_rewards[-50:])
            avg_runs_50 = np.mean(episode_runs[-50:])
            avg_balls_50 = np.mean(episode_balls_survived[-50:])
            
            print(f"\n{'='*80}")
            print(f"Episode {episode}/{NUM_EPISODES}")
            print(f"{'='*80}")
            print(f"Last 10 Overs:")
            print(f"  Total Runs:       {sum(last_10_runs):3d} ({np.mean(last_10_runs):.1f} per over)")
            print(f"  Wickets:          {sum(last_10_wickets)}/10")
            print(f"  Avg Balls Faced:  {np.mean(last_10_balls):.1f}/6")
            print(f"  Success Rate:     {(10-sum(last_10_wickets))/10*100:.0f}%")
            print(f"\nRolling Averages (50 episodes):")
            print(f"  Avg Reward: {avg_reward_50:7.2f}")
            print(f"  Avg Runs:   {avg_runs_50:5.2f}/6 balls")
            print(f"  Avg Balls:  {avg_balls_50:5.2f}/6")
            print(f"  Epsilon:    {agent.epsilon:.3f}")
            print(f"{'='*80}")
    
    # Final stats
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final 100 episodes:")
    print(f"  Average runs per over:  {np.mean(episode_runs[-100:]):.2f}")
    print(f"  Average balls survived: {np.mean(episode_balls_survived[-100:]):.2f}/6")
    print(f"  Wickets:                {sum(episode_wickets[-100:])}/100")
    print(f"  Success rate:           {(100-sum(episode_wickets[-100:]))}%")
    
    agent.save('cricket_agent_final.pth')
    return agent

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    agent = train()