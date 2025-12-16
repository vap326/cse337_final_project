import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import joblib
from collections import deque
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

# ==============================================================================
# 0. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# --- File Paths ---
MODEL_PATH = 'cricket_predictor_model.joblib'
ENCODER_PATH = 'feature_encoder.joblib'
DATA_PATH = 'cricket_shot_selection_updated.csv'

# --- RL Constants ---
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
OUTCOMES = ['0', '1', '2', '3', '4', '6', 'Wicket']
N_ACTIONS = len(ACTION_MAP)
N_OBSERVATIONS = 14  # Fixed: Correct observation size

# --- DQN Hyperparameters ---
GAMMA = 0.99            # Discount Factor
LEARNING_RATE = 0.0005
REPLAY_CAPACITY = 10000
BATCH_SIZE = 64
EPSILON_START = 1.0     # Initial exploration rate
EPSILON_END = 0.01      # Minimum exploration rate
EPSILON_DECAY = 0.995   # Decay factor per episode
TARGET_UPDATE = 10      # Update target network every N episodes
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 30  # Maximum balls per episode

# Set device to CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. ENVIRONMENT IMPLEMENTATION (CricketEnv)
# ==============================================================================

class CricketEnv:
    """
    Cricket Environment for Reinforcement Learning.
    
    State Space: Bowling conditions (Bowler Type, Ball Length, Ball Line, Speed, 
                 Field Placement, Angle, Bounce)
    Action Space: Shot selection (9 different cricket shots)
    Reward: Runs scored (0-6) or -100 for wicket
    """
    
    def __init__(self):
        try:
            # Load the trained predictor model and RL observation encoder
            self.predictor_pipeline = joblib.load(MODEL_PATH)
            self.rl_obs_preprocessor = joblib.load(ENCODER_PATH)
            self.df = pd.read_csv(DATA_PATH)
        except FileNotFoundError as e:
            print(f"ERROR: Required files not found: {e}")
            print("Run train_predictor.py first to generate model files!")
            exit(1)

        # Define state columns for sampling
        self.state_cols = [
            'Bowler Type', 'Ball Length', 'Ball Line', 
            'Speed (km/h)', 'Field Placement', 'Angle', 'Bounce (cm)'
        ]
        self.state_df_raw = self.df[self.state_cols].copy()
        
        self.current_state_dict = None
        self.current_step = 0
        self.action_space = N_ACTIONS
        self.observation_space = N_OBSERVATIONS
        
        # Episode statistics
        self.episode_runs = 0
        self.episode_wickets = 0

    def _get_observation(self, state_dict):
        """
        Converts raw state features into the observation vector 
        using the fitted encoder.
        
        Args:
            state_dict: Dictionary with state features
            
        Returns:
            torch.Tensor: Observation vector of size N_OBSERVATIONS
        """
        # Ensure column order matches state_cols
        state_df = pd.DataFrame([state_dict])[self.state_cols]
        
        # Transform using the fitted encoder
        obs_array = self.rl_obs_preprocessor.transform(state_df).flatten()
        
        # Verify size matches expected dimensions
        if obs_array.shape[0] != N_OBSERVATIONS:
            raise RuntimeError(
                f"Observation size mismatch! Expected {N_OBSERVATIONS}, "
                f"got {obs_array.shape[0]}. Check encoder configuration."
            )
        
        return torch.tensor(obs_array, dtype=torch.float32, device=DEVICE)

    def reset(self):
        """
        Resets the environment to start a new episode.
        
        Returns:
            torch.Tensor: Initial observation
        """
        # Sample a random delivery from the dataset
        raw_sample = self.state_df_raw.sample(1).iloc[0]
        self.current_state_dict = {col: raw_sample[col] for col in self.state_cols}
        
        # Reset episode counters
        self.current_step = 0
        self.episode_runs = 0
        self.episode_wickets = 0
        
        observation = self._get_observation(self.current_state_dict)
        return observation

    def step(self, action_int):
        """
        Executes one step in the environment.
        
        Args:
            action_int: Integer representing the selected shot
            
        Returns:
            tuple: (next_observation, reward, done, info)
        """
        # Map action to shot type
        shot_type = ACTION_MAP[action_int]
        
        # Construct input for the predictive model (State + Action)
        pred_input_raw = self.current_state_dict.copy()
        pred_input_raw['Shot Type'] = shot_type
        pred_input_df = pd.DataFrame([pred_input_raw])

        # Query the prediction model for outcome probabilities
        probs = self.predictor_pipeline.predict_proba(pred_input_df).flatten()
        
        # Sample outcome based on probabilities (stochastic environment)
        sampled_outcome = np.random.choice(
            self.predictor_pipeline.classes_, 
            p=probs
        )
        
        # Calculate reward and determine if episode ends
        done = False
        if sampled_outcome == 'Wicket':
            reward = -20
            self.episode_wickets += 1
            done = True  # Episode ends on wicket (max 1 wicket per episode)
        else:
            reward = int(sampled_outcome)
            self.episode_runs += reward

        # Increment step counter
        self.current_step += 1
        
        # End episode if max steps reached (without getting out)
        if self.current_step >= MAX_STEPS_PER_EPISODE:
            done = True
        
        # Sample next state only if episode continues
        if not done:
            raw_sample = self.state_df_raw.sample(1).iloc[0]
            self.current_state_dict = {col: raw_sample[col] for col in self.state_cols}
        
        next_observation = self._get_observation(self.current_state_dict)
        
        # Info dictionary for debugging
        info = {
            'outcome': sampled_outcome,
            'shot_type': shot_type,
            'episode_runs': self.episode_runs,
            'episode_wickets': self.episode_wickets,
            'step': self.current_step
        }
        
        return next_observation, reward, done, info

# ==============================================================================
# 2. REPLAY BUFFER
# ==============================================================================

class ReplayBuffer:
    """
    Experience Replay Memory for stable DQN learning.
    Stores transitions and samples random batches for training.
    """
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of transitions.
        
        Returns:
            tuple: Batched tensors (state, action, reward, next_state, done)
        """
        batch = random.sample(self.buffer, batch_size)
        
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to tensors
        state = torch.stack(state).to(DEVICE)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_state = torch.stack(next_state).to(DEVICE)
        done = torch.tensor(done, dtype=torch.bool).unsqueeze(1).to(DEVICE)
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# 3. Q-NETWORK ARCHITECTURE
# ==============================================================================

class QNetwork(nn.Module):
    """
    Deep Q-Network: Maps state observations to Q-values for each action.
    
    Architecture:
        Input: N_OBSERVATIONS (14-dimensional state vector)
        Hidden: 2 layers of 128 neurons with ReLU activation
        Output: N_ACTIONS (9 Q-values, one per shot type)
    """
    
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# ==============================================================================
# 4. DQN AGENT
# ==============================================================================

class DQNAgent:
    """
    Deep Q-Network Agent that learns optimal shot selection policy.
    """
    
    def __init__(self, env):
        self.env = env
        
        # Policy network (being trained)
        self.policy_net = QNetwork(N_OBSERVATIONS, N_ACTIONS).to(DEVICE)
        
        # Target network (for stable Q-value targets)
        self.target_net = QNetwork(N_OBSERVATIONS, N_ACTIONS).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=REPLAY_CAPACITY)
        
        # Exploration rate
        self.epsilon = EPSILON_START
        
        # Training statistics
        self.training_step = 0

    def select_action(self, state, training=True):
        """
        Selects action using epsilon-greedy strategy.
        
        Args:
            state: Current observation
            training: If False, always exploit (no exploration)
            
        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            # Explore: Random action
            return random.randrange(N_ACTIONS)
        else:
            # Exploit: Best action according to Q-network
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def optimize_model(self):
        """
        Performs one step of gradient descent on the Q-network.
        Implements the DQN loss function with target network.
        """
        if len(self.memory) < BATCH_SIZE:
            return None
        
        # Sample batch from replay buffer
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        
        # Compute Q(s_t, a) - predicted Q-values for taken actions
        current_q_values = self.policy_net(state).gather(1, action)
        
        # Compute V(s_{t+1}) = max_a Q_target(s_{t+1}, a) for next states
        with torch.no_grad():
            next_q_values = torch.zeros(BATCH_SIZE, 1, device=DEVICE)
            non_final_mask = ~done.squeeze()
            
            if non_final_mask.any():
                next_q_values[non_final_mask] = \
                    self.target_net(next_state[non_final_mask]).max(1)[0].unsqueeze(1)
        
        # Compute target Q-values: r + gamma * V(s_{t+1})
        target_q_values = reward + (GAMMA * next_q_values)
        
        # Compute Huber loss (smooth L1)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()

    def update_target_network(self):
        """Copies policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decays exploration rate."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path='cricket_dqn_agent.pth'):
        """Saves the trained agent."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path='cricket_dqn_agent.pth'):
        """Loads a trained agent."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Agent loaded from {path}")

# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================

def train_agent():
    """Main training loop for the DQN agent."""
    
    print("="*70)
    print("CRICKET SHOT SELECTION - DQN TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Observation Size: {N_OBSERVATIONS}")
    print(f"Action Space: {N_ACTIONS} shots")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Max Steps per Episode: {MAX_STEPS_PER_EPISODE}")
    print("="*70 + "\n")
    
    # Initialize environment and agent
    env = CricketEnv()
    agent = DQNAgent(env)
    
    # Training statistics
    episode_rewards = []
    episode_runs = []
    episode_lengths = []
    episode_wickets = []
    episode_balls_before_first_wicket = []  # New tracking
    recent_losses = []
    
    # Best model tracking
    best_avg_reward = -float('inf')
    
    for episode in range(1, NUM_EPISODES + 1):
        # Reset environment
        observation = env.reset()
        total_reward = 0
        done = False
        episode_losses = []
        first_wicket_ball = None  # Track when first wicket occurs
        
        # Episode loop
        step_count = 0
        while not done:
            # Select and perform action
            action = 6 #agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            step_count += 1
            
            # Track first wicket
            if reward == -100 and first_wicket_ball is None:
                first_wicket_ball = step_count
            
            # Store transition in replay buffer
            agent.memory.push(observation, action, reward, next_observation, done)
            
            # Train the network
            loss = agent.optimize_model()
            if loss is not None:
                episode_losses.append(loss)
            
            # Move to next state
            observation = next_observation
            total_reward += reward
        
        # Episode finished - record statistics
        episode_rewards.append(total_reward)
        episode_runs.append(env.episode_runs)
        episode_lengths.append(env.current_step)
        episode_wickets.append(env.episode_wickets)
        
        # Record balls faced before first wicket (or full episode if no wicket)
        if first_wicket_ball is not None:
            episode_balls_before_first_wicket.append(first_wicket_ball)
        else:
            episode_balls_before_first_wicket.append(env.current_step)
        
        if episode_losses:
            recent_losses.append(np.mean(episode_losses))
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Detailed logging every 10 episodes
        if episode % 10 == 0:
            # Get last 10 episodes stats
            last_10_runs = episode_runs[-10:]
            last_10_balls_before_wicket = episode_balls_before_first_wicket[-10:]
            last_10_wickets = episode_wickets[-10:]
            last_10_rewards = episode_rewards[-10:]
            
            # Get last 50 episodes for averages
            avg_reward_50 = np.mean(episode_rewards[-50:])
            avg_runs_50 = np.mean(episode_runs[-50:])
            avg_length_50 = np.mean(episode_lengths[-50:])
            avg_loss = np.mean(recent_losses[-50:]) if recent_losses else 0
            
            print(f"\n{'='*70}")
            print(f"Episode {episode:4d}/{NUM_EPISODES}")
            print(f"{'='*70}")
            print(f"Last 10 Episodes Performance:")
            print(f"  Total Runs:              {sum(last_10_runs):3d} runs ({np.mean(last_10_runs):.1f} per episode)")
            print(f"  Total Wickets:           {sum(last_10_wickets):3d} wickets ({np.mean(last_10_wickets):.1f} per episode)")
            print(f"  Avg Balls Before Wicket: {np.mean(last_10_balls_before_wicket):5.1f} balls")
            print(f"  Min Balls Before Wicket: {min(last_10_balls_before_wicket):3d} balls")
            print(f"  Max Balls Before Wicket: {max(last_10_balls_before_wicket):3d} balls")
            print(f"  Avg Reward:              {np.mean(last_10_rewards):7.2f}")
            print(f"\nRolling Averages (Last 50 episodes):")
            print(f"  Avg Reward: {avg_reward_50:7.2f}")
            print(f"  Avg Runs:   {avg_runs_50:5.2f}")
            print(f"  Avg Steps:  {avg_length_50:5.1f}")
            print(f"\nTraining Metrics:")
            print(f"  Loss:       {avg_loss:.4f}")
            print(f"  Epsilon:    {agent.epsilon:.3f}")
            print(f"  Buffer:     {len(agent.memory)}/{REPLAY_CAPACITY}")
            print(f"{'='*70}\n")
            
            # Save best model
            if avg_reward_50 > best_avg_reward:
                best_avg_reward = avg_reward_50
                agent.save('cricket_dqn_best.pth')
    
    # Final statistics
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final Average Reward (last 100):            {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final Average Runs (last 100):              {np.mean(episode_runs[-100:]):.2f}")
    print(f"Final Average Balls Before Wicket (last 100): {np.mean(episode_balls_before_first_wicket[-100:]):.1f}")
    print(f"Final Average Wickets per Episode (last 100): {np.mean(episode_wickets[-100:]):.2f}")
    print(f"Best Average Reward: {best_avg_reward:.2f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    
    # Save final model
    agent.save('cricket_dqn_final.pth')
    
    # Return all statistics for plotting
    return agent, {
        'rewards': episode_rewards,
        'runs': episode_runs,
        'balls_before_wicket': episode_balls_before_first_wicket,
        'wickets': episode_wickets
    }

# ==============================================================================
# 6. EVALUATION
# ==============================================================================

def evaluate_agent(agent, num_episodes=100):
    """
    Evaluates the trained agent without exploration.
    
    How it works:
    1. Sets training=False so agent always exploits (no random actions)
    2. Runs agent for num_episodes to collect performance statistics
    3. Analyzes shot selection patterns, survival rates, scoring ability
    
    Args:
        agent: Trained DQNAgent
        num_episodes: Number of evaluation episodes
        
    Returns:
        dict: Evaluation statistics
    """
    print("\n" + "="*70)
    print("EVALUATING AGENT (Greedy Policy - No Exploration)")
    print("="*70)
    
    env = agent.env
    eval_rewards = []
    eval_runs = []
    eval_wickets = []
    eval_balls_before_wicket = []
    shot_distribution = {shot: 0 for shot in ACTION_MAP.values()}
    
    # Track outcomes by shot type
    shot_outcomes = {shot: {'runs': 0, 'wickets': 0, 'attempts': 0} 
                     for shot in ACTION_MAP.values()}
    
    for episode in range(num_episodes):
        observation = env.reset()
        total_reward = 0
        done = False
        first_wicket_ball = None
        step_count = 0
        
        while not done:
            step_count += 1
            
            # Agent selects best action (no exploration)
            action = agent.select_action(observation, training=False)
            shot_name = ACTION_MAP[action]
            shot_distribution[shot_name] += 1
            shot_outcomes[shot_name]['attempts'] += 1
            
            observation, reward, done, info = env.step(action)
            total_reward += reward
            
            # Track outcomes
            if reward == -100:
                shot_outcomes[shot_name]['wickets'] += 1
                if first_wicket_ball is None:
                    first_wicket_ball = step_count
            else:
                shot_outcomes[shot_name]['runs'] += reward
        
        eval_rewards.append(total_reward)
        eval_runs.append(env.episode_runs)
        eval_wickets.append(env.episode_wickets)
        
        if first_wicket_ball is not None:
            eval_balls_before_wicket.append(first_wicket_ball)
        else:
            eval_balls_before_wicket.append(env.current_step)
    
    # Print results
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward:           {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Average Runs:             {np.mean(eval_runs):.2f} ± {np.std(eval_runs):.2f}")
    print(f"  Average Wickets:          {np.mean(eval_wickets):.2f}")
    print(f"  Survival Rate:            {(1 - np.mean(eval_wickets))*100:.1f}%")
    print(f"  Avg Balls Before Wicket:  {np.mean(eval_balls_before_wicket):.1f}")
    
    print(f"\nShot Selection Distribution:")
    total_shots = sum(shot_distribution.values())
    for shot, count in sorted(shot_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_shots) * 100
        outcomes = shot_outcomes[shot]
        avg_runs = outcomes['runs'] / max(outcomes['attempts'], 1)
        wicket_rate = outcomes['wickets'] / max(outcomes['attempts'], 1) * 100
        print(f"  {shot:15s}: {count:4d} ({percentage:5.1f}%) | "
              f"Avg: {avg_runs:.2f} runs | Wicket Rate: {wicket_rate:.1f}%")
    
    return {
        'rewards': eval_rewards,
        'runs': eval_runs,
        'wickets': eval_wickets,
        'balls_before_wicket': eval_balls_before_wicket,
        'shot_distribution': shot_distribution,
        'shot_outcomes': shot_outcomes
    }

def plot_training_progress(episode_rewards, episode_runs, episode_balls_before_wicket, 
                          episode_wickets, save_path='training_progress.png'):
    """
    Creates comprehensive visualization of training progress.
    
    Args:
        episode_rewards: List of total rewards per episode
        episode_runs: List of runs scored per episode
        episode_balls_before_wicket: List of balls survived per episode
        episode_wickets: List of wickets lost per episode
        save_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Create moving averages for smoother visualization
    window = 50
    
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('DQN Training Progress - Cricket Shot Selection', fontsize=16, fontweight='bold')
    
    # 1. Average Reward Per Episode (NEW - Main Learning Curve)
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_rewards, alpha=0.3, color='purple', linewidth=0.5, label='Raw')
    if len(episode_rewards) >= window:
        ma_rewards = moving_average(episode_rewards, window)
        ax1.plot(episodes[window-1:], ma_rewards, color='darkviolet', linewidth=2.5, 
                label=f'{window}-Episode Moving Avg')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Average Reward', fontsize=11)
    ax1.set_title('Learning Curve: Average Reward Per Episode', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Rewards (Cumulative)
    ax2 = axes[0, 1]
    ax2.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    if len(episode_rewards) >= window:
        ma_rewards = moving_average(episode_rewards, window)
        ax2.plot(episodes[window-1:], ma_rewards, color='darkblue', linewidth=2, 
                label=f'{window}-Episode Moving Avg')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Total Reward', fontsize=11)
    ax2.set_title('Cumulative Reward Per Episode', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Runs Scored
    ax3 = axes[0, 2]
    ax3.plot(episodes, episode_runs, alpha=0.3, color='green', label='Raw')
    if len(episode_runs) >= window:
        ma_runs = moving_average(episode_runs, window)
        ax3.plot(episodes[window-1:], ma_runs, color='darkgreen', linewidth=2,
                label=f'{window}-Episode Moving Avg')
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Runs Scored', fontsize=11)
    ax3.set_title('Runs Scored Per Episode', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Balls Before Wicket (Survival)
    ax4 = axes[1, 0]
    ax4.plot(episodes, episode_balls_before_wicket, alpha=0.3, color='orange', label='Raw')
    if len(episode_balls_before_wicket) >= window:
        ma_balls = moving_average(episode_balls_before_wicket, window)
        ax4.plot(episodes[window-1:], ma_balls, color='darkorange', linewidth=2,
                label=f'{window}-Episode Moving Avg')
    ax4.axhline(y=MAX_STEPS_PER_EPISODE, color='red', linestyle='--', 
               alpha=0.5, label=f'Max ({MAX_STEPS_PER_EPISODE} balls)')
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Balls Faced Before Wicket', fontsize=11)
    ax4.set_title('Survival Rate (Balls Before First Wicket)', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Wickets Per Episode
    ax5 = axes[1, 1]
    ax5.plot(episodes, episode_wickets, alpha=0.3, color='red', label='Raw')
    if len(episode_wickets) >= window:
        ma_wickets = moving_average(episode_wickets, window)
        ax5.plot(episodes[window-1:], ma_wickets, color='darkred', linewidth=2,
                label=f'{window}-Episode Moving Avg')
    ax5.set_xlabel('Episode', fontsize=11)
    ax5.set_ylabel('Wickets Lost', fontsize=11)
    ax5.set_title('Wickets Per Episode', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Success Rate (% of episodes survived)
    ax6 = axes[1, 2]
    # Calculate rolling success rate (survived full episode)
    survived = [1 if w == 0 else 0 for w in episode_wickets]
    if len(survived) >= window:
        rolling_success = []
        for i in range(window-1, len(survived)):
            success_rate = sum(survived[i-window+1:i+1]) / window * 100
            rolling_success.append(success_rate)
        ax6.plot(episodes[window-1:], rolling_success, color='teal', linewidth=2,
                label=f'{window}-Episode Success Rate')
    ax6.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    ax6.set_xlabel('Episode', fontsize=11)
    ax6.set_ylabel('Success Rate (%)', fontsize=11)
    ax6.set_title('Episode Survival Rate (No Wicket)', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 100)
    ax6.legend(loc='lower right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved to: {save_path}")
    plt.close()

def plot_evaluation_results(eval_stats, save_path='evaluation_results.png'):
    """
    Creates visualization of evaluation results.
    
    Args:
        eval_stats: Dictionary returned by evaluate_agent()
        save_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Evaluation Results (Greedy Policy)', fontsize=16, fontweight='bold')
    
    # 1. Reward Distribution
    ax1 = axes[0, 0]
    ax1.hist(eval_stats['rewards'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(eval_stats['rewards']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(eval_stats["rewards"]):.1f}')
    ax1.set_xlabel('Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Runs Distribution
    ax2 = axes[0, 1]
    ax2.hist(eval_stats['runs'], bins=20, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(eval_stats['runs']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(eval_stats["runs"]):.1f}')
    ax2.set_xlabel('Runs Scored')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Runs Per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Shot Selection Distribution
    ax3 = axes[1, 0]
    shot_dist = eval_stats['shot_distribution']
    shots = list(shot_dist.keys())
    counts = list(shot_dist.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(shots)))
    
    bars = ax3.bar(range(len(shots)), counts, color=colors, edgecolor='black')
    ax3.set_xticks(range(len(shots)))
    ax3.set_xticklabels(shots, rotation=45, ha='right')
    ax3.set_ylabel('Number of Times Selected')
    ax3.set_title('Shot Selection Frequency')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    # 4. Shot Effectiveness
    ax4 = axes[1, 1]
    shot_outcomes = eval_stats['shot_outcomes']
    
    # Filter shots that were actually used
    used_shots = [shot for shot, data in shot_outcomes.items() if data['attempts'] > 0]
    avg_runs = [shot_outcomes[shot]['runs'] / shot_outcomes[shot]['attempts'] 
                for shot in used_shots]
    wicket_rates = [shot_outcomes[shot]['wickets'] / shot_outcomes[shot]['attempts'] * 100 
                    for shot in used_shots]
    
    x = np.arange(len(used_shots))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, avg_runs, width, label='Avg Runs', 
                    color='green', alpha=0.7, edgecolor='black')
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, wicket_rates, width, label='Wicket Rate (%)', 
                         color='red', alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel('Shot Type')
    ax4.set_ylabel('Average Runs per Shot', color='green')
    ax4_twin.set_ylabel('Wicket Rate (%)', color='red')
    ax4.set_title('Shot Effectiveness')
    ax4.set_xticks(x)
    ax4.set_xticklabels(used_shots, rotation=45, ha='right')
    ax4.tick_params(axis='y', labelcolor='green')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation results plot saved to: {save_path}")
    plt.close()

def compare_training_vs_evaluation(training_stats, eval_stats, save_path='comparison.png'):
    """
    Compares training performance vs final evaluation.
    
    Args:
        training_stats: Dict with training episode data
        eval_stats: Dict returned by evaluate_agent()
        save_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training vs Evaluation Comparison', fontsize=16, fontweight='bold')
    
    # Get last 100 episodes of training for comparison
    train_rewards = training_stats['rewards'][-100:]
    train_runs = training_stats['runs'][-100:]
    train_survival = training_stats['balls_before_wicket'][-100:]
    
    # 1. Rewards Comparison
    ax1 = axes[0]
    data = [train_rewards, eval_stats['rewards']]
    bp1 = ax1.boxplot(data, labels=['Training\n(Last 100)', 'Evaluation\n(Greedy)'],
                      patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add mean markers
    ax1.plot([1, 2], [np.mean(train_rewards), np.mean(eval_stats['rewards'])],
            'ro-', linewidth=2, markersize=8, label='Mean')
    ax1.legend()
    
    # 2. Runs Comparison
    ax2 = axes[1]
    data = [train_runs, eval_stats['runs']]
    bp2 = ax2.boxplot(data, labels=['Training\n(Last 100)', 'Evaluation\n(Greedy)'],
                      patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightgreen')
    ax2.set_ylabel('Runs Scored')
    ax2.set_title('Runs Per Episode')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax2.plot([1, 2], [np.mean(train_runs), np.mean(eval_stats['runs'])],
            'ro-', linewidth=2, markersize=8, label='Mean')
    ax2.legend()
    
    # 3. Survival Comparison
    ax3 = axes[2]
    data = [train_survival, eval_stats['balls_before_wicket']]
    bp3 = ax3.boxplot(data, labels=['Training\n(Last 100)', 'Evaluation\n(Greedy)'],
                      patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightblue')
    bp3['boxes'][1].set_facecolor('lightgreen')
    ax3.set_ylabel('Balls Before Wicket')
    ax3.set_title('Survival Rate')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=MAX_STEPS_PER_EPISODE, color='red', linestyle='--', 
               alpha=0.5, label=f'Max ({MAX_STEPS_PER_EPISODE})')
    
    ax3.plot([1, 2], [np.mean(train_survival), np.mean(eval_stats['balls_before_wicket'])],
            'ro-', linewidth=2, markersize=8, label='Mean')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Train the agent
    print("Starting training...")
    agent, training_stats = train_agent()
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot training progress
    plot_training_progress(
        training_stats['rewards'],
        training_stats['runs'],
        training_stats['balls_before_wicket'],
        training_stats['wickets'],
        save_path='training_progress.png'
    )
    
    # Evaluate the trained agent
    print("\nRunning evaluation...")
    eval_stats = evaluate_agent(agent, num_episodes=100)
    
    # Plot evaluation results
    plot_evaluation_results(eval_stats, 'evaluation_results.png')
    
    # Compare training vs evaluation
    compare_training_vs_evaluation(training_stats, eval_stats, 'comparison.png')
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print("Generated files:")
    print("  1. training_progress.png - Shows learning over time")
    print("  2. evaluation_results.png - Final agent performance")
    print("  3. comparison.png - Training vs Evaluation comparison")
    print("="*70)