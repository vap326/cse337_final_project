import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import joblib
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

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
# METRICS TRACKING
# ==============================================================================

class TrainingMetrics:
    """Comprehensive metrics tracking for DQN training"""
    
    def __init__(self):
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_runs = []
        self.episode_balls_survived = []
        self.episode_wickets = []
        self.episode_shots = []  # List of shot sequences
        
        # Step-level metrics
        self.losses = []
        self.q_values = []
        self.epsilon_history = []
        
        # Per-shot statistics
        self.shot_outcomes = {shot: {'count': 0, 'runs': 0, 'wickets': 0} 
                             for shot in ACTION_MAP.values()}
    
    def record_episode(self, reward, runs, balls, wicket, shots):
        """Record metrics for completed episode"""
        self.episode_rewards.append(reward)
        self.episode_runs.append(runs)
        self.episode_balls_survived.append(balls)
        self.episode_wickets.append(1 if wicket else 0)
        self.episode_shots.append(shots)
    
    def record_step(self, loss=None, q_value=None, epsilon=None):
        """Record metrics for training step"""
        if loss is not None:
            self.losses.append(loss)
        if q_value is not None:
            self.q_values.append(q_value)
        if epsilon is not None:
            self.epsilon_history.append(epsilon)
    
    def record_shot_outcome(self, shot, runs, wicket):
        """Track individual shot performance"""
        self.shot_outcomes[shot]['count'] += 1
        if wicket:
            self.shot_outcomes[shot]['wickets'] += 1
        else:
            self.shot_outcomes[shot]['runs'] += runs
    
    def get_recent_stats(self, window=100):
        """Get statistics for recent episodes"""
        if len(self.episode_rewards) < window:
            window = len(self.episode_rewards)
        
        if window == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_runs = self.episode_runs[-window:]
        recent_balls = self.episode_balls_survived[-window:]
        recent_wickets = self.episode_wickets[-window:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'avg_runs': np.mean(recent_runs),
            'avg_balls': np.mean(recent_balls),
            'wicket_rate': np.mean(recent_wickets),
            'success_rate': 1 - np.mean(recent_wickets),
            'strike_rate': np.mean([r/max(b,1) for r, b in zip(recent_runs, recent_balls)])
        }
    
    def plot_training_progress(self, save_path='dqn_cricket_training.png'):
        """Generate comprehensive training progress plots"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
        
        # 1. Episode Rewards
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.episode_rewards) > 0:
            window = min(50, len(self.episode_rewards))
            rewards_smooth = np.convolve(self.episode_rewards, 
                                        np.ones(window)/window, mode='valid')
            ax1.plot(self.episode_rewards, alpha=0.2, color='blue', linewidth=0.5)
            ax1.plot(range(window-1, len(self.episode_rewards)), rewards_smooth, 
                    color='blue', linewidth=2, label=f'{window}-ep MA')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax1.set_xlabel('Episode', fontsize=10)
            ax1.set_ylabel('Total Reward', fontsize=10)
            ax1.set_title('Episode Reward', fontweight='bold', fontsize=11)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Runs per Over
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.episode_runs) > 0:
            window = min(50, len(self.episode_runs))
            runs_smooth = np.convolve(self.episode_runs,
                                     np.ones(window)/window, mode='valid')
            ax2.plot(self.episode_runs, alpha=0.2, color='green', linewidth=0.5)
            ax2.plot(range(window-1, len(self.episode_runs)), runs_smooth,
                    color='green', linewidth=2, label=f'{window}-ep MA')
            ax2.set_xlabel('Episode', fontsize=10)
            ax2.set_ylabel('Runs', fontsize=10)
            ax2.set_title('Runs per Over', fontweight='bold', fontsize=11)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Balls Survived (Survival)
        ax3 = fig.add_subplot(gs[0, 2])
        if len(self.episode_balls_survived) > 0:
            window = min(50, len(self.episode_balls_survived))
            balls_smooth = np.convolve(self.episode_balls_survived,
                                      np.ones(window)/window, mode='valid')
            ax3.plot(self.episode_balls_survived, alpha=0.2, color='orange', linewidth=0.5)
            ax3.plot(range(window-1, len(self.episode_balls_survived)), balls_smooth,
                    color='orange', linewidth=2, label=f'{window}-ep MA')
            ax3.axhline(y=6, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Full Over')
            ax3.set_xlabel('Episode', fontsize=10)
            ax3.set_ylabel('Balls Faced', fontsize=10)
            ax3.set_title('Balls Survived per Over', fontweight='bold', fontsize=11)
            ax3.set_ylim([0, 7])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Training Loss
        ax4 = fig.add_subplot(gs[0, 3])
        if len(self.losses) > 100:
            window = min(100, len(self.losses))
            loss_smooth = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            ax4.plot(loss_smooth, color='red', linewidth=2)
            ax4.set_xlabel('Training Step', fontsize=10)
            ax4.set_ylabel('Loss', fontsize=10)
            ax4.set_title('Training Loss (Smoothed)', fontweight='bold', fontsize=11)
            ax4.grid(True, alpha=0.3)
        
        # 5. Wicket Rate
        ax5 = fig.add_subplot(gs[1, 0])
        if len(self.episode_wickets) > 0:
            window = min(100, len(self.episode_wickets))
            wicket_rate = np.convolve(self.episode_wickets,
                                     np.ones(window)/window, mode='valid')
            ax5.plot(range(window-1, len(self.episode_wickets)), wicket_rate,
                    color='darkred', linewidth=2)
            ax5.set_xlabel('Episode', fontsize=10)
            ax5.set_ylabel('Wicket Rate', fontsize=10)
            ax5.set_title('Wicket Rate (Rolling 100 Overs)', fontweight='bold', fontsize=11)
            ax5.set_ylim([0, 1])
            ax5.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax5.grid(True, alpha=0.3)
        
        # 6. Average Q-Values
        ax6 = fig.add_subplot(gs[1, 1])
        if len(self.q_values) > 1000:
            window = min(1000, len(self.q_values))
            q_smooth = np.convolve(self.q_values, np.ones(window)/window, mode='valid')
            ax6.plot(q_smooth, color='purple', linewidth=2)
            ax6.set_xlabel('Training Step', fontsize=10)
            ax6.set_ylabel('Average Q-Value', fontsize=10)
            ax6.set_title('Average Q-Value (Smoothed)', fontweight='bold', fontsize=11)
            ax6.grid(True, alpha=0.3)
        
        # 7. Strike Rate (Runs per Ball)
        ax7 = fig.add_subplot(gs[1, 2])
        if len(self.episode_runs) > 0 and len(self.episode_balls_survived) > 0:
            strike_rates = [r/max(b,1) for r, b in 
                           zip(self.episode_runs, self.episode_balls_survived)]
            window = min(50, len(strike_rates))
            sr_smooth = np.convolve(strike_rates, np.ones(window)/window, mode='valid')
            ax7.plot(range(window-1, len(strike_rates)), sr_smooth,
                    color='teal', linewidth=2)
            ax7.set_xlabel('Episode', fontsize=10)
            ax7.set_ylabel('Runs per Ball', fontsize=10)
            ax7.set_title('Strike Rate', fontweight='bold', fontsize=11)
            ax7.grid(True, alpha=0.3)
        
        # 8. Epsilon Decay
        ax8 = fig.add_subplot(gs[1, 3])
        if len(self.epsilon_history) > 0:
            sample_rate = max(1, len(self.epsilon_history) // 1000)
            sampled_eps = self.epsilon_history[::sample_rate]
            ax8.plot(range(0, len(self.epsilon_history), sample_rate), sampled_eps,
                    color='brown', linewidth=2)
            ax8.set_xlabel('Training Step', fontsize=10)
            ax8.set_ylabel('Epsilon', fontsize=10)
            ax8.set_title('Exploration Rate (Epsilon)', fontweight='bold', fontsize=11)
            ax8.set_ylim([0, 1.05])
            ax8.grid(True, alpha=0.3)
        
        # 9. Shot Distribution (Recent)
        ax9 = fig.add_subplot(gs[2, 0])
        if len(self.episode_shots) > 0:
            n_recent = min(500, len(self.episode_shots))
            recent_shots = [shot for episode in self.episode_shots[-n_recent:] 
                           for shot in episode]
            if recent_shots:
                shot_counts = pd.Series(recent_shots).value_counts()
                colors = plt.cm.Set3(np.linspace(0, 1, len(shot_counts)))
                bars = ax9.bar(range(len(shot_counts)), shot_counts.values, color=colors)
                ax9.set_xticks(range(len(shot_counts)))
                ax9.set_xticklabels(shot_counts.index, rotation=45, ha='right', fontsize=8)
                ax9.set_ylabel('Frequency', fontsize=10)
                ax9.set_title(f'Shot Distribution (Last {n_recent} Overs)', 
                             fontweight='bold', fontsize=11)
                ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. Shot Efficiency (Runs per Shot)
        ax10 = fig.add_subplot(gs[2, 1])
        shot_efficiency = {}
        for shot, stats in self.shot_outcomes.items():
            if stats['count'] > 0:
                total_deliveries = stats['count']
                avg_runs = stats['runs'] / total_deliveries if total_deliveries > 0 else 0
                shot_efficiency[shot] = avg_runs
        
        if shot_efficiency:
            sorted_shots = sorted(shot_efficiency.items(), key=lambda x: x[1], reverse=True)
            shots, efficiency = zip(*sorted_shots)
            colors = ['green' if e > 1 else 'orange' if e > 0.5 else 'red' for e in efficiency]
            ax10.barh(range(len(shots)), efficiency, color=colors, alpha=0.7)
            ax10.set_yticks(range(len(shots)))
            ax10.set_yticklabels(shots, fontsize=9)
            ax10.set_xlabel('Avg Runs per Shot', fontsize=10)
            ax10.set_title('Shot Efficiency', fontweight='bold', fontsize=11)
            ax10.grid(True, alpha=0.3, axis='x')
            ax10.invert_yaxis()
        
        # 11. Reward Distribution
        ax11 = fig.add_subplot(gs[2, 2])
        if len(self.episode_rewards) > 0:
            ax11.hist(self.episode_rewards, bins=30, color='skyblue', 
                     edgecolor='black', alpha=0.7)
            mean_reward = np.mean(self.episode_rewards)
            ax11.axvline(x=mean_reward, color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {mean_reward:.1f}')
            ax11.set_xlabel('Episode Reward', fontsize=10)
            ax11.set_ylabel('Frequency', fontsize=10)
            ax11.set_title('Reward Distribution', fontweight='bold', fontsize=11)
            ax11.legend()
            ax11.grid(True, alpha=0.3)
        
        # 12. Performance by Training Phase
        ax12 = fig.add_subplot(gs[2, 3])
        if len(self.episode_rewards) > 100:
            n_episodes = len(self.episode_rewards)
            quartile_size = n_episodes // 4
            
            quartile_rewards = [
                self.episode_rewards[:quartile_size],
                self.episode_rewards[quartile_size:2*quartile_size],
                self.episode_rewards[2*quartile_size:3*quartile_size],
                self.episode_rewards[3*quartile_size:]
            ]
            
            bp = ax12.boxplot(quartile_rewards, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                             patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax12.set_xlabel('Training Quarter', fontsize=10)
            ax12.set_ylabel('Episode Reward', fontsize=10)
            ax12.set_title('Performance by Phase', fontweight='bold', fontsize=11)
            ax12.grid(True, alpha=0.3, axis='y')
        
        # 13. Cumulative Runs Over Time
        ax13 = fig.add_subplot(gs[3, 0])
        if len(self.episode_runs) > 0:
            cumulative_runs = np.cumsum(self.episode_runs)
            ax13.plot(cumulative_runs, color='darkgreen', linewidth=2)
            ax13.set_xlabel('Episode', fontsize=10)
            ax13.set_ylabel('Cumulative Runs', fontsize=10)
            ax13.set_title('Total Runs Accumulated', fontweight='bold', fontsize=11)
            ax13.grid(True, alpha=0.3)
        
        # 14. Success Rate (Non-Wicket Rate)
        ax14 = fig.add_subplot(gs[3, 1])
        if len(self.episode_wickets) > 0:
            window = min(100, len(self.episode_wickets))
            success_rate = 1 - np.convolve(self.episode_wickets,
                                           np.ones(window)/window, mode='valid')
            ax14.plot(range(window-1, len(self.episode_wickets)), success_rate,
                     color='green', linewidth=2)
            ax14.set_xlabel('Episode', fontsize=10)
            ax14.set_ylabel('Success Rate', fontsize=10)
            ax14.set_title('Success Rate (No Wicket)', fontweight='bold', fontsize=11)
            ax14.set_ylim([0, 1])
            ax14.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='80% Target')
            ax14.legend()
            ax14.grid(True, alpha=0.3)
        
        # 15. Runs Distribution
        ax15 = fig.add_subplot(gs[3, 2])
        if len(self.episode_runs) > 0:
            ax15.hist(self.episode_runs, bins=range(0, max(self.episode_runs)+2), 
                     color='orange', edgecolor='black', alpha=0.7)
            ax15.set_xlabel('Runs per Over', fontsize=10)
            ax15.set_ylabel('Frequency', fontsize=10)
            ax15.set_title('Runs Distribution', fontweight='bold', fontsize=11)
            ax15.grid(True, alpha=0.3, axis='y')
        
        # 16. Summary Statistics
        ax16 = fig.add_subplot(gs[3, 3])
        ax16.axis('off')
        
        recent_stats = self.get_recent_stats(100)
        all_stats = self.get_recent_stats(len(self.episode_rewards))
        
        if recent_stats and all_stats:
            summary_text = f"""
TRAINING SUMMARY
{'='*35}

Total Episodes: {len(self.episode_rewards)}
Training Steps: {len(self.losses)}

Last 100 Overs:
  Avg Reward:     {recent_stats['avg_reward']:7.2f}
  Avg Runs:       {recent_stats['avg_runs']:7.2f}
  Avg Balls:      {recent_stats['avg_balls']:7.2f}/6
  Wicket Rate:    {recent_stats['wicket_rate']:7.1%}
  Success Rate:   {recent_stats['success_rate']:7.1%}
  Strike Rate:    {recent_stats['strike_rate']:7.3f}

Overall Performance:
  Total Runs:     {sum(self.episode_runs)}
  Best Over:      {max(self.episode_runs)} runs
  Wickets:        {sum(self.episode_wickets)}/{len(self.episode_wickets)}
  Success Rate:   {all_stats['success_rate']:.1%}
            """
            
            ax16.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                     verticalalignment='center', transform=ax16.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('DQN Training Progress - Cricket Over (6 Balls)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training progress plots saved to '{save_path}'")
        plt.close()

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
            reward = -5
            self.got_out = True
            done = True
            runs_this_ball = 0
        else:
            runs_this_ball = int(outcome)
            reward = runs_this_ball
            self.total_runs += runs_this_ball
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
            'shot': shot_type,
            'runs_this_ball': runs_this_ball
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
        
        # Metrics tracking
        self.metrics = TrainingMetrics()

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        else:
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
            'epsilon': self.epsilon,
            'metrics': self.metrics
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
    
    for episode in range(1, NUM_EPISODES + 1):
        observation = env.reset()
        total_reward = 0
        episode_shots = []
        
        for ball in range(BALLS_PER_OVER):
            # Agent CHOOSES shot
            action = agent.select_action(observation)
            
            # Environment responds
            next_observation, reward, done, info = env.step(action)
            
            # Store experience and learn
            agent.memory.push(observation, action, reward, next_observation, done)
            loss = agent.optimize_model()
            
            # Record step-level metrics
            if loss is not None:
                with torch.no_grad():
                    q_value = agent.policy_net(observation).mean().item()
                agent.metrics.record_step(loss=loss, q_value=q_value, epsilon=agent.epsilon)
            
            # Track shot outcome
            agent.metrics.record_shot_outcome(
                info['shot'], 
                info['runs_this_ball'], 
                info['got_out']
            )
            
            episode_shots.append(info['shot'])
            observation = next_observation
            total_reward += reward
            
            if done:
                break
        
        # Record episode metrics
        agent.metrics.record_episode(
            reward=total_reward,
            runs=info['total_runs'],
            balls=info['ball'],
            wicket=info['got_out'],
            shots=episode_shots
        )
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
        
        # Logging
        if episode % 10 == 0:
            stats = agent.metrics.get_recent_stats(10)
            
            print(f"\n{'='*80}")
            print(f"Episode {episode}/{NUM_EPISODES}")
            print(f"{'='*80}")
            print(f"Last 10 Overs:")
            print(f"  Avg Reward:     {stats['avg_reward']:7.2f}")
            print(f"  Avg Runs:       {stats['avg_runs']:7.2f}/6 balls")
            print(f"  Avg Balls:      {stats['avg_balls']:7.2f}/6")
            print(f"  Wicket Rate:    {stats['wicket_rate']:7.1%}")
            print(f"  Success Rate:   {stats['success_rate']:7.1%}")
            print(f"  Strike Rate:    {stats['strike_rate']:7.3f}")
            print(f"  Epsilon:        {agent.epsilon:7.3f}")
            print(f"{'='*80}")
        
        # Generate plots periodically
        if episode % 250 == 0:
            agent.metrics.plot_training_progress(
                save_path=f'dqn_progress_ep{episode}.png'
            )
    
    # Final statistics
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    final_stats = agent.metrics.get_recent_stats(100)
    print(f"\nFinal 100 Episodes:")
    print(f"  Avg Reward:      {final_stats['avg_reward']:.2f}")
    print(f"  Avg Runs:        {final_stats['avg_runs']:.2f}/6 balls")
    print(f"  Avg Balls:       {final_stats['avg_balls']:.2f}/6")
    print(f"  Wicket Rate:     {final_stats['wicket_rate']:.1%}")
    print(f"  Success Rate:    {final_stats['success_rate']:.1%}")
    print(f"  Strike Rate:     {final_stats['strike_rate']:.3f}")
    
    # Generate final comprehensive plots
    agent.metrics.plot_training_progress(save_path='dqn_final_training.png')
    
    # Save agent
    agent.save('cricket_agent_final.pth')
    print(f"\n✓ Agent saved to 'cricket_agent_final.pth'")
    
    return agent

# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate(agent, env, num_episodes=100):
    """Evaluate trained agent"""
    print("\n" + "="*80)
    print(f"EVALUATING AGENT - {num_episodes} Episodes")
    print("="*80)
    
    eval_rewards = []
    eval_runs = []
    eval_balls = []
    eval_wickets = []
    eval_shots = []
    
    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        episode_shots_list = []
        done = False
        
        while not done:
            action = agent.select_action(observation, training=False)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            episode_shots_list.append(info['shot'])
        
        eval_rewards.append(episode_reward)
        eval_runs.append(info['total_runs'])
        eval_balls.append(info['ball'])
        eval_wickets.append(1 if info['got_out'] else 0)
        eval_shots.extend(episode_shots_list)
    
    # Statistics
    print(f"\nEvaluation Results:")
    print(f"  Avg Reward:      {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Avg Runs:        {np.mean(eval_runs):.2f} ± {np.std(eval_runs):.2f}")
    print(f"  Avg Balls:       {np.mean(eval_balls):.2f}/6")
    print(f"  Wicket Rate:     {np.mean(eval_wickets):.1%}")
    print(f"  Success Rate:    {(1-np.mean(eval_wickets)):.1%}")
    print(f"  Total Runs:      {sum(eval_runs)}")
    print(f"  Best Over:       {max(eval_runs)} runs")
    
    # Shot distribution
    print(f"\nShot Distribution:")
    shot_counts = pd.Series(eval_shots).value_counts()
    for shot, count in shot_counts.items():
        print(f"  {shot:15s}: {count:4d} ({count/len(eval_shots)*100:5.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward distribution
    axes[0, 0].hist(eval_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(eval_rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(eval_rewards):.1f}')
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Evaluation: Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Runs distribution
    axes[0, 1].hist(eval_runs, bins=range(0, max(eval_runs)+2), 
                    color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(eval_runs), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(eval_runs):.1f}')
    axes[0, 1].set_xlabel('Runs per Over')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Evaluation: Runs Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Shot distribution
    colors = plt.cm.Set3(np.linspace(0, 1, len(shot_counts)))
    axes[1, 0].bar(range(len(shot_counts)), shot_counts.values, color=colors)
    axes[1, 0].set_xticks(range(len(shot_counts)))
    axes[1, 0].set_xticklabels(shot_counts.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Evaluation: Shot Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Performance metrics
    axes[1, 1].axis('off')
    summary_text = f"""
EVALUATION SUMMARY
{'='*30}

Episodes: {num_episodes}

Performance:
  Avg Reward:    {np.mean(eval_rewards):7.2f}
  Avg Runs:      {np.mean(eval_runs):7.2f}
  Avg Balls:     {np.mean(eval_balls):7.2f}/6
  
  Wicket Rate:   {np.mean(eval_wickets):7.1%}
  Success Rate:  {(1-np.mean(eval_wickets)):7.1%}
  Strike Rate:   {np.mean([r/max(b,1) for r,b in zip(eval_runs, eval_balls)]):7.3f}

Totals:
  Total Runs:    {sum(eval_runs)}
  Total Wickets: {sum(eval_wickets)}/{num_episodes}
  Best Over:     {max(eval_runs)} runs
  Worst Over:    {min(eval_runs)} runs
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('dqn_evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Evaluation plots saved to 'dqn_evaluation_results.png'")
    plt.close()
    
    return {
        'avg_reward': np.mean(eval_rewards),
        'avg_runs': np.mean(eval_runs),
        'avg_balls': np.mean(eval_balls),
        'wicket_rate': np.mean(eval_wickets),
        'success_rate': 1 - np.mean(eval_wickets)
    }

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train agent
    agent = train()
    
    # Evaluate agent
    env = CricketOverEnv()
    eval_results = evaluate(agent, env, num_episodes=100)
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)