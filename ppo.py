import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# PPO Hyperparameters
LEARNING_RATE = 0.0003      # Lower LR is usually better for PPO
GAMMA = 0.99                # Discount factor
EPS_CLIP = 0.2              # PPO Clip parameter (prevents drastic policy changes)
K_EPOCHS = 4                # Update policy for K epochs per batch
UPDATE_TIMESTEP = 300       # Update policy every N timesteps
ENTROPY_COEF = 0.01         # Encourages exploration
NUM_EPISODES = 1000
BALLS_PER_OVER = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# METRICS TRACKING (FIXED)
# ==============================================================================

class TrainingMetrics:
    """Comprehensive metrics tracking for PPO training"""
    
    def __init__(self):
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_runs = []
        self.episode_balls_survived = []
        self.episode_wickets = []
        self.episode_shots = []
        
        # Step-level metrics (PPO updates are sparse, so we track lists)
        self.losses = []
        self.values = []
        
        # Per-shot statistics
        self.shot_outcomes = {shot: {'count': 0, 'runs': 0, 'wickets': 0} 
                             for shot in ACTION_MAP.values()}
    
    def record_episode(self, reward, runs, balls, wicket, shots):
        self.episode_rewards.append(reward)
        self.episode_runs.append(runs)
        self.episode_balls_survived.append(balls)
        self.episode_wickets.append(1 if wicket else 0)
        self.episode_shots.append(shots)
    
    def record_step(self, loss=None, value=None):
        if loss is not None:
            self.losses.append(loss)
        if value is not None:
            self.values.append(value)
    
    def record_shot_outcome(self, shot, runs, wicket):
        self.shot_outcomes[shot]['count'] += 1
        if wicket:
            self.shot_outcomes[shot]['wickets'] += 1
        else:
            self.shot_outcomes[shot]['runs'] += runs
    
    def get_recent_stats(self, window=100):
        """Get statistics for recent episodes"""
        if len(self.episode_rewards) < window:
            window = len(self.episode_rewards)
        if window == 0: return {}
        
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
    
    def plot_training_progress(self, save_path='ppo_cricket_training.png'):
        """Generate comprehensive training progress plots"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Helper for smoothing
        def smooth(data, win_size=None):
            if not data: return []
            series = pd.Series(data)
            if win_size is None:
                win_size = max(1, len(data) // 10)
            return series.rolling(window=win_size, min_periods=1).mean()

        # 1. Episode Rewards
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.episode_rewards) > 0:
            ax1.plot(self.episode_rewards, alpha=0.2, color='blue', linewidth=0.5)
            ax1.plot(smooth(self.episode_rewards), color='blue', linewidth=2)
            ax1.set_title('Episode Reward', fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # 2. Runs per Over
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.episode_runs) > 0:
            ax2.plot(self.episode_runs, alpha=0.2, color='green', linewidth=0.5)
            ax2.plot(smooth(self.episode_runs), color='green', linewidth=2)
            ax2.set_title('Runs per Over', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Balls Survived
        ax3 = fig.add_subplot(gs[0, 2])
        if len(self.episode_balls_survived) > 0:
            ax3.plot(smooth(self.episode_balls_survived), color='orange', linewidth=2)
            ax3.set_ylim(0, 7)
            ax3.axhline(y=6, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Avg Balls Survived', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Training Loss (Fixed for PPO sparsity)
        ax4 = fig.add_subplot(gs[0, 3])
        if len(self.losses) > 0:
            ax4.plot(self.losses, alpha=0.3, color='red', linewidth=0.5)
            ax4.plot(smooth(self.losses), color='darkred', linewidth=2)
            ax4.set_title(f'PPO Loss (Updates: {len(self.losses)})', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Loss Data Yet', ha='center', va='center')
        
        # 5. Wicket Rate
        ax5 = fig.add_subplot(gs[1, 0])
        if len(self.episode_wickets) > 0:
            ax5.plot(smooth(self.episode_wickets), color='darkred', linewidth=2)
            ax5.set_title('Wicket Rate', fontweight='bold')
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
        
        # 6. Critic Value
        ax6 = fig.add_subplot(gs[1, 1])
        if len(self.values) > 0:
            ax6.plot(smooth(self.values), color='purple', linewidth=2)
            ax6.set_title('Critic Value Estimate', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
        # 7. Strike Rate
        ax7 = fig.add_subplot(gs[1, 2])
        if len(self.episode_runs) > 0:
            srs = [r/max(b,1) for r, b in zip(self.episode_runs, self.episode_balls_survived)]
            ax7.plot(smooth(srs), color='teal', linewidth=2)
            ax7.set_title('Strike Rate (Runs/Ball)', fontweight='bold')
            ax7.grid(True, alpha=0.3)

        # 8. Success Rate (No Wicket) -- ADDED
        ax8 = fig.add_subplot(gs[1, 3])
        if len(self.episode_wickets) > 0:
            success_rate = 1 - smooth(self.episode_wickets)
            ax8.plot(success_rate, color='green', linewidth=2)
            ax8.set_title('Success Rate (Not Out)', fontweight='bold')
            ax8.set_ylim(0, 1)
            ax8.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
            ax8.grid(True, alpha=0.3)

        # 9. Shot Distribution
        ax9 = fig.add_subplot(gs[2, 0])
        if len(self.episode_shots) > 0:
            n_recent = min(500, len(self.episode_shots))
            recent_shots = [shot for episode in self.episode_shots[-n_recent:] for shot in episode]
            if recent_shots:
                shot_counts = pd.Series(recent_shots).value_counts()
                ax9.bar(range(len(shot_counts)), shot_counts.values, alpha=0.7)
                ax9.set_xticks(range(len(shot_counts)))
                ax9.set_xticklabels(shot_counts.index, rotation=45, ha='right', fontsize=8)
                ax9.set_title(f'Shot Distribution', fontweight='bold')

        # 10. Efficiency
        ax10 = fig.add_subplot(gs[2, 1])
        shot_efficiency = {}
        for shot, stats in self.shot_outcomes.items():
            if stats['count'] > 0:
                shot_efficiency[shot] = stats['runs'] / stats['count']
        if shot_efficiency:
            sorted_shots = sorted(shot_efficiency.items(), key=lambda x: x[1], reverse=True)
            shots, efficiency = zip(*sorted_shots)
            ax10.barh(range(len(shots)), efficiency, color='green', alpha=0.6)
            ax10.set_yticks(range(len(shots)))
            ax10.set_yticklabels(shots, fontsize=9)
            ax10.invert_yaxis()
            ax10.set_title('Avg Runs per Shot', fontweight='bold')

        plt.suptitle('PPO Training Progress', fontsize=16, fontweight='bold', y=0.99)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training plots saved to {save_path}")
        plt.close()

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

class CricketOverEnv:
    def __init__(self):
        self.predictor = joblib.load(MODEL_PATH)
        self.rl_encoder = joblib.load(ENCODER_PATH)
        self.df = pd.read_csv(DATA_PATH)
        self.state_cols = [
            'Bowler Type', 'Ball Length', 'Ball Line', 
            'Speed (km/h)', 'Field Placement', 'Angle', 'Bounce (cm)'
        ]
        self.state_df = self.df[self.state_cols].copy()
        self.ball_count = 0
        self.total_runs = 0
        self.got_out = False
        self.current_state = None

    def _get_observation(self, state_dict):
        state_df = pd.DataFrame([state_dict])[self.state_cols]
        obs_array = self.rl_encoder.transform(state_df).flatten()
        return torch.tensor(obs_array, dtype=torch.float32, device=DEVICE)

    def _sample_delivery(self):
        raw_sample = self.state_df.sample(1).iloc[0]
        return {col: raw_sample[col] for col in self.state_cols}

    def reset(self):
        self.ball_count = 0
        self.total_runs = 0
        self.got_out = False
        self.current_state = self._sample_delivery()
        return self._get_observation(self.current_state)

    def step(self, action_int):
        shot_type = ACTION_MAP[action_int]
        pred_input = self.current_state.copy()
        pred_input['Shot Type'] = shot_type
        pred_input_df = pd.DataFrame([pred_input])
        
        probs = self.predictor.predict_proba(pred_input_df).flatten()
        outcome = np.random.choice(self.predictor.classes_, p=probs)
        
        if outcome == 'Wicket':
            reward = -5.0
            self.got_out = True
            done = True
            runs_this_ball = 0
        else:
            runs_this_ball = int(outcome)
            reward = float(runs_this_ball)
            self.total_runs += runs_this_ball
            self.ball_count += 1
            done = (self.ball_count >= BALLS_PER_OVER)
        
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
# PPO COMPONENTS
# ==============================================================================

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )
    
    def act(self, state):
        features = self.feature_layer(state)
        action_probs = self.actor(features)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        features = self.feature_layer(state)
        action_probs = self.actor(features)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.policy = ActorCritic(N_OBSERVATIONS, N_ACTIONS).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(N_OBSERVATIONS, N_ACTIONS).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = RolloutBuffer()
        self.metrics = TrainingMetrics()
        
    def select_action(self, state, training=True):
        if training:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            
            return action.item()
        else:
            with torch.no_grad():
                features = self.policy.feature_layer(state)
                action_probs = self.policy.actor(features)
                return torch.argmax(action_probs).item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        # Normalize rewards
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(DEVICE)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(DEVICE)

        # Optimize policy for K epochs
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, rewards) - ENTROPY_COEF * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            # Record metrics
            self.metrics.record_step(loss=loss.mean().item(), value=state_values.mean().item())

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path='ppo_agent.pth'):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)

    def load(self, path='ppo_agent.pth'):
        try:
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy_old.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            print(f"✓ PPO Agent loaded from {path}")
        except Exception as e:
            print(f"Error loading PPO model: {e}")

# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================

def train():
    print("="*80)
    print("CRICKET SHOT SELECTION - PPO AGENT")
    print("="*80)
    
    env = CricketOverEnv()
    agent = PPOAgent(env)
    
    time_step = 0
    update_counter = 0
    
    for episode in range(1, NUM_EPISODES + 1):
        observation = env.reset()
        total_reward = 0
        episode_shots = []
        
        for ball in range(BALLS_PER_OVER):
            time_step += 1
            
            action = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            
            # Save experience
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            # Record details
            agent.metrics.record_shot_outcome(info['shot'], info['runs_this_ball'], info['got_out'])
            episode_shots.append(info['shot'])
            
            total_reward += reward
            observation = next_observation
            
            # PPO Update check
            if time_step % UPDATE_TIMESTEP == 0:
                agent.update()
                update_counter += 1
            
            if done:
                break
        
        agent.metrics.record_episode(total_reward, info['total_runs'], info['ball'], info['got_out'], episode_shots)
        
        # Logging
        if episode % 20 == 0:
            stats = agent.metrics.get_recent_stats(20)
            print(f"Ep {episode:4d} | R: {stats['avg_runs']:5.2f} | "
                  f"Wkt: {stats['wicket_rate']:4.0%} | "
                  f"Success: {stats['success_rate']:4.0%} | Updates: {update_counter}")
        
        # Plotting
        if episode % 100 == 0:
            agent.metrics.plot_training_progress(save_path=f'ppo_progress.png')
            
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    
    agent.metrics.plot_training_progress(save_path='ppo_final_training.png')
    agent.save('ppo_agent_final.pth')
    print(f"✓ Agent saved to 'ppo_agent_final.pth'")
    
    return agent

def evaluate(agent, env, num_episodes=100):
    print("\n" + "="*80)
    print(f"EVALUATING AGENT - {num_episodes} Episodes")
    print("="*80)
    
    eval_rewards = []
    eval_runs = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_rew = 0
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, info = env.step(action)
            ep_rew += reward
        eval_rewards.append(ep_rew)
        eval_runs.append(info['total_runs'])
        
    print(f"Avg Reward: {np.mean(eval_rewards):.2f}")
    print(f"Avg Runs:   {np.mean(eval_runs):.2f}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        trained_agent = train()
        evaluate(trained_agent, CricketOverEnv())
    else:
        print(f"ERROR: Missing data files!")
        print(f"Ensure '{MODEL_PATH}' and '{ENCODER_PATH}' are in this directory.")