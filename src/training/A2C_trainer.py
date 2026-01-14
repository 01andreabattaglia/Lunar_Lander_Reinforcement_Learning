import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


class Trainer:
    """
    Trainer class for N-step Advantage Actor-Critic (A2C) with parallel environments.
    
    Key features:
    - N parallel environments for diverse experience collection
    - n-step returns for balanced bias/variance tradeoff
    - Batch updates at specified intervals (train_every)
    
    n-step return calculation:
    G_t^(n) = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{n-1}*r_{t+n-1} + γ^n*V(s_{t+n})
    
    Advantage:
    A_t = G_t^(n) - V(s_t)
    """
    
    def __init__(self, policy_network, value_network, 
                 actor_learning_rate=0.001, critic_learning_rate=0.001, gamma=0.99,
                 n_steps=5, train_every=4, n_envs=8, max_grad_norm=0.5):
        """
        Initialize the N-step A2C Trainer.
        
        Args:
            policy_network (PolicyNetwork): The Policy Network (actor) to train
            value_network (ValueNetwork): The Value Network (critic) to train
            actor_learning_rate (float): Learning rate α for actor (policy) optimizer
            critic_learning_rate (float): Learning rate β for critic (value) optimizer
            gamma (float): Discount factor for future rewards
            n_steps (int): Number of steps for n-step returns (default: 5)
            train_every (int): Update networks every N steps (default: 4)
            n_envs (int): Number of parallel environments (default: 8)
            max_grad_norm (float): Max norm for gradient clipping (set to None or 0 to disable)
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.gamma = gamma
        self.n_steps = n_steps
        self.train_every = train_every
        self.n_envs = n_envs
        self.max_grad_norm = max_grad_norm
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        # Actor optimizer (policy network)
        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_learning_rate)
        
        # Critic optimizer (value network)
        self.critic_optimizer = optim.Adam(self.value_network.parameters(), lr=critic_learning_rate)
        
        # Rollout buffer for n-step returns
        self.reset_rollout_buffer()
        
        # Episode tracking (per environment)
        self.episode_rewards = [[] for _ in range(n_envs)]
        self.episode_lengths = [0 for _ in range(n_envs)]
        self.completed_episodes = []
        
        # Training statistics
        self.train_actor_losses = []
        self.train_critic_losses = []
        self.global_step = 0
    
    def reset_rollout_buffer(self):
        """Reset the rollout buffer for collecting n-step transitions."""
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.values_buffer = []
        self.log_probs_buffer = []
        self.step_count = 0
    
    def select_actions(self, states):
        """
        Select actions for a batch of states from parallel environments.
        
        Args:
            states (np.ndarray): Batch of states, shape (n_envs, state_dim)
            
        Returns:
            tuple: (actions, log_probs, values, entropy)
        """
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        
        # Get action probabilities from policy network
        probs = self.policy_network.forward(states_tensor)
        
        # Create categorical distribution
        dist = Categorical(probs)
        
        # Sample actions
        actions = dist.sample()
        
        # Compute log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Get state values from critic
        values = self.value_network.forward(states_tensor).squeeze(-1)
        
        return (actions.cpu().numpy(), 
                log_probs.detach(), 
                values.detach(), 
                entropy.mean().item())
    
    def store_transition(self, states, actions, rewards, dones, values, log_probs):
        """
        Store a transition from all parallel environments.
        
        Args:
            states (np.ndarray): Batch of states, shape (n_envs, state_dim)
            actions (np.ndarray): Batch of actions, shape (n_envs,)
            rewards (np.ndarray): Batch of rewards, shape (n_envs,)
            dones (np.ndarray): Batch of done flags, shape (n_envs,)
            values (torch.Tensor): Batch of state values, shape (n_envs,)
            log_probs (torch.Tensor): Batch of log probabilities, shape (n_envs,)
        """
        self.states_buffer.append(states.copy())
        self.actions_buffer.append(actions.copy())
        self.rewards_buffer.append(rewards.copy())
        self.dones_buffer.append(dones.copy())
        self.values_buffer.append(values)
        self.log_probs_buffer.append(log_probs)
        
        # Track episode rewards
        for i in range(self.n_envs):
            self.episode_rewards[i].append(rewards[i])
            
            if dones[i]:
                # Episode completed for environment i
                # Increment length here to count this final step
                self.episode_lengths[i] += 1
                
                total_reward = sum(self.episode_rewards[i])
                length = self.episode_lengths[i]
                self.completed_episodes.append({
                    'env_id': i,
                    'total_reward': total_reward,
                    'length': length
                })
                # Reset tracking for this environment
                self.episode_rewards[i] = []
                self.episode_lengths[i] = 0
            else:
                # Not done yet, increment length for next iteration
                self.episode_lengths[i] += 1
        
        self.step_count += 1
        self.global_step += 1
    
    def compute_n_step_returns(self, next_values, next_dones):
        """
        Compute n-step returns and advantages for the rollout buffer.
        
        G_t^(n) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n*V(s_{t+n})
        A_t = G_t^(n) - V(s_t)
        
        Args:
            next_values (torch.Tensor): Values of states after the last step, shape (n_envs,)
            next_dones (np.ndarray): Done flags after the last step, shape (n_envs,)
            
        Returns:
            tuple: (returns, advantages) each of shape (n_steps, n_envs)
        """
        n_steps = len(self.rewards_buffer)
        
        # Convert buffers to tensors
        rewards = torch.tensor(np.array(self.rewards_buffer), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(self.dones_buffer), dtype=torch.float32, device=self.device)
        values = torch.stack(self.values_buffer)
        
        # Initialize returns with bootstrap value
        returns = torch.zeros((n_steps, self.n_envs), device=self.device)
        
        # Bootstrap from the last value (masked by done)
        next_dones_tensor = torch.tensor(next_dones, dtype=torch.float32, device=self.device)
        running_return = next_values * (1 - next_dones_tensor)
        
        # Compute returns backwards
        for t in reversed(range(n_steps)):
            # If done at step t, the return is just the reward (no bootstrap)
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        
        # Compute advantages
        advantages = returns - values
        
        return returns, advantages
    
    def train_step(self, next_states, next_dones):
        """
        Perform a training update using the collected rollout buffer.
        
        Only updates at steps that are multiples of train_every.
        
        Args:
            next_states (np.ndarray): States after the last collected step
            next_dones (np.ndarray): Done flags after the last collected step
            
        Returns:
            dict: Training statistics or None if not updating this step
        """
        # Only update when we have enough steps
        if self.step_count < self.n_steps:
            return None
        
        # Get bootstrap values for n-step returns
        with torch.no_grad():
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            next_values = self.value_network.forward(next_states_tensor).squeeze(-1)
        
        # Compute n-step returns and advantages
        returns, advantages = self.compute_n_step_returns(next_values, next_dones)
        
        # Select only transitions at multiples of train_every
        # For n_steps=5 and train_every=4: we select step 0 and 4
        update_indices = list(range(0, self.step_count, self.train_every))
        
        if len(update_indices) == 0:
            self.reset_rollout_buffer()
            return None
        
        # Gather data for selected steps
        batch_states = []
        batch_actions = []
        batch_returns = []
        batch_advantages = []
        
        for t in update_indices:
            batch_states.append(self.states_buffer[t])
            batch_actions.append(self.actions_buffer[t])
            batch_returns.append(returns[t])
            batch_advantages.append(advantages[t])
        
        # Stack and flatten: (num_updates, n_envs, ...) -> (batch_size, ...)
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float32, device=self.device)
        batch_states = batch_states.view(-1, batch_states.shape[-1])  # (batch, state_dim)
        
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.long, device=self.device)
        batch_actions = batch_actions.view(-1)  # (batch,)
        
        batch_returns = torch.stack(batch_returns).view(-1)  # (batch,)
        batch_advantages = torch.stack(batch_advantages).view(-1)  # (batch,)
        
        # Normalize advantages (reduces variance)
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        
        # =====================
        # Actor (Policy) Update
        # =====================
        probs = self.policy_network.forward(batch_states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(batch_actions)
        
        # Actor loss: -advantage * log_prob
        actor_loss = -(batch_advantages.detach() * log_probs).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping to prevent exploding gradients
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.max_grad_norm)
        self.actor_optimizer.step()
        
        # =====================
        # Critic (Value) Update
        # =====================
        values = self.value_network.forward(batch_states).squeeze(-1)
        critic_loss = nn.functional.mse_loss(values, batch_returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping to prevent exploding gradients
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Track statistics
        self.train_actor_losses.append(actor_loss.item())
        self.train_critic_losses.append(critic_loss.item())
        
        # Reset buffer for next rollout
        self.reset_rollout_buffer()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'batch_size': len(batch_actions),
            'num_updates': len(update_indices)
        }
    
    def get_completed_episodes(self):
        """
        Get and clear the list of completed episodes since last call.
        
        Returns:
            list: List of dicts with 'env_id', 'total_reward', 'length'
        """
        episodes = self.completed_episodes.copy()
        self.completed_episodes = []
        return episodes
    
    def get_episode_stats(self):
        """
        Get aggregate training statistics.
        
        Returns:
            dict: Training statistics
        """
        stats = {
            'avg_actor_loss': np.mean(self.train_actor_losses[-100:]) if self.train_actor_losses else 0.0,
            'avg_critic_loss': np.mean(self.train_critic_losses[-100:]) if self.train_critic_losses else 0.0,
            'global_step': self.global_step,
            'total_updates': len(self.train_actor_losses)
        }
        return stats
    
    def reset_episode(self):
        """Reset episode tracking for all environments."""
        self.episode_rewards = [[] for _ in range(self.n_envs)]
        self.episode_lengths = [0 for _ in range(self.n_envs)]
        self.completed_episodes = []
