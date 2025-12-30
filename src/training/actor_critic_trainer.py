import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


class Trainer:
    """
    Trainer class for 1-step Actor-Critic algorithm.
    Updates both actor (policy) and critic (value) networks at every step using TD error.
    
    At each step t:
    1. Actor samples a_t ~ π_θ(·|s_t)
    2. Environment returns r_t, s_{t+1}
    3. Critic computes TD error: δ_t = r_t + γ * V_w(s_{t+1}) - V_w(s_t)
    4. Actor update: θ ← θ + α * δ_t * ∇_θ log π_θ(a_t|s_t)
    5. Critic update: w ← w + β * δ_t * ∇_w V_w(s_t)
    
    Uses the same policy and value networks as REINFORCE.
    """
    
    def __init__(self, policy_network, value_network, 
                 actor_learning_rate=0.001, critic_learning_rate=0.001, gamma=0.99):
        """
        Initialize the Actor-Critic Trainer.
        
        Args:
            policy_network (PolicyNetwork): The Policy Network (actor) to train
            value_network (ValueNetwork): The Value Network (critic) to train
            actor_learning_rate (float): Learning rate α for actor (policy) optimizer
            critic_learning_rate (float): Learning rate β for critic (value) optimizer
            gamma (float): Discount factor for future rewards
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.gamma = gamma
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        # Actor optimizer (policy network)
        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_learning_rate)
        
        # Critic optimizer (value network)
        self.critic_optimizer = optim.Adam(self.value_network.parameters(), lr=critic_learning_rate)
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_actor_losses = []
        self.episode_critic_losses = []
    
    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single 1-step Actor-Critic update.
        
        Updates both actor and critic networks using the TD error at this step.
        
        Args:
            state (np.ndarray): Current state s_t
            action (int): Action taken a_t
            reward (float): Reward received r_t
            next_state (np.ndarray): Next state s_{t+1}
            done (bool): Whether the episode has ended
            
        Returns:
            tuple: (actor_loss, critic_loss, td_error) for logging
        """
        # Convert states to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        
        # =====================
        # Compute TD Error δ_t
        # =====================
        
        # V(s_t) - current state value
        value_current = self.value_network.forward(state_tensor).squeeze()
        
        # V(s_{t+1}) - next state value (0 if terminal)
        with torch.no_grad():
            if done:
                value_next = torch.tensor(0.0, device=self.device)
            else:
                value_next = self.value_network.forward(next_state_tensor).squeeze()
        
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        td_target = reward + self.gamma * value_next
        td_error = td_target - value_current.detach()  # Detach for actor update
        
        # =====================
        # Actor (Policy) Update
        # =====================
        # θ ← θ + α * δ_t * ∇_θ log π_θ(a_t|s_t)
        # In PyTorch, we minimize -δ_t * log π_θ(a_t|s_t)
        
        # Get action probabilities from policy network
        probs = self.policy_network.forward(state_tensor)
        
        # Create categorical distribution and compute log probability
        dist = Categorical(probs)
        action_tensor = torch.tensor(action, device=self.device)
        log_prob = dist.log_prob(action_tensor)
        
        # Actor loss: -δ_t * log π_θ(a_t|s_t)
        actor_loss = -td_error * log_prob
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # =====================
        # Critic (Value) Update
        # =====================
        # w ← w + β * δ_t * ∇_w V_w(s_t)
        # Equivalently, minimize (r_t + γ * V(s_{t+1}) - V(s_t))^2
        
        # Recompute value for fresh gradients
        value_current = self.value_network.forward(state_tensor).squeeze()
        
        # Critic loss: TD error squared (MSE)
        td_target_tensor = torch.tensor(td_target, dtype=torch.float32, device=self.device)
        critic_loss = nn.functional.mse_loss(value_current, td_target_tensor)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Track losses for episode
        self.episode_rewards.append(reward)
        self.episode_actor_losses.append(actor_loss.item())
        self.episode_critic_losses.append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item(), td_error.item()
    
    def get_episode_stats(self):
        """
        Get statistics for the current episode.
        
        Returns:
            dict: Dictionary containing episode statistics
        """
        stats = {
            'total_reward': sum(self.episode_rewards),
            'num_steps': len(self.episode_rewards),
            'avg_actor_loss': np.mean(self.episode_actor_losses) if self.episode_actor_losses else 0.0,
            'avg_critic_loss': np.mean(self.episode_critic_losses) if self.episode_critic_losses else 0.0,
        }
        return stats
    
    def reset_episode(self):
        """
        Reset episode-specific data. Call this at the start of a new episode.
        """
        self.episode_rewards = []
        self.episode_actor_losses = []
        self.episode_critic_losses = []
    
    def store_transition(self, state, action, reward):
        """
        Compatibility method for agents that store transitions.
        In 1-step Actor-Critic, we update immediately, so this just tracks reward.
        
        Args:
            state (np.ndarray): Current state (not used, kept for API compatibility)
            action (int): Action taken (not used, kept for API compatibility)
            reward (float): Reward received
        """
        self.episode_rewards.append(reward)
