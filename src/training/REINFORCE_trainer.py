import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


class Trainer:
    """
    Trainer class for REINFORCE algorithm (with optional baseline).
    Handles episode collection, Monte Carlo returns computation, and policy updates.
    
    Without baseline: Uses policy gradient -Σ log π(a_t|s_t) * G_t
    With baseline: Uses advantage A_t = G_t - V(s_t) and updates value network with MSE loss.
    
    Updates policy once per episode.
    """
    
    def __init__(self, policy_network, learning_rate=0.001, gamma=0.99, 
                 baseline=False, value_network=None, value_learning_rate=0.001, max_grad_norm=0.5):
        """
        Initialize the REINFORCE Trainer.
        
        Args:
            policy_network (PolicyNetwork): The Policy Network to train (PyTorch model)
            learning_rate (float): Learning rate for policy Adam optimizer
            gamma (float): Discount factor for future rewards
            baseline (bool): Whether to use baseline (value network) for variance reduction
            value_network (ValueNetwork): The Value Network for baseline (required if baseline=True)
            value_learning_rate (float): Learning rate for value network optimizer
            max_grad_norm (float): Max norm for gradient clipping (set to None or 0 to disable)
        """
        self.policy_network = policy_network
        self.gamma = gamma
        self.baseline = baseline
        self.max_grad_norm = max_grad_norm
        
        # Use PyTorch's built-in Adam optimizer for policy
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Initialize value network and optimizer if using baseline
        self.value_network = value_network
        if self.baseline:
            if self.value_network is None:
                raise ValueError("value_network must be provided when baseline=True")
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=value_learning_rate)
            self.value_criterion = nn.MSELoss()
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
        self.states = []  # Store states for baseline computation
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        if self.value_network is not None:
            self.value_network.to(self.device)
    
    def store_transition(self, state, action, reward):
        """
        Store a single transition during episode collection.
        Computes and stores log probability for the action taken.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
        """
        # Get action probabilities from policy network
        probs = self.policy_network.forward(state)
        
        # Create categorical distribution and compute log probability
        dist = Categorical(probs)
        action_tensor = torch.tensor(action, device=self.device)
        log_prob = dist.log_prob(action_tensor)
        
        # Store log probability, reward, and state (for baseline)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.states.append(state)
    
    def compute_returns(self, normalize=True):
        """
        Compute Monte Carlo returns (discounted cumulative rewards) for each time step.
        
        Args:
            normalize (bool): Whether to normalize returns (disabled when using baseline)
        
        Returns:
            torch.Tensor: Returns G_t for each time step in the episode
        """
        returns = []
        G = 0
        
        # Compute returns by working backwards through the episode
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns for stability (skip if using baseline, as advantage handles this)
        if normalize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def train_step(self):
        """
        Perform a single training step (one episode update).
        
        Computes policy gradient loss and updates network weights.
        When baseline=True, also computes advantage A_t = G_t - V(s_t) and updates value network.
        Should be called once at the end of each episode.
        
        Returns:
            float: Loss value for this training step (policy loss)
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Compute returns (don't normalize if using baseline)
        returns = self.compute_returns(normalize=not self.baseline)
        
        if self.baseline:
            # Compute state values V(s_t) for all states in episode
            states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
            values = self.value_network.forward(states_tensor).squeeze()
            
            # Compute advantages: A_t = G_t - V(s_t)
            advantages = returns - values.detach()  # Detach to not backprop through value network here
            
            # Normalize advantages for stability
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute policy loss: -Σ log π(a_t|s_t) * A_t
            policy_loss = []
            for log_prob, advantage in zip(self.log_probs, advantages):
                policy_loss.append(-log_prob * advantage)
            
            # Sum over all time steps in the episode
            policy_loss = torch.stack(policy_loss).sum()
            
            # Update policy network
            self.optimizer.zero_grad()
            policy_loss.backward()
            # Gradient clipping to prevent exploding gradients
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            
            # Update value network with MSE loss: (G_t - V(s_t))^2
            value_loss = self.value_criterion(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            # Gradient clipping to prevent exploding gradients
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=self.max_grad_norm)
            self.value_optimizer.step()
        else:
            # Standard REINFORCE without baseline
            # Compute policy loss: -Σ log π(a_t|s_t) * G_t
            policy_loss = []
            for log_prob, G in zip(self.log_probs, returns):
                policy_loss.append(-log_prob * G)
            
            # Sum over all time steps in the episode
            policy_loss = torch.stack(policy_loss).sum()
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()  # Clear gradients
            policy_loss.backward()      # Compute gradients
            # Gradient clipping to prevent exploding gradients
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()       # Update weights
        
        # Clear episode data
        loss_value = policy_loss.item()
        self.log_probs = []
        self.rewards = []
        self.states = []
        
        return loss_value
    
    def reset_episode(self):
        """
        Reset episode-specific data. Call this at the start of a new episode.
        """
        self.log_probs = []
        self.rewards = []
        self.states = []
