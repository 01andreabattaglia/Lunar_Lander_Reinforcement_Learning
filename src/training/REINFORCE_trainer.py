import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class Trainer:
    """
    Trainer class for REINFORCE algorithm.
    Handles episode collection, Monte Carlo returns computation, and policy updates.
    
    Uses policy gradient: -Σ log π(a_t|s_t) * G_t
    Updates policy once per episode.
    """
    
    def __init__(self, policy_network, learning_rate=0.001, gamma=0.99):
        """
        Initialize the REINFORCE Trainer.
        
        Args:
            policy_network (PolicyNetwork): The Policy Network to train (PyTorch model)
            learning_rate (float): Learning rate for Adam optimizer
            gamma (float): Discount factor for future rewards
        """
        self.policy_network = policy_network
        self.gamma = gamma
        
        # Use PyTorch's built-in Adam optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
    
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
        
        # Store log probability and reward
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def compute_returns(self):
        """
        Compute Monte Carlo returns (discounted cumulative rewards) for each time step.
        
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
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def train_step(self):
        """
        Perform a single training step (one episode update).
        
        Computes policy gradient loss and updates network weights.
        Should be called once at the end of each episode.
        
        Returns:
            float: Loss value for this training step
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Compute returns
        returns = self.compute_returns()
        
        # Compute policy loss: -Σ log π(a_t|s_t) * G_t
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Sum over all time steps in the episode
        policy_loss = torch.stack(policy_loss).sum()
        
        # Backpropagation and optimization
        self.optimizer.zero_grad()  # Clear gradients
        policy_loss.backward()      # Compute gradients
        self.optimizer.step()       # Update weights
        
        # Clear episode data
        loss_value = policy_loss.item()
        self.log_probs = []
        self.rewards = []
        
        return loss_value
    
    def reset_episode(self):
        """
        Reset episode-specific data. Call this at the start of a new episode.
        """
        self.log_probs = []
        self.rewards = []
