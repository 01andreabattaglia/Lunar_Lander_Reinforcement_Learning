import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy


class Trainer:
    """
    Trainer class that handles experience collection, batch sampling, and training steps.
    Uses PyTorch's built-in optimizer and loss functions.
    
    Uses two networks:
    - q_online: The network being trained (updated with optimizer)
    - q_target: A copy used for computing TD targets (updated periodically)
    """
    
    def __init__(self, q_network, learning_rate=0.001, replay_buffer=None, gamma=0.99, target_update_freq=1000):
        """
        Initialize the Trainer with dual Q-Networks.
        
        Args:
            q_network (QNetwork): The Q-Network to train (PyTorch model)
            learning_rate (float): Learning rate for Adam optimizer
            replay_buffer (ReplayBuffer): The replay buffer for sampling batches
            gamma (float): Discount factor for future rewards
            target_update_freq (int): How many steps between target network updates
        """
        # Online network (being trained)
        self.q_online = q_network
        
        # Target network (copy of online, updated periodically)
        self.q_target = deepcopy(q_network)
        
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.step_count = 0  # Track number of training steps
        
        # Use PyTorch's built-in Adam optimizer (only for online network)
        self.optimizer = optim.Adam(self.q_online.parameters(), lr=learning_rate)
        
        # Use PyTorch's built-in MSE loss
        self.loss_fn = nn.MSELoss()
        
        self.device = torch.device("cuda")  # Can be changed to "cuda" if GPU available
        self.q_online.to(self.device)
        self.q_target.to(self.device)
    
    def train_step(self, batch_size=32):
        """
        Perform a single training step.
        
        Samples a batch from replay buffer and updates Q-Network weights.
        Uses q_target for TD target computation.
        Periodically copies online network to target network.
        
        Args:
            batch_size (int): Size of the batch to sample
        
        Returns:
            float: Loss value for this training step
        """
        # Check if we have enough experience
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        obs = torch.tensor(batch['obs'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['action'], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch['reward'], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['done'], dtype=torch.float32, device=self.device)
        
        # Compute Q-values for current states using ONLINE network
        q_values = self.q_online(obs)  # shape: (batch_size, 4)
        
        # Compute Q-values for next states using TARGET network
        with torch.no_grad():
            q_next = self.q_target(next_obs)  # shape: (batch_size, 4)
            max_q_next = torch.max(q_next, dim=1)[0]  # shape: (batch_size,)
        
        # Compute target Q-values using Bellman equation
        # Q_target(s,a) = r + gamma * max_Q_target(s', a') * (1 - done)
        targets = q_values.clone().detach()
        for i in range(batch_size):
            targets[i, actions[i]] = rewards[i] + self.gamma * max_q_next[i] * (1 - dones[i])
        
        # Compute loss
        loss = self.loss_fn(q_values, targets)
        
        # Backpropagation and optimization
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()              # Compute gradients
        self.optimizer.step()        # Update weights
        
        # Increment step counter and update target network if needed
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss.item()
    
    def _update_target_network(self):
        """
        Copy weights from online network to target network.
        This is done periodically to stabilize training.
        """
        self.q_target.load_state_dict(self.q_online.state_dict())
        print(f"Target network updated at step: {self.step_count}")
