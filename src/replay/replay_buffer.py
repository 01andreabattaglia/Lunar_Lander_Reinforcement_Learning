import numpy as np
from collections import deque


class ReplayBuffer:
    """
    A simple replay buffer to store experience tuples (transitions).
    
    Each transition contains:
    - obs: Current observation (state) - shape (8,)
    - action: Action taken - integer 0..3
    - reward: Reward received - float
    - next_obs: Next observation (state) - shape (8,)
    - done: Episode termination flag - boolean
    """
    
    def __init__(self, max_size=10000):
        """
        Initialize the replay buffer.
        
        Args:
            max_size (int): Maximum number of transitions to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, obs, action, reward, next_obs, done):
        """
        Add a transition to the replay buffer.
        
        Args:
            obs (np.ndarray): Current observation - shape (8,)
            action (int): Action taken (0-3)
            reward (float): Reward received
            next_obs (np.ndarray): Next observation - shape (8,)
            done (bool): Whether episode ended (terminated OR truncated)
        """
        self.buffer.append({
            'obs': np.array(obs),
            'action': action,
            'reward': reward,
            'next_obs': np.array(next_obs),
            'done': done
        })
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
        
        Returns:
            dict: Dictionary containing batches of obs, action, reward, next_obs, done
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'obs': np.array([b['obs'] for b in batch]),
            'action': np.array([b['action'] for b in batch]),
            'reward': np.array([b['reward'] for b in batch]),
            'next_obs': np.array([b['next_obs'] for b in batch]),
            'done': np.array([b['done'] for b in batch])
        }
    
    def __len__(self):
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the replay buffer."""
        self.buffer.clear()
