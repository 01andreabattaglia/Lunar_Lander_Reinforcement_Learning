import numpy as np
import torch
from torch.distributions import Categorical


class REINFORCEAgent:
    """
    A REINFORCE agent that uses a Policy Network to select actions.
    
    Samples actions from the policy's probability distribution.
    """
    
    def __init__(self, action_space, policy_network):
        """
        Initialize the REINFORCEAgent.
        
        Args:
            action_space (int): Number of possible actions (e.g., 4 for LunarLander)
            policy_network (PolicyNetwork): The Policy Network to compute action probabilities
        """
        self.action_space = action_space
        self.policy_network = policy_network
    
    def act(self, obs):
        """
        Select an action by sampling from the policy distribution.
        
        Args:
            obs (np.ndarray): Current observation/state with shape (8,)
        
        Returns:
            int: Action to take (0-3 for LunarLander)
        """
        # Get action probabilities from policy network
        probs = self.policy_network.predict(obs)
        
        # Create categorical distribution
        dist = Categorical(probs)
        
        # Sample action
        action = dist.sample()
        
        return action.item()
