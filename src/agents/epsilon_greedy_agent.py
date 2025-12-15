import numpy as np
import torch


class EpsilonGreedyAgent:
    """
    An epsilon-greedy agent that uses a Q-Network to select actions.
    
    With probability epsilon: select a random action (exploration)
    With probability 1-epsilon: select the best action according to Q-values (exploitation)
    """
    
    def __init__(self, action_space, q_network, epsilon=0.2):
        """
        Initialize the EpsilonGreedyAgent.
        
        Args:
            action_space (int): Number of possible actions (e.g., 4 for LunarLander)
            q_network (QNetwork): The Q-Network to compute Q-values
            epsilon (float): Exploration rate, probability of taking a random action (default: 0.1)
        """
        self.action_space = action_space
        self.q_network = q_network
        self.epsilon = epsilon
    
    def act(self, obs):
        """
        Select an action using the epsilon-greedy strategy.
        
        Args:
            obs (np.ndarray): Current observation/state with shape (8,)
        
        Returns:
            int: Action to take (0-3 for LunarLander)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_space)
        else:
            # Exploitation: select action with highest Q-value
            q_values = self.q_network.predict(obs)
            
            # Convert tensor to numpy if needed
            if isinstance(q_values, torch.Tensor):
                q_values = q_values.cpu().detach().numpy()
            
            action = int(np.argmax(q_values))
        
        return action
    
    def set_epsilon(self, epsilon):
        """
        Update the exploration rate epsilon.
        
        Args:
            epsilon (float): New exploration rate
        """
        self.epsilon = epsilon
