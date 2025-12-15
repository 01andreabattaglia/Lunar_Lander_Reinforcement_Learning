import random


class RandomAgent:
    """
    A simple agent that takes random actions in the LunarLander environment.
    
    Actions:
    - 0 = no engine
    - 1 = left engine
    - 2 = main engine
    - 3 = right engine
    """
    
    def __init__(self):
        """Initialize the RandomAgent."""
        self.action_space_size = 4
    
    def act(self, obs):
        """
        Selects a random valid action based on the observation.
        
        Args:
            obs: The observation from the environment (not used for random agent)
        
        Returns:
            int: A random action in the range [0, 3]
        """
        return random.randint(0, self.action_space_size - 1)
