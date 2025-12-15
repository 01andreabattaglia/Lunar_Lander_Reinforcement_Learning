import gymnasium as gym


def make_env(env_id="LunarLander-v3", render_mode="rgb_array"):
    """
    Factory function to create and return a Gymnasium environment.
    
    Args:
        env_id (str): The environment ID (default: "LunarLander-v3")
        render_mode (str): The rendering mode ("rgb_array", "human", None)
    
    Returns:
        gym.Env: The created environment
    """
    env = gym.make(env_id, render_mode=render_mode)
    return env
