import gymnasium as gym


def make_env(env_id="LunarLander-v3", render_mode="human"):
    """
    Factory function to create and return a Gymnasium environment.
    
    Args:
        env_id (str): The environment ID (default: "LunarLander-v3")
        render_mode (str): The rendering mode ("rgb_array", "human", None)
    
    Returns:
        gym.Env: The created environment
    """
    env = gym.make(env_id, render_mode=render_mode, continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    return env


def make_vec_env(env_id="LunarLander-v3", n_envs=8, render_mode=None):
    """
    Factory function to create vectorized parallel environments.
    
    Uses gymnasium's SyncVectorEnv for parallel environment execution.
    
    Args:
        env_id (str): The environment ID (default: "LunarLander-v3")
        n_envs (int): Number of parallel environments (default: 8)
        render_mode (str): The rendering mode (default: None for training)
    
    Returns:
        gym.vector.VectorEnv: Vectorized environment with n_envs parallel instances
    """
    def _make_env():
        return gym.make(env_id, render_mode=render_mode, continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    vec_env = gym.vector.SyncVectorEnv([_make_env for _ in range(n_envs)])
    return vec_env
