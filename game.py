import gymnasium as gym
from gymnasium.utils.play import play
import pygame  # we use pygame keycodes

# Global variable to accumulate reward per episode
total_reward = 0.0


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    """
    Called at every environment step by play().
    Accumulates rewards and prints them.
    """
    global total_reward
    total_reward += rew

    # When the episode ends, print the final total reward and reset it
    if terminated or truncated:
        print(f"\n=== EPISODE FINISHED | Total reward: {total_reward:.3f} ===\n")
        total_reward = 0.0


def main():
    global total_reward
    total_reward = 0.0

    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # LunarLander actions:
    # 0 = no engine
    # 1 = left engine
    # 2 = main engine
    # 3 = right engine

    # Key mapping (pygame) -> actions
    keys_to_action = {
        (pygame.K_UP,): 2,
        (pygame.K_LEFT,): 1,
        (pygame.K_RIGHT,): 3,

        # optional combinations
        (pygame.K_UP, pygame.K_LEFT): 2,
        (pygame.K_UP, pygame.K_RIGHT): 2,
    }

    play(
        env,
        keys_to_action=keys_to_action,
        callback=callback,  # print rewards
        noop=0,             # action when no key is pressed
        fps=30,
    )

    env.close()


if __name__ == "__main__":
    main()
