import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
import pygame

gym.register_envs(ale_py)

# play() NON accetta render_mode="human"
env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

key_to_action = {
    (pygame.K_LEFT,): 3,    # LEFT
    (pygame.K_RIGHT,): 2,   # RIGHT
    (pygame.K_UP,): 1,      # UP
    (pygame.K_DOWN,): 4,    # DOWN

    # diagonali opzionali
    (pygame.K_UP, pygame.K_RIGHT): 5,      # UPRIGHT
    (pygame.K_UP, pygame.K_LEFT): 6,       # UPLEFT
    (pygame.K_DOWN, pygame.K_RIGHT): 7,    # DOWNRIGHT
    (pygame.K_DOWN, pygame.K_LEFT): 8,     # DOWNLEFT
}

play(env, keys_to_action=key_to_action, fps=30, zoom=2)
env.close()
