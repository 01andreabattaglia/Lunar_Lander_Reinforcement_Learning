import gymnasium as gym
import pygame

# Controls:
# LEFT  -> left engine (1)
# UP    -> main engine (2)
# RIGHT -> right engine (3)
# no key -> no engine (0)
# ESC   -> quit

def keys_to_action(keys):
    if keys[pygame.K_UP]:
        return 2
    if keys[pygame.K_LEFT]:
        return 1
    if keys[pygame.K_RIGHT]:
        return 3
    return 0


def main():
    env = gym.make("LunarLander-v3", render_mode="human")

    episode = 0
    running = True

    obs, info = env.reset()
    total_reward = 0.0
    terminated = truncated = False

    print("🎮 LunarLander started — play with arrow keys (ESC to quit)")

    try:
        while running:
            # Handle window close / ESC
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            # If episode ended, reset and start a new one
            if terminated or truncated:
                episode += 1
                print(f"Episode {episode} finished | Total reward: {total_reward:.3f}")

                obs, info = env.reset()
                total_reward = 0.0
                terminated = truncated = False
                continue

            keys = pygame.key.get_pressed()
            action = keys_to_action(keys)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

    finally:
        env.close()
        print("\n👋 Game closed")


if __name__ == "__main__":
    main()
