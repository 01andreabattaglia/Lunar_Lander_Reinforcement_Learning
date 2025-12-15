import gymnasium as gym
from src.envs.make_env import make_env
from src.agents.random_agent import RandomAgent
from src.replay.replay_buffer import ReplayBuffer

# Global variable to accumulate reward per episode
total_reward = 0.0


def main():
    global total_reward

    # Initialize agent and replay buffer
    agent = RandomAgent()
    replay_buffer = ReplayBuffer(max_size=50000)

    env = make_env(render_mode="rgb_array")

    num_episodes = 10
    best_reward = float('-inf')
    best_episode = -100
    episode_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Ask agent for action
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition in replay buffer
            done = terminated or truncated
            replay_buffer.add(obs, action, reward, next_obs, done)

            total_reward += reward
            obs = next_obs

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} | Total reward: {total_reward:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode

    print(f"\n=== Best episode: {best_episode + 1} with reward: {best_reward:.3f} ===")
    print(f"Replay buffer size: {len(replay_buffer)} transitions\n")

    # Replay the best episode with visualization
    env_visual = make_env(render_mode="rgb_array")
    obs, info = env_visual.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    print(f"Replaying best episode {best_episode + 1}...\n")

    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env_visual.step(action)
        total_reward += reward

    print(f"Best episode replay | Total reward: {total_reward:.3f}")
    env_visual.close()
    env.close()


if __name__ == "__main__":
    main()
