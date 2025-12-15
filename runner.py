import gymnasium as gym
from src.envs.make_env import make_env
from src.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from src.networks.q_network import QNetwork
from src.replay.replay_buffer import ReplayBuffer
from src.training.trainer import Trainer

# Global variable to accumulate reward per episode
total_reward = 0.0


def main():
    global total_reward
    
    # Initialize Q-Network
    q_network = QNetwork(input_size=8, output_size=4)
    
    # Initialize agent with epsilon-greedy strategy
    agent = EpsilonGreedyAgent(action_space=4, q_network=q_network, epsilon=0.1)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=50000)
    
    # Initialize trainer with PyTorch's built-in optimizer
    trainer = Trainer(q_network, learning_rate=0.001, replay_buffer=replay_buffer, gamma=0.99)
    
    env = make_env(render_mode="rgb_array")

    num_episodes = 100
    best_reward = float('-inf')
    best_episode = -100
    episode_rewards = []
    training_steps = 0
    train_every = 1  # Training step every N environment steps
    batch_size = 32
    
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
            
            # Training step: sample batch and update network
            if training_steps % train_every == 0:
                loss = trainer.train_step(batch_size=batch_size)
                if loss > 0:
                    if training_steps % 500 == 0:
                        print(f"  Training step {training_steps} | Loss: {loss:.4f}")
            
            training_steps += 1
            total_reward += reward
            obs = next_obs
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} | Total reward: {total_reward:.3f} | Training steps: {training_steps}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode

    print(f"\n=== Best episode: {best_episode + 1} with reward: {best_reward:.3f} ===")
    print(f"Replay buffer size: {len(replay_buffer)} transitions")
    print(f"Total training steps: {training_steps}\n")
    
    # Replay the best episode with visualization
    env_visual = make_env(render_mode="human")
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
