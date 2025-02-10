import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time


# Define the DQN network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.network(x)


# Training parameters
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
EPISODES = 500


def train():
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_size, output_size).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    # Add time tracking
    start_time = time.time()
    last_episode_checkpoint = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()

            # Take action and observe next state
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Store transition in memory
            memory.append((state, action, reward, next_state, done))
            state = next_state

            # Train on random batch from memory
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Compute Q values
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                next_q = policy_net(next_states).max(1)[0].detach()
                target_q = rewards + GAMMA * next_q * (1 - dones)

                # Compute loss and update
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Log every 25 episodes
        if (episode + 1) % 25 == 0:
            current_time = time.time()
            recent_eps = 25 / (current_time - start_time - last_episode_checkpoint)
            total_eps = (episode + 1) / (current_time - start_time)
            last_episode_checkpoint = current_time - start_time
            print(
                f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, "
                f"Recent EPS: {recent_eps:.2f}, Average EPS: {total_eps:.2f}"
            )

    # Print final statistics
    end_time = time.time()
    training_time = end_time - start_time
    final_eps = EPISODES / training_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Overall average EPS: {final_eps:.2f}")

    env.close()
    return policy_net


if __name__ == "__main__":
    trained_model = train()
