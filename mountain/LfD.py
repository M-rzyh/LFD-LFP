import numpy as np
import gymnasium as gym
import os
import ezpickle
from agent import DISCRETE_OBSERVATION_SPACE_SIZE, get_discrete_state

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


def offline_q_learning():
    # Load demonstration dataset
    demo_dataset = np.load("good_episodes13.npy", allow_pickle=True)#change

    # Environment setup
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 200

    # Save paths
    room_name = "lfd_17"#change
    SAVE_PATH = f"./mountain/static/mountain/run/{room_name}/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    writer = SummaryWriter(log_dir=SAVE_PATH)
    REWARD_LOG = []

    # Q-table
    # q_table = np.random.uniform(
    #     low=-2, high=0,
    #     size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n])
    # )
    q_table = np.zeros(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n])

    LEARNING_RATE = 0.1
    DISCOUNT = 0.95

    # === Offline Q-learning from demo dataset ===
    for episode_idx, episode in enumerate(demo_dataset):
        states = episode['states']
        actions = episode['actions']
        next_states = episode['next_states']
        rewards = episode['rewards']
        dones = episode['dones']

        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]
            reward = rewards[i]
            done = dones[i]

            state_discrete = get_discrete_state(state)
            next_state_discrete = get_discrete_state(next_state)

            max_future_q = np.max(q_table[next_state_discrete])
            current_q = q_table[state_discrete + (action,)]
            new_q = reward if done else (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # q_table[state_discrete + (action,)] = new_q
            q_table[state_discrete + (action,)] = np.clip(new_q, -100, 100)

        if episode_idx % 100 == 0:
            print(f"Trained on episode {episode_idx}/{len(demo_dataset)}")

    # === Online fine-tuning after offline LfD ===
    EPISODES_FINE_TUNE = 5000
    epsilon = 0.1  # Low epsilon to allow mostly greedy behavior
    MIN_EPSILON = 0.01
    EPSILON_DECAY = 0.999

    for ep in range(EPISODES_FINE_TUNE):
        obs, _ = env.reset()
        discrete_obs = get_discrete_state(obs)
        total_reward = 0

        for _ in range(env._max_episode_steps):
            if np.random.rand() > epsilon:
                action = np.argmax(q_table[discrete_obs])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_discrete_obs = get_discrete_state(new_obs)

            max_future_q = np.max(q_table[new_discrete_obs])
            current_q = q_table[discrete_obs + (action,)]
            new_q = reward + DISCOUNT * max_future_q
            q_table[discrete_obs + (action,)] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * new_q

            discrete_obs = new_discrete_obs
            total_reward += reward

            if terminated or truncated:
                break

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        writer.add_scalar("Reward/FineTune", total_reward, ep)

        if ep % 500 == 0:
            print(f"[Fine-Tune] Episode {ep}, Reward: {total_reward}")

    # === Evaluation ===
    for ep in range(100):
        obs, _ = env.reset()
        total_reward = 0
        discrete_obs = get_discrete_state(obs)

        for _ in range(env._max_episode_steps):
            action = np.argmax(q_table[discrete_obs])
            new_obs, reward, terminated, truncated, _ = env.step(action)
            discrete_obs = get_discrete_state(new_obs)
            total_reward += reward
            if terminated or truncated:
                break

        REWARD_LOG.append(total_reward)
        writer.add_scalar("Reward/EvalEpisode", total_reward, ep)

    # === Save final Q-table and log ===
    ezpickle.pickle_data(q_table, os.path.join(SAVE_PATH, "agent.pkl"), overwrite=True)

    with open(os.path.join(SAVE_PATH, "reward_log.csv"), "w") as f:
        for r in REWARD_LOG:
            f.write(f"{r}\n")

    writer.close()
    print(f"Offline Q-learning from demos complete and saved to '{SAVE_PATH}'")


def behavior_cloning():
    # Load demonstrations
    demo_dataset = np.load("good_episodes13.npy", allow_pickle=True)#change

    # Prepare training data
    states, actions = [], []
    for ep in demo_dataset:
        states.extend(ep['states'])
        actions.extend(ep['actions'])

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)

    # Define behavior cloning policy network
    class PolicyNet(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PolicyNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, x):
            return self.net(x)

    # Set up training
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNet(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Logging setup
    # Save paths
    room_name = "lfd_17_bc"#change
    SAVE_PATH = f"./mountain/static/mountain/run/{room_name}/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    writer = SummaryWriter(log_dir=SAVE_PATH)
    
    # Supervised training
    EPOCHS = 100
    for epoch in range(EPOCHS):
        logits = policy(states)
        loss = loss_fn(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), epoch)
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    q_table = np.zeros(DISCRETE_OBSERVATION_SPACE_SIZE + [action_dim])

    for i in range(DISCRETE_OBSERVATION_SPACE_SIZE[0]):
        for j in range(DISCRETE_OBSERVATION_SPACE_SIZE[1]):
            discrete_state = (i, j)

            # Convert discrete state back to continuous mid-point
            env_low = env.observation_space.low
            env_high = env.observation_space.high
            bins = np.array(DISCRETE_OBSERVATION_SPACE_SIZE)
            cont_state = inverse_discretize(discrete_state, env_low, env_high, bins)
            state_tensor = torch.tensor(cont_state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = policy(state_tensor).numpy().squeeze()

            # Assign the logits to the Q-table (or softmax for probabilities)
            q_table[i, j] = logits  # Or use: softmax(logits) if you prefer
    ezpickle.pickle_data(q_table, "agent.pkl", overwrite=True)
    
    # Evaluate policy
    REWARD_LOG = []
    for ep in range(500):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(env._max_episode_steps):
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
            action = torch.argmax(logits).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        REWARD_LOG.append(total_reward)
        writer.add_scalar("Reward/EvalEpisode", total_reward, ep)

    # Save model and rewards
    ezpickle.pickle_data(q_table, os.path.join(SAVE_PATH, "agent.pkl"), overwrite=True)
    with open(os.path.join(SAVE_PATH, "reward_log.csv"), "w") as f:
        for r in REWARD_LOG:
            f.write(f"{r}\n")

    writer.close()
    print(f"Behavior cloning agent trained and saved in '{SAVE_PATH}'")

def inverse_discretize(discrete_state, env_low, env_high, bins):
    ratios = [
        (discrete_state[i] + 0.5) / bins[i] for i in range(len(discrete_state))
    ]
    cont_state = [env_low[i] + ratios[i] * (env_high[i] - env_low[i]) for i in range(len(discrete_state))]
    return np.array(cont_state)

behavior_cloning()
# offline_q_learning()