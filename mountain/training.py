import numpy as np
import gymnasium as gym
import os
import ezpickle
from agent import DISCRETE_OBSERVATION_SPACE_SIZE, get_discrete_state
from torch.utils.tensorboard import SummaryWriter

room_name = "31"#change
SAVE_PATH = f"./mountain/static/mountain/run/{room_name}/"

def train():
    # Create environment
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 200  # Same as in SHARPIE

    # Initialize Q-table
    q_table = np.random.uniform(
        low=-2, high=0, 
        size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n])
    )

    # Q-learning config
    EPISODES = 100000
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    # epsilon = 0.05

    epsilon = 1.0
    EPSILON_DECAY = 0.9995
    MIN_EPSILON = 0.05

    
    REWARD_LOG = []
    writer = SummaryWriter(log_dir=f"runs/{room_name}")
    # f"/Users/marziehghayour/Library/Mobile Documents/com~apple~CloudDocs/Academia/RA/Code/H-AI/SHARPIE/mountain/static/mountain/run/{room_name}/"

    reward_threshold = -110
    demo_dataset = []
    all_episodes = []

    def update_q_table(discrete_obs, new_discrete_obs, action, reward, agent, terminated, truncated, observation):
        goal_reached = observation[0] >= 0.5

        if goal_reached:
            agent[discrete_obs + (action,)] = 100
        # else:
        elif not (terminated or truncated):
            max_future_q = np.max(agent[new_discrete_obs])
            current_q = agent[discrete_obs + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            agent[discrete_obs + (action,)] = new_q
            
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)

    for episode in range(EPISODES):
        obs, _ = env.reset()
        # print("initial obs:", obs)
        
        #manually randomize the initial state a little bit more
        # obs[0] += np.random.uniform(-0.02, 0.02)  # position
        # obs[1] += np.random.uniform(-0.005, 0.005)  # velocity
        # env.state = obs
        
        discrete_obs = get_discrete_state(obs)
        done = False
        total_reward = 0

        episode_data = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }
        
        for step in range(env._max_episode_steps):
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_obs])
                # print("Action taken:", action)  # Debugging output
            else:
                # print("Random action taken")
                action = np.random.randint(0, env.action_space.n)

            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_discrete_obs = get_discrete_state(new_obs)

            # Log transition
            episode_data['states'].append(obs)
            episode_data['actions'].append(action)
            episode_data['next_states'].append(new_obs)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(terminated or truncated)

            update_q_table(discrete_obs, new_discrete_obs, action, reward, q_table, terminated, truncated, new_obs)

            discrete_obs = new_discrete_obs
            total_reward += reward
            if terminated or truncated or new_obs[0] >= 0.5:
                break
        
        all_episodes.append((episode_data, total_reward))
        REWARD_LOG.append(total_reward)
        writer.add_scalar("Reward/Episode", total_reward, episode)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
            
        # if total_reward >= reward_threshold:
        #     demo_dataset.append(episode_data)

    # np.save("good_episodes.npy", demo_dataset)
    # Sort episodes by a performance metric: shortest to reach goal OR highest reward
    # Sort by reward descending (can change to episode length or another custom score)
    # sorted_episodes = sorted(all_episodes, key=lambda x: x[1], reverse=True)
    # Select top 5000 (or fewer if fewer exist)
    # demo_dataset = [ep[0] for ep in sorted_episodes[:5000]]
    # Filter for episodes where the goal was reached (position >= 0.5 at any time)
    successful_episodes = []
    for episode_data, total_reward in all_episodes:
        positions = [s[0] for s in episode_data['next_states']]
        if any(pos >= 0.5 for pos in positions) and total_reward >= reward_threshold:
            successful_episodes.append((episode_data, total_reward))

    # Sort by reward descending or another metric
    sorted_successful = sorted(successful_episodes, key=lambda x: x[1], reverse=True)

    # Select top N (e.g., 5000) for LfD
    demo_dataset = [ep[0] for ep in sorted_successful[:5000]]

    # Save demonstration dataset
    np.save("good_episodes14.npy", demo_dataset)
    print(f"Saved top {len(demo_dataset)} episodes to good_episodes.npy")

    # Save final Q-table
    ezpickle.pickle_data(q_table, os.path.join(SAVE_PATH, "agent.pkl"), overwrite=True)
    # Save reward log
    with open(os.path.join(SAVE_PATH, "reward_log.csv"), "w") as f:
        for r in REWARD_LOG:
            f.write(f"{r}\n")

    writer.close()
    print("Training finished and Q-table saved.")

    # for ep in range(100):
    #     obs, _ = env.reset()
    #     total_reward = 0
    #     discrete_obs = get_discrete_state(obs)
    #     writer = SummaryWriter(log_dir=SAVE_PATH)

    #     for _ in range(env._max_episode_steps):
    #         action = np.argmax(q_table[discrete_obs])
    #         new_obs, reward, terminated, truncated, _ = env.step(action)
    #         discrete_obs = get_discrete_state(new_obs)
    #         total_reward += reward
    #         if terminated or truncated:
    #             break

    #     REWARD_LOG.append(total_reward)
    #     writer.add_scalar("Reward/EvalEpisode", total_reward, ep)


    #     with open(os.path.join(SAVE_PATH, "reward_log_eval.csv"), "w") as f:
    #         for r in REWARD_LOG:
    #             f.write(f"{r}\n")

    #     writer.close()

# === Evaluation Function for Trained Agent ===
def evaluate_trained_agent(q_table_path, log_dir, num_episodes=100):
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 200
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "eval"))

    q_table = ezpickle.unpickle_data(q_table_path)
    eval_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        discrete_obs = get_discrete_state(obs)
        total_reward = 0

        for _ in range(env._max_episode_steps):
            action = np.argmax(q_table[discrete_obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            discrete_obs = get_discrete_state(obs)
            total_reward += reward
            if terminated or truncated:
                break

        eval_rewards.append(total_reward)
        writer.add_scalar("Reward/EvalEpisode", total_reward, ep)

    writer.close()
    print(f"Evaluation finished. Average reward over {num_episodes} episodes: {np.mean(eval_rewards):.2f}")

# train()

# evaluate_trained_agent(
#     q_table_path=os.path.join(SAVE_PATH, "agent.pkl"),
#     log_dir=SAVE_PATH,
#     num_episodes=200
# )