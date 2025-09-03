import os
import sys
import matplotlib.pyplot as plt

# Base path to static run folders
BASE_STATIC_PATH = os.path.join(
    os.path.dirname(__file__), 'mountain', 'static', 'mountain', 'run'
)

def get_reward_log_path(run_id='0'):
    return os.path.join(BASE_STATIC_PATH, run_id, 'reward_log.csv'), os.path.join(BASE_STATIC_PATH, run_id, 'preference_scores.csv')

def load_rewards(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reward log not found at: {path}")
    
    with open(path, 'r') as f:
        rewards = [float(line.strip()) for line in f if line.strip()]
    return rewards

def plot_rewards(rewards, run_id):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label=f"Run {run_id} - Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"learning_curve_{run_id}.png")
    print(f"Saved learning curve to: learning_curve_{run_id}.png")
    plt.show()

def plot_preferences(preferences, run_id):
    if not preferences:
        print("No preference scores found.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(preferences, label=f"Run {run_id} - Preference Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Preference Score")
    plt.title("Preference Learning Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"preference_curve_{run_id}.png")
    print(f"Saved preference curve to: preference_curve_{run_id}.png")
    plt.show()

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else '0'
    path_reward, path_preference = get_reward_log_path(run_id)
    rewards = load_rewards(path_reward)
    preferences = load_rewards(path_preference)
    plot_rewards(rewards, run_id)
    plot_preferences(preferences, run_id)