import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- Parameter Configuration -----------------------------

def set_parameters():
    """
    Set and return all the parameters required for the training.
    """
    params = {
        'number_of_players': 2,
        'number_of_cards': 2,
        'number_of_actions': 3,
        'batch_size': 256,
        'learning_rate': 0.5,
        'n_runs': 1,
        'n_episodes': 10000,
        'save_dir': "data/sad_txt",
        'seed': 42
    }
    return params


def get_payoff_values(number_of_cards, number_of_actions):
    """
    Define and return the payoff values tensor.
    """
    payoff_values = torch.tensor([
        [[[10, 0, 0], [4, 8, 4], [10, 0, 0]],
         [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
        [[[0, 0, 10], [4, 8, 4], [0, 0, 0]],
         [[10, 0, 0], [4, 8, 4], [10, 0, 0]]]
    ], dtype=torch.float32).to(device)
    return payoff_values


# ----------------------------- Model Definition -----------------------------

class QNet(nn.Module):
    """
    Neural network model for Q-learning.
    """

    def __init__(self, number_of_cards, number_of_actions, bad_mode):
        super(QNet, self).__init__()
        self.number_of_cards = number_of_cards
        self.number_of_actions = number_of_actions
        self.bad_mode = bad_mode
        input_size_1 = number_of_cards * number_of_actions ** 2

        # Define weights as parameters
        self.weights_0 = nn.Parameter(torch.zeros((number_of_cards, number_of_actions), device=device))
        self.weights_1 = nn.Parameter(torch.zeros((input_size_1, number_of_actions), device=device))

    def forward(self, cards_0, cards_1, u0, u0_greedy):
        """
        Forward pass to compute Q-values for both agents.
        """
        # Agent 0's Q-values
        q0 = self.weights_0[cards_0]

        # Agent 1's Q-values
        greedy_factor = 1 if self.bad_mode > 3 else 0
        joint_input_1 = cards_1 * self.number_of_actions ** 2 + u0 * self.number_of_actions + u0_greedy * greedy_factor
        q1 = self.weights_1[joint_input_1]
        return q0, q1


# ----------------------------- Action Selection -----------------------------

def select_action(q_values, epsilon, number_of_actions):
    """
    Select an action based on epsilon-greedy policy.
    """
    probs = torch.full_like(q_values, epsilon / number_of_actions)
    greedy_action = q_values.argmax(dim=-1)
    probs.scatter_(-1, greedy_action.unsqueeze(-1), 1 - epsilon + (epsilon / number_of_actions))
    action = torch.multinomial(probs, 1).squeeze(-1)
    selected_q_value = q_values[range(len(action)), action]
    greedy_q_value = q_values[range(len(greedy_action)), greedy_action]
    return action, selected_q_value, greedy_action, greedy_q_value


# ----------------------------- Training Function -----------------------------

def train_table_q_learning(params, bad_mode, vdn):
    """
    Train the Q-learning agents and save the Q-tables.
    """
    # Unpack parameters
    number_of_cards = params['number_of_cards']
    number_of_actions = params['number_of_actions']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    n_runs = params['n_runs']
    n_episodes = params['n_episodes']
    seed = params['seed']
    save_dir = params['save_dir']

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get payoff values
    payoff_values = get_payoff_values(number_of_cards, number_of_actions)

    all_rewards = np.zeros((n_runs, n_episodes // 100 + 1))
    interval = n_episodes // 100

    os.makedirs(save_dir, exist_ok=True)
    model = QNet(number_of_cards, number_of_actions, bad_mode).to(device)

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        # Initialize optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for episode in range(n_episodes + 1):
            # Generate sample inputs
            cards_0 = torch.randint(0, number_of_cards, (batch_size,)).to(device)
            cards_1 = torch.randint(0, number_of_cards, (batch_size,)).to(device)
            epsilon = max(0.05, 1 - 2 * episode / n_episodes)

            # Agent 0
            q0 = model.weights_0[cards_0]
            u0, q0_selected, u0_greedy, q0_greedy = select_action(q0, epsilon, number_of_actions)

            # Agent 1
            q1 = model(cards_0, cards_1, u0, u0_greedy)[1]
            u1, q1_selected, u1_greedy, q1_greedy = select_action(q1, epsilon, number_of_actions)

            # Compute rewards
            rewards = payoff_values[cards_0, cards_1, u0, u1]

            # Compute loss
            target_q0 = rewards + q1_greedy * vdn
            loss_q0 = F.mse_loss(q0_selected, target_q0.detach())
            loss_q1 = F.mse_loss(q1_selected, rewards.detach())
            total_loss = loss_q0 + loss_q1

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Record rewards
            if episode % interval == 0:
                with torch.no_grad():
                    # Evaluate with epsilon=0 (greedy actions)
                    q0_eval = model.weights_0[cards_0]
                    u0_eval = q0_eval.argmax(dim=-1)
                    q1_eval = model(cards_0, cards_1, u0_eval, u0_eval)[1]
                    u1_eval = q1_eval.argmax(dim=-1)
                    rewards_eval = payoff_values[cards_0, cards_1, u0_eval, u1_eval]
                    all_rewards[run, episode // interval] = rewards_eval.mean().item()
            if episode % (n_episodes // 10) == 0:
                print(f"Episode {episode}/{n_episodes}, Loss: {total_loss.item():.4f}, Reward: {rewards.mean().item():.4f}")

        # Save Q-tables
        save_q_tables(model, save_dir, bad_mode, run)
    return all_rewards


# ----------------------------- Utility Functions -----------------------------

def save_q_tables(model, save_dir, bad_mode, run):
    """
    Save the Q-tables to files.
    """
    w0 = model.weights_0.detach().cpu().numpy()
    w1 = model.weights_1.detach().cpu().numpy()

    q_tables = {
        'weights_0': w0.tolist(),
        'weights_1': w1.tolist()
    }

    w0_txt_path = os.path.join(save_dir, f"badmode_{bad_mode}_run_{run}-w0.txt")
    w1_txt_path = os.path.join(save_dir, f"badmode_{bad_mode}_run_{run}-w1.txt")
    np.savetxt(w0_txt_path, w0, fmt='%.6f', delimiter=' ')
    np.savetxt(w1_txt_path, w1, fmt='%.6f', delimiter=' ')

    print(f"Saved weights to {w0_txt_path} and {w1_txt_path}")


def plot_rewards(rewards, bad_mode):
    """
    Plot the training rewards.
    """
    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)
    episodes = np.arange(rewards.shape[1]) * (10000 // 100)

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, mean_rewards, label=f"Mode {bad_mode}")
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"Training Performance (Mode {bad_mode})")
    plt.legend()
    plt.show()


# ----------------------------- Main Execution -----------------------------

def main():
    """
    Main function to execute the training and plotting.
    """
    params = set_parameters()
    mode_labels = ['', '', 'IQL', '', 'SAD', '']

    for bad_mode in [4]:
        print(f"Running for {mode_labels[bad_mode]}")
        rewards = train_table_q_learning(params, bad_mode, vdn=0)
        plot_rewards(rewards, bad_mode)


if __name__ == "__main__":
    main()
