import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


# 파라미터 설정
number_of_players = 2
number_of_cards = 2
number_of_actions = 3
batch_size = 32
learning_rate = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 행동 선택 # 보상 값을 정의
payoff_values = torch.tensor([
    [[[10, 0, 0], [4, 8, 4], [10, 0, 0]],
     [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
    [[[0, 0, 10], [4, 8, 4], [0, 0, 0]],
     [[10, 0, 0], [4, 8, 4], [10, 0, 0]]]
], dtype=torch.float32)

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# `payoff_values`를 device에 맞게 이동
payoff_values = payoff_values.to(device)


def select_action(q_values, epsilon):
    probs = torch.full_like(q_values, epsilon / number_of_actions)
    greedy_action = q_values.argmax(dim=-1)
    probs.scatter_(-1, greedy_action.unsqueeze(-1), 1 - epsilon + (epsilon / number_of_actions))
    action = torch.multinomial(probs, 1).squeeze(-1)
    return action, q_values[range(len(action)), action]


# 학습 함수
def train_table_q_learning(bad_mode, seed, vdn, n_runs=20, n_episodes=10000, save_dir="./saved_models"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Q-value 테이블 초기화
    weights_0 = torch.zeros((number_of_cards, number_of_actions), device=device, requires_grad=True)
    input_size_1 = number_of_cards * number_of_actions ** 2
    weights_1 = torch.zeros((input_size_1, number_of_actions), device=device, requires_grad=True)

    optimizer = torch.optim.SGD([weights_0, weights_1], lr=learning_rate)

    all_rewards = np.zeros((n_runs, 101))
    interval = n_episodes // 100

    os.makedirs(save_dir, exist_ok=True)

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        for episode in range(n_episodes + 1):
            # 샘플 입력 생성
            cards_0 = torch.randint(0, number_of_cards, (batch_size,)).to(device)
            cards_1 = torch.randint(0, number_of_cards, (batch_size,)).to(device)
            epsilon = max(0.05, 1 - 2 * episode / n_episodes)

            # 에이전트 0 Q값과 행동 계산
            q0 = weights_0[cards_0]
            u0, q0_selected = select_action(q0, epsilon)

            # 에이전트 1 입력 생성
            greedy_factor = 1 if bad_mode > 3 else 0
            joint_input_1 = cards_1 * number_of_actions ** 2 + u0 * number_of_actions
            joint_input_1 += (u0 * greedy_factor)

            # 에이전트 1 Q값과 행동 계산
            q1 = weights_1[joint_input_1]
            u1, q1_selected = select_action(q1, epsilon)

            # 보상 계산
            rewards = payoff_values[cards_0, cards_1, u0, u1]  # 이미 device에 맞춰짐

            # 손실 계산
            target_q0 = rewards + q1.max(dim=-1)[0] * vdn
            loss_q0 = F.mse_loss(q0_selected, target_q0)
            loss_q1 = F.mse_loss(q1_selected, rewards)
            total_loss = loss_q0 + loss_q1

            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 보상 기록
            if episode % interval == 0:
                all_rewards[run, episode // interval] = rewards.mean().item()
            if episode % (n_episodes // 10) == 0:
                print(f"Episode {episode}/{n_episodes}, Loss: {total_loss.item():.4f}, Reward: {rewards.mean().item():.4f}")

        # 모델 저장
        save_path = os.path.join(save_dir, f"badmode_{bad_mode}_run_{run}.pt")
        torch.save({'weights_0': weights_0.detach().cpu(), 'weights_1': weights_1.detach().cpu()}, save_path)
        print(f"Saved model to {save_path}")

    return all_rewards



# 모델 불러오기 함수
def load_model(filepath):
    checkpoint = torch.load(filepath)
    weights_0 = checkpoint['weights_0']
    weights_1 = checkpoint['weights_1']
    print(f"Loaded model from {filepath}")
    return weights_0, weights_1


# 학습 실행
if __name__ == "__main__":
    mode_labels = ['', '', 'IQL', '', 'SAD', '']
    for bad_mode in [2, 4]:
        print(f"Running for {mode_labels[bad_mode]}")
        rewards = train_table_q_learning(bad_mode, seed=42, vdn=0)

        # 결과 시각화
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(rewards.shape[1]) * (10000 // 100), rewards.mean(axis=0), label=f"Mode {bad_mode}")
        plt.fill_between(
            np.arange(rewards.shape[1]) * (10000 // 100),
            rewards.mean(axis=0) - rewards.std(axis=0),
            rewards.mean(axis=0) + rewards.std(axis=0),
            alpha=0.2
        )
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Training Performance (Mode {bad_mode})")
        plt.legend()
        plt.show()

        # 불러오기 테스트
        last_saved_model = f"./saved_models/badmode_{bad_mode}_run_{0}.pt"
        w0, w1 = load_model(last_saved_model)
        print("Loaded weights for agent 0:", w0)
        print("Loaded weights for agent 1:", w1)
