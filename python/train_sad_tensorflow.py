# conda create -n tf1_env python=3.7
# pip install tensorflow==1.15.0 numpy matplotlib scikit-learn
# pip install protobuf==3.20.3


# 필요한 라이브러리를 임포트합니다.
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # TensorFlow 1.x 모드 활성화
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 반복 텐서를 생성하는 함수입니다.
def repeat_tensor(tensor, repetition):
    with tf.variable_scope("rep"):
        exp_tensor = tf.expand_dims(tensor, -1)
        tensor_t = tf.tile(exp_tensor, [1] + repetition)
        tensor_r = tf.reshape(tensor_t, repetition * tf.shape(tensor))
    return tensor_r

# 보상 값을 정의합니다.
payoff_values = [
    [[[10, 0, 0], [4, 8, 4], [10, 0, 0]],
     [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
    [[[0, 0, 10], [4, 8, 4], [0, 0, 0]],
     [[10, 0, 0], [4, 8, 4], [10, 0, 0]]]
]
payoff_values = np.array(payoff_values)

# 파라미터 설정
number_of_players = 2  # 플레이어 수
number_of_cards = 2    # 카드 종류 수
number_of_actions = 3  # 행동 수
bs = 32                # 배치 크기

# 에이전트의 행동과 q-값을 계산하는 함수입니다.
def p0(in0, w0_, eps, input_size):
    # in0: 에이전트의 입력, w0_: 가중치
    q_vals = tf.matmul(tf.one_hot(in0, input_size), w0_)
    max_vals = tf.math.argmax(q_vals, 1)
    probs = q_vals * 0 + eps / number_of_actions
    probs = probs + (1 - eps) * tf.one_hot(max_vals, number_of_actions)
    logs = tf.log(probs)

    u0_ = tf.stop_gradient(tf.cast(tf.squeeze(tf.multinomial(logs, 1)), tf.int32))
    u0_greedy = tf.stop_gradient(tf.cast(tf.squeeze(tf.math.argmax(logs, 1)), tf.int32))

    q_val = tf.reduce_sum(tf.multiply(q_vals, tf.one_hot(u0_, number_of_actions)), -1)
    q_greedy = tf.reduce_sum(tf.multiply(q_vals, tf.one_hot(u0_greedy, number_of_actions)), -1)

    return q_val, u0_, u0_greedy, q_greedy



# 학습을 위한 그래프와 연산을 정의하는 함수입니다.
def get_ops(bad_mode, seed, vdn):
    tf.reset_default_graph()

    # 난수 시드를 설정합니다.
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # 플레이어 0과 1의 입력 플레이스홀더를 정의합니다.
    input_0 = tf.placeholder(tf.int32, shape=(bs))
    input_1 = tf.placeholder(tf.int32, shape=(bs))
    eps = tf.placeholder(tf.float32, shape=())

    # 보상 매트릭스를 위한 플레이스홀더를 정의합니다.
    payout_ph = tf.placeholder(tf.float32, shape=(2, 2, 3, 3))

    # 에이전트의 가중치 변수를 정의합니다.
    weights_0 = tf.get_variable('agent0', shape=(number_of_cards, number_of_actions))
    if bad_mode > 3:
        greedy = 1
    else:
        greedy = 0

    input_size_1 = number_of_cards * number_of_actions ** 2
    weights_1 = tf.get_variable('agent1', shape=(input_size_1, number_of_actions))

    # 에이전트 0의 q-값과 행동을 계산합니다.
    q0, u0, u0_greedy, _ = p0(input_0, weights_0, eps, number_of_cards)

    # 에이전트 1의 입력을 생성합니다.
    joint_in1 = input_1 * number_of_actions ** 2 + u0 * number_of_actions + u0_greedy * greedy

    # 에이전트 1의 q-값과 행동을 계산합니다.
    q1, u1, _, q1_greedy = p0(joint_in1, weights_1, eps, input_size_1)

    # 배치의 각 요소에 대한 보상을 계산합니다.
    rewards = tf.cast(tf.stack([payout_ph[input_0[i], input_1[i], u0[i], u1[i]] for i in range(bs)], 0), tf.float32)

    # 손실 함수를 정의하고 최적화합니다.
    opt = tf.train.GradientDescentOptimizer(0.5)
    total_value_loss = tf.reduce_mean(tf.pow(tf.stop_gradient(rewards + q1_greedy * vdn) - q0, 2))
    total_value_loss += tf.reduce_mean(tf.pow(rewards - q1, 2))
    train_value = opt.minimize(total_value_loss)

    # 변수 초기화를 위한 연산을 정의합니다.
    init = tf.global_variables_initializer()

    # 플레이스홀더와 트레이닝 연산을 묶어서 반환합니다.
    ph = {
        'payout_ph': payout_ph,
        'input_0': input_0,
        'input_1': input_1,
        'eps': eps
    }
    tr_ops = {
        'v': train_value,
        'weights0': weights_0,
        'u0': u0,
        'u1': u1,
        'q_p1': q1,
        'q_0': q0
    }

    return rewards, total_value_loss, tr_ops, init, ph

# 학습 설정
debug = True
seed = 42
vdn = 0
final_epsilon = 0.05

if debug:
    n_runs = 20
    n_episodes = 10000
    n_readings = 100
else:
    n_runs = 100
    n_episodes = 35000
    n_readings = 100

mode_labels = ['', '', 'IQL', '', 'SAD', '']
all_cards = np.zeros((1, number_of_cards))
for i in range(number_of_cards):
    all_cards[0, i] = i

all_r = np.zeros((6, n_runs, n_readings + 1))
all_w = np.zeros((6, n_runs, n_readings + 1, number_of_cards, number_of_actions))
interval = n_episodes // n_readings

# 학습 루프
for bad_mode in [2, 4]:
    print('Running for', mode_labels[bad_mode])
    rewards_op, total_value_loss, train_ops, init, ph = get_ops(bad_mode, seed, vdn)
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()  # Saver 객체 생성
        for n_r in range(n_runs):
            print('run', n_r, 'out of', n_runs)
            for j in range(n_episodes + 1):
                cards_0 = np.random.choice(number_of_cards, size=(bs))
                cards_1 = np.random.choice(number_of_cards, size=(bs))
                epsilon = max(final_epsilon, 1 - 2 * j / n_episodes)

                feed_dict = {
                    ph['eps']: epsilon,
                    ph['payout_ph']: payoff_values,
                    ph['input_0']: cards_0,
                    ph['input_1']: cards_1
                }

                qp0, qp1, u0v, u1v, w0, rew, v1, _ = sess.run(
                    [train_ops['q_0'], train_ops['q_p1'], train_ops['u0'], train_ops['u1'],
                     train_ops['weights0'], rewards_op, total_value_loss, train_ops['v']],
                    feed_dict)
                if j % interval == 0:
                    rew = sess.run(rewards_op, {
                        ph['eps']: 0.00,
                        ph['payout_ph']: payoff_values,
                        ph['input_0']: cards_0,
                        ph['input_1']: cards_1
                    })
                    all_r[bad_mode, n_r, int(j / interval)] = np.mean(rew)
                    all_w[bad_mode, n_r, int(j / interval)] = w0
                if j % (n_episodes // 10) == 0 and debug:
                    print(j, 'rew', np.mean(rew))
            # 모델 저장
            save_path = saver.save(sess, f"./ckpt_files/model_badmode{bad_mode}_run{n_r}.ckpt")
            print(f"Model saved in path: {save_path}")
            

# # 결과를 시각화합니다.
# mode_labels = ['', '', 'IQL', 'IQL+aux', 'SAD', '']
# colors = ['', '', '#1f77b4', '', '#d62728']
# plt.figure(figsize=(3.5, 3.5))
# x_vals = np.arange(n_readings + 1) * interval
# for bad_mode in [2, 4]:
#     vals = all_r[bad_mode]
#     y_m = vals.mean(0)
#     y_std = vals.std(0) / (n_runs ** 0.5)
#     plt.plot(x_vals, y_m, colors[bad_mode], label=mode_labels[bad_mode])
#     plt.fill_between(x_vals, y_m + y_std, y_m - y_std, alpha=0.3)
#     plt.ylim([7.5, 10.3])
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Reward')
# plt.savefig('matrix_game.pdf')
# plt.show()
# 

