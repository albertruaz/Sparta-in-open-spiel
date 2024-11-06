#include "searchBot.h"
#include "open_spiel/spiel_utils.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <map>

SearchBot::SearchBot(int player_id, int num_simulations)
    : player_id_(player_id), num_simulations_(num_simulations),
      rng_(std::random_device{}()) {}

SearchBot::~SearchBot() {}

// UCB 계산에 필요한 구조체 정의
struct ActionStats {
  int N = 0;           // 해당 행동의 시뮬레이션 횟수
  double W = 0.0;      // 누적 보상 합계
  double mean = 0.0;   // 평균 보상
  bool pruned = false; // 가지치기 여부
};

// GetAction 메서드 수정
open_spiel::Action SearchBot::GetAction(const open_spiel::State *state,
                                        open_spiel::Action frame_move) {
  // 현재 플레이어의 합법적 행동 가져오기
  std::vector<open_spiel::Action> legal_actions =
      state->LegalActions(player_id_);

  // 가능한 행동이 하나뿐이라면 바로 반환
  if (legal_actions.size() == 1) {
    return legal_actions[0];
  }

  // 기본 정책 움직임(bp_move) 설정
  open_spiel::Action bp_move =
      legal_actions[0] /* 기본 정책 함수 호출하여 bp_move를 설정 */;

  // 각 행동에 대한 통계 초기화
  std::map<open_spiel::Action, ActionStats> stats;
  for (auto action : legal_actions) {
    stats[action] = ActionStats();
  }

  int total_simulations = 0;
  int max_simulations = num_simulations_;
  bool frame_bail = false; // 프레임워크 움직임 중단 여부

  // 시뮬레이션 루프 시작
  while (total_simulations < max_simulations && !frame_bail) {
    // 총 시뮬레이션 횟수 계산
    double total_N = 0;
    for (const auto &kv : stats)
      total_N += kv.second.N;

    double log_total_N = std::log(total_N + 1);

    // UCB 값을 기반으로 다음 시뮬레이션할 행동 선택
    double best_ucb = -std::numeric_limits<double>::infinity();
    open_spiel::Action best_action = open_spiel::kInvalidAction;
    for (const auto &kv : stats) {
      open_spiel::Action action = kv.first;
      const ActionStats &action_stats = kv.second;
      if (action_stats.pruned)
        continue;

      double ucb_value;
      if (action_stats.N == 0) {
        ucb_value =
            std::numeric_limits<double>::infinity(); // 아직 시뮬레이션하지 않은
                                                     // 행동은 우선 선택
      } else {
        double c = std::sqrt(2); // 탐색 매개변수
        ucb_value =
            action_stats.mean + c * std::sqrt(log_total_N / action_stats.N);
      }
      if (ucb_value > best_ucb) {
        best_ucb = ucb_value;
        best_action = action;
      }
    }

    // 선택된 행동 시뮬레이션
    std::unique_ptr<open_spiel::State> state_clone = state->Clone();
    state_clone->ApplyAction(best_action);
    std::vector<double> returns = SimulatePlay(state_clone.get());
    double reward = returns[player_id_];

    // 통계 업데이트
    ActionStats &action_stats = stats[best_action];
    action_stats.N += 1;
    action_stats.W += reward;
    action_stats.mean = action_stats.W / action_stats.N;

    total_simulations += 1;

    // 가지치기 조건 검사
    double best_mean = -std::numeric_limits<double>::infinity();
    for (const auto &kv : stats) {
      if (!kv.second.pruned && kv.second.mean > best_mean) {
        best_mean = kv.second.mean;
      }
    }

    for (auto &kv : stats) {
      open_spiel::Action action = kv.first;
      ActionStats &action_stats = kv.second;
      if (action_stats.pruned || action_stats.N == 0) {
        continue;
      }
      double c = std::sqrt(2);
      double ucb =
          action_stats.mean + c * std::sqrt(log_total_N / action_stats.N);
      if (best_mean > ucb) {
        action_stats.pruned = true;
        if (action == frame_move) {
          frame_bail = true;
          break;
        }
      }
    }
  }

  // 프레임워크 움직임이 가지치기된 경우, 특별한 값 반환
  if (frame_bail) {
    return open_spiel::kInvalidAction; // 또는 특정한 동작을 정의할 수 있음
  }

  // 가지치기되지 않은 행동 중 평균 보상이 가장 높은 행동 선택
  double best_mean = -std::numeric_limits<double>::infinity();
  open_spiel::Action best_action = legal_actions[0];

  for (const auto &kv : stats) {
    open_spiel::Action action = kv.first;
    const ActionStats &action_stats = kv.second;
    if (action_stats.pruned) {
      continue;
    }
    if (action_stats.mean > best_mean) {
      best_mean = action_stats.mean;
      best_action = action;
    }
  }

  // 최종적으로 선택된 행동 출력 (디버깅용)
  if (bp_move != best_action)
    std::cerr << "Search changed move. ";
  std::cout << "Blueprint picked " << state->ActionToString(player_id_, bp_move)
            << " with average score " << stats[bp_move].mean
            << "; search picked "
            << state->ActionToString(player_id_, best_action)
            << " with average score " << stats[best_action].mean << std::endl;

  return best_action;
}

// SimulatePlay 메서드 수정
std::vector<double> SearchBot::SimulatePlay(open_spiel::State *state) {
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      // rng를 사용하여 SampleAction 호출
      auto action = open_spiel::SampleAction(outcomes, rng_).first;
      state->ApplyAction(action);
    } else if (state->IsSimultaneousNode()) {
      std::vector<open_spiel::Action> joint_action(state->NumPlayers());
      for (open_spiel::Player p = 0; p < state->NumPlayers(); ++p) {
        std::vector<open_spiel::Action> actions = state->LegalActions(p);
        std::uniform_int_distribution<> dist(0, actions.size() - 1);
        joint_action[p] = actions[dist(rng_)];
      }
      state->ApplyActions(joint_action);
    } else {
      std::vector<open_spiel::Action> actions = state->LegalActions();
      std::uniform_int_distribution<> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng_)]);
    }
  }
  return state->Returns();
}
