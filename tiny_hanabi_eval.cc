#include "agents/searchBot.h"
#include "agents/torchBot.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

// 결과를 출력하는 함수
void PrintResults(const std::map<std::pair<int, int>, int> &state_counts,
                  const std::map<std::pair<int, int>, double> &state_rewards,
                  int iterations) {
  std::cout << "\n총 반복 횟수: " << iterations << "\n";
  std::cout << "상태별 빈도 및 평균 보상:\n";
  for (int card0 = 0; card0 <= 1; ++card0) {
    for (int card1 = 0; card1 <= 1; ++card1) {
      int count = state_counts.at({card0, card1});
      double frequency = static_cast<double>(count) / iterations;
      double average_reward =
          count > 0 ? state_rewards.at({card0, card1}) / count : 0.0;
      std::cout << "State (" << card0 << ", " << card1 << "): "
                << "빈도 = " << frequency << ", 평균 보상 = " << average_reward
                << "\n";
    }
  }
}

int main(int argc, char **argv) {
  // 파라미터 설정
  std::string playerList[2] = {"searchbot", "blueprint"};
  int iterations = 100; // 반복 횟수를 늘려서 테스트

  // 통계 변수
  std::map<std::pair<int, int>, int>
      state_counts; // 각 (card0, card1) 상태의 빈도수
  std::map<std::pair<int, int>, double> state_rewards; // 각 상태의 총 보상

  // 가능한 모든 상태에 대해 초기화
  for (int card0 = 0; card0 <= 1; ++card0) {
    for (int card1 = 0; card1 <= 1; ++card1) {
      state_counts[{card0, card1}] = 0;
      state_rewards[{card0, card1}] = 0.0;
    }
  }

  // 랜덤 넘버 생성기 (반복문 밖에서 초기화)
  std::mt19937 rng(std::random_device{}());

  // 반복 실행
  for (int iter = 0; iter < iterations; ++iter) {
    // 게임 초기화
    std::shared_ptr<const open_spiel::Game> game =
        open_spiel::LoadGame("tiny_hanabi");
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();

    // 봇 인스턴스 생성
    std::string q_table_w0_path = "data/sad_txt/badmode_4_run_0-w0.txt";
    std::string q_table_w1_path = "data/sad_txt/badmode_4_run_0-w1.txt";

    TorchBot torch_bot(1, q_table_w0_path, q_table_w1_path);
    SearchBot search_bot(0, 100);

    // 게임 플레이
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        // 랜덤으로 액션 선택
        auto action = open_spiel::SampleAction(outcomes, rng).first;
        state->ApplyAction(action);
      } else {
        // 현재 플레이어 정보 및 액션 선택
        open_spiel::Player player = state->CurrentPlayer();

        // 봇 결정
        std::string method = playerList[player];
        open_spiel::Action action;

        if (method == "searchbot") {
          // 플레이어 0
          action = search_bot.GetAction(state.get());
          torch_bot.SetU0(action); // TorchBot에 플레이어 0의 액션 전달
        } else if (method == "blueprint") {
          // 플레이어 1
          action = torch_bot.GetAction(state.get());
        } else {
          // 랜덤 액션 (여기서는 사용되지 않음)
          auto actions = state->LegalActions(player);
          std::uniform_int_distribution<> dis(0, actions.size() - 1);
          action = actions[dis(rng)];
        }

        state->ApplyAction(action);
      }
    }

    // 최종 보상 획득
    std::vector<double> final_returns = state->Returns();
    double total_reward = final_returns[0]; // + final_returns[1];

    int card0 = -1;
    int card1 = -1;

    // 플레이어 0의 관찰 문자열에서 카드 추출
    std::string observation_p0 = state->ObservationString(0);
    size_t pos0 = observation_p0.find("p0:d");
    if (pos0 != std::string::npos && pos0 + 4 < observation_p0.size()) {
      char card_char = observation_p0[pos0 + 4];
      if (card_char >= '0' && card_char <= '9') {
        card0 = card_char - '0';
      }
    }

    // 플레이어 1의 관찰 문자열에서 카드 추출
    std::string observation_p1 = state->ObservationString(1);
    size_t pos1 = observation_p1.find("p1:d");
    if (pos1 != std::string::npos && pos1 + 4 < observation_p1.size()) {
      char card_char = observation_p1[pos1 + 4];
      if (card_char >= '0' && card_char <= '9') {
        card1 = card_char - '0';
      }
    }

    // 상태 정보 업데이트
    if (card0 != -1 && card1 != -1) {
      state_counts[{card0, card1}] += 1;
      state_rewards[{card0, card1}] += total_reward;
    } else {
      std::cerr << "Iteration " << iter << ": 카드 정보를 추출하지 못했습니다."
                << std::endl;
    }
  }

  // 결과 출력
  PrintResults(state_counts, state_rewards, iterations);

  return 0;
}
