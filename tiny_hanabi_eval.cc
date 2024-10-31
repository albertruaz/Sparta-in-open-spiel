#include "agents/searchBot.h"
#include "open_cspiel/spiel_utils.h"
#include "open_spiel/spiel.h"
#include <iostream>
#include <memory>
#include <random>
#include <string>

int main(int argc, char **argv) {
  // 게임 로드: const 추가
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("tiny_hanabi");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  // 랜덤 넘버 생성기 선언
  std::mt19937 rng(std::random_device{}());

  // SearchBot 인스턴스 생성
  SearchBot search_bot(0, 100);

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      // rng를 사용하여 SampleAction 호출
      auto action = open_spiel::SampleAction(outcomes, rng).first;
      state->ApplyAction(action);
    } else {
      // 현재 플레이어 정보 및 가능한 행동 가져오기
      open_spiel::Player player = state->CurrentPlayer();
      auto actions = state->LegalActions(player);

      // 플레이어에 따른 봇 종류 결정
      std::string method = (player == 0) ? "searchbot" : "random";

      open_spiel::Action action;
      if (method == "searchbot") {
        // SearchBot을 사용하여 행동 결정
        action = search_bot.GetAction(state.get());
      } else {
        // 랜덤하게 행동 선택
        std::uniform_int_distribution<> dis(0, actions.size() - 1);
        action = actions[dis(rng)];
      }

      std::cout << "Player " << player
                << " chooses action: " << state->ActionToString(player, action)
                << "\n";

      state->ApplyAction(action);
    }
  }

  // 최종 결과 출력
  std::vector<double> final_returns = state->Returns();
  for (size_t i = 0; i < final_returns.size(); ++i) {
    std::cout << "Player " << i << " final return: " << final_returns[i]
              << "\n";
  }

  return 0;
}
