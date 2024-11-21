#include "torchBot.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// 생성자: Q-테이블 로드
TorchBot::TorchBot(int player_id, const std::string &q_table_w0_path,
                   const std::string &q_table_w1_path)
    : player_id_(player_id), u0_(0), rng_(std::random_device{}()) {
  // Q-테이블 로드
  LoadQTable(q_table_w0_path, w0_);
  LoadQTable(q_table_w1_path, w1_);
}

TorchBot::~TorchBot() {}

// Q-테이블을 파일에서 로드하는 함수
void TorchBot::LoadQTable(const std::string &file_path,
                          std::vector<std::vector<float>> &q_table) {
  std::ifstream infile(file_path);
  if (!infile.is_open()) {
    std::cerr << "Error opening Q-table file: " << file_path << std::endl;
    exit(-1);
  }
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::vector<float> row;
    float value;
    while (iss >> value) {
      row.push_back(value);
    }
    q_table.push_back(row);
  }
  infile.close();
}
void TorchBot::SetU0(int u0) { u0_ = u0; }

// GetAction 메서드
open_spiel::Action TorchBot::GetAction(const open_spiel::State *state) {
  // 현재 플레이어의 합법적 행동 가져오기
  std::vector<open_spiel::Action> legal_actions =
      state->LegalActions(player_id_);

  // 가능한 행동이 하나뿐이라면 바로 반환
  if (legal_actions.size() == 1) {
    if (player_id_ == 0) {
      u0_ = legal_actions[0]; // 이전 행동 업데이트
    }
    return legal_actions[0];
  }

  // 게임 상태에서 필요한 정보 추출
  int card0 = GetPlayerCard(state, 0);
  int card1 = GetPlayerCard(state, 1);

  int number_of_actions = 3;
  open_spiel::Action selected_action = legal_actions[0];

  if (player_id_ == 0) {
    // 에이전트 0의 턴
    // 에이전트 0의 Q값 가져오기
    std::vector<float> q_values = w0_[card0];

    // 합법적 행동 중에서 최대 Q값을 가지는 행동 선택
    double max_q = -std::numeric_limits<double>::infinity();
    for (auto action : legal_actions) {
      if (action >= q_values.size())
        continue; // 범위 체크
      double q_value = q_values[action];
      if (q_value > max_q) {
        max_q = q_value;
        selected_action = action;
      }
    }
    u0_ = selected_action; // 이전 행동 업데이트
  } else {
    // 에이전트 1의 턴
    // 인덱스 계산
    int greedy_factor = 1; // bad_mode > 3인 경우
    int index = card1 * number_of_actions * number_of_actions +
                u0_ * number_of_actions + (u0_ * greedy_factor);

    if (index >= w1_.size()) {
      std::cerr << "Index out of bounds for w1_: " << index << std::endl;
      exit(-1);
    }

    // 에이전트 1의 Q값 가져오기
    std::vector<float> q_values = w1_[index];

    // 합법적 행동 중에서 최대 Q값을 가지는 행동 선택
    double max_q = -std::numeric_limits<double>::infinity();
    for (auto action : legal_actions) {
      if (action >= q_values.size())
        continue; // 범위 체크
      double q_value = q_values[action];
      if (q_value > max_q) {
        max_q = q_value;
        selected_action = action;
      }
    }
  }

  return selected_action;
}

// 현재 플레이어의 카드를 얻는 함수
int TorchBot::GetPlayerCard(const open_spiel::State *state, int player_id) {
  // Tiny-Hanabi의 상태에서 플레이어의 카드를 얻는 로직 구현
  // 상태의 관찰 문자열에서 "Card:X" 형태로 카드 정보를 추출한다고 가정

  std::string observation = state->ObservationString(player_id);
  // "Card:" 위치 찾기
  size_t pos = observation.find("Card:");
  if (pos != std::string::npos && pos + 6 <= observation.size()) {
    // 카드 번호 추출
    char card_char = observation[pos + 5];
    if (card_char >= '0' && card_char <= '9') {
      int card = card_char - '0'; // 한 자리수 카드 번호라고 가정
      return card;
    }
  }
  // 카드를 찾지 못한 경우 기본값 0 반환
  return 0;
}
