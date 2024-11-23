// TorchBot.cpp
#include "torchBot.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// 생성자: Q-테이블 로드
TorchBot::TorchBot(int player_id, const std::string &q_table_w0_path,
                   const std::string &q_table_w1_path)
    : player_id_(player_id), u0_(0) {
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

// 다른 에이전트의 행동을 전달하는 함수
void TorchBot::SetU0(int u0) { u0_ = u0; }

// 행동 선택 함수: 플레이어 ID에 따라 분기
open_spiel::Action TorchBot::GetAction(const open_spiel::State *state) {
  if (player_id_ == 0) {
    return GetActionPlayer0(state);
  } else if (player_id_ == 1) {
    return GetActionPlayer1(state);
  } else {
    std::cerr << "Unsupported player_id_: " << player_id_ << std::endl;
    exit(-1);
  }
}

// 플레이어 0의 행동 선택 함수
open_spiel::Action TorchBot::GetActionPlayer0(const open_spiel::State *state) {
  // 상태에서 card0 추출
  int card0 = 0; // 기본값

  // state에서 card0 추출
  // 별도의 함수 없이 직접 추출
  std::string observation = state->ObservationString(player_id_);
  size_t pos = observation.find("Card:");
  if (pos != std::string::npos && pos + 6 <= observation.size()) {
    char card_char = observation[pos + 5];
    if (card_char >= '0' && card_char <= '9') {
      card0 = card_char - '0';
    }
  }

  // 에이전트 0의 Q-값 가져오기
  if (card0 < 0 || card0 >= static_cast<int>(w0_.size())) {
    std::cerr << "Card index out of bounds for w0_: " << card0 << std::endl;
    exit(-1);
  }
  std::vector<float> q_values = w0_[card0];

  // 최대 Q-값을 가지는 행동 선택
  double max_q = -std::numeric_limits<double>::infinity();
  open_spiel::Action selected_action = 0;

  for (int action = 0; action <= 2; ++action) {
    double q_value = q_values[action];
    if (q_value > max_q) {
      max_q = q_value;
      selected_action = action;
    }
  }
  u0_ = selected_action; // 이전 행동 업데이트

  return selected_action;
}

// 플레이어 1의 행동 선택 함수
open_spiel::Action TorchBot::GetActionPlayer1(const open_spiel::State *state) {
  // 상태에서 card1 추출
  int card1 = 0; // 기본값

  // state에서 card1 추출
  std::string observation = state->ObservationString(player_id_);
  size_t pos = observation.find("Card:");
  if (pos != std::string::npos && pos + 6 <= observation.size()) {
    char card_char = observation[pos + 5];
    if (card_char >= '0' && card_char <= '9') {
      card1 = card_char - '0';
    }
  }

  int action0 = u0_; // 이전에 전달받은 플레이어 0의 행동

  int number_of_actions = 3;
  int number_of_actions_sq = number_of_actions * number_of_actions;
  int greedy_factor = 1; // bad_mode > 3인 경우

  // 인덱스 계산
  int index = card1 * number_of_actions_sq + action0 * number_of_actions +
              (action0 * greedy_factor);

  if (index < 0 || index >= static_cast<int>(w1_.size())) {
    std::cerr << "Index out of bounds for w1_: " << index << std::endl;
    exit(-1);
  }

  // 에이전트 1의 Q-값 가져오기
  std::vector<float> q_values = w1_[index];

  // 최대 Q-값을 가지는 행동 선택
  double max_q = -std::numeric_limits<double>::infinity();
  open_spiel::Action selected_action = 0;

  for (int action = 0; action <= 2; ++action) {
    double q_value = q_values[action];
    if (q_value > max_q) {
      max_q = q_value;
      selected_action = action;
    }
  }

  return selected_action;
}
