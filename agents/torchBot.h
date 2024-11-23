// TorchBot.h
#ifndef TORCHBOT_H
#define TORCHBOT_H

#include "open_spiel/spiel.h"
#include <string>
#include <vector>

class TorchBot {
public:
  TorchBot(int player_id, const std::string &q_table_w0_path,
           const std::string &q_table_w1_path);
  ~TorchBot();

  open_spiel::Action GetAction(const open_spiel::State *state);
  void SetU0(int u0); // 다른 에이전트의 행동을 전달하기 위한 함수

private:
  // Q-테이블을 로드하는 함수
  void LoadQTable(const std::string &file_path,
                  std::vector<std::vector<float>> &q_table);

  // 플레이어별 행동 선택 함수
  open_spiel::Action GetActionPlayer0(const open_spiel::State *state);
  open_spiel::Action GetActionPlayer1(const open_spiel::State *state);

  int player_id_;
  int u0_; // 플레이어 0의 이전 행동을 저장
  std::vector<std::vector<float>> w0_; // 에이전트 0의 Q-테이블
  std::vector<std::vector<float>> w1_; // 에이전트 1의 Q-테이블
};

#endif // TORCHBOT_H
