#ifndef TORCHBOT_H
#define TORCHBOT_H

#include "open_spiel/spiel.h"
#include <random>
#include <string>
#include <vector>

class TorchBot {
public:
  TorchBot(int player_id, const std::string &q_table_w0_path,
           const std::string &q_table_w1_path);
  ~TorchBot();

  open_spiel::Action GetAction(const open_spiel::State *state);

  void SetU0(int u0);

private:
  int player_id_;
  int u0_; // 에이전트 0의 이전 행동

  std::vector<std::vector<float>> w0_; // 에이전트 0의 Q-테이블
  std::vector<std::vector<float>> w1_; // 에이전트 1의 Q-테이블

  int GetPlayerCard(const open_spiel::State *state, int player_id);
  void LoadQTable(const std::string &file_path,
                  std::vector<std::vector<float>> &q_table);

  std::mt19937 rng_; // 랜덤 넘버 생성기 (필요한 경우)
};

#endif // TORCHBOT_H
