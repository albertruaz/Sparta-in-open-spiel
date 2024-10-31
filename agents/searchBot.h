#ifndef SEARCHBOT_H
#define SEARCHBOT_H

#include "open_spiel/spiel.h"
#include <random>
#include <vector>

class SearchBot {
public:
  SearchBot(int player_id, int num_simulations = 100); // 생성자
  ~SearchBot();                                        // 소멸자

  open_spiel::Action
  GetAction(const open_spiel::State *state,
            open_spiel::Action frame_move = open_spiel::kInvalidAction);

private:
  std::vector<double> SimulatePlay(open_spiel::State *state);
  int player_id_;
  int num_simulations_;
  std::mt19937 rng_;
};

#endif // SEARCHBOT_H
