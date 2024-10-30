#include "open_spiel/spiel.h"
#include <iostream>

int main() {
  std::vector<std::string> games = open_spiel::RegisteredGames();
  std::cout << "Available games:\n";
  for (const auto& game : games) {
    std::cout << " - " << game << "\n";
  }
  return 0;
}