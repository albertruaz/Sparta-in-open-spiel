#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/games/hanabi/hanabi.h"
#include "open_spiel/games/tiny_hanabi/tiny_hanabi.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Function to print legal actions for a player
void PrintLegalActions(const open_spiel::State &state,
                       open_spiel::Player player,
                       const std::vector<open_spiel::Action> &actions) {
  std::cout << "Legal actions for player " << player << ":\n";
  for (auto action : actions) {
    std::cout << "  " << state.ActionToString(player, action) << "\n";
  }
}

// Function to print the information state for each player
void PrintInformationState(const open_spiel::State &state,
                           open_spiel::Player player) {
  std::cout << "Information state for player " << player << ":\n";
  std::cout << state.InformationStateString(player) << "\n";
}

// Function to print observation state for each player
void PrintObservationState(const open_spiel::State &state,
                           open_spiel::Player player) {
  std::cout << "Observation state for player " << player << ":\n";
  std::cout << state.ObservationString(player) << "\n";
}

int main(int argc, char **argv) {
  // Seed for random number generator
  std::mt19937 rng(time(0));

  // Load Tiny Hanabi game
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("tiny_hanabi");

  if (!game) {
    std::cerr << "Failed to load Tiny Hanabi game.\n";
    return -1;
  }

  // Create initial game state
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  std::cout << "Starting Tiny Hanabi game...\n";

  // Main game loop
  while (!state->IsTerminal()) {
    std::cout << "\nCurrent State Representation:\n"
              << state->ToString() << "\n";
    std::cout << "Current player ID: " << state->CurrentPlayer() << "\n";

    // Check if it's a chance node
    if (state->IsChanceNode()) {
      // Print and handle chance node
      std::cout << "This is a chance node. Possible outcomes:\n";
      auto outcomes = state->ChanceOutcomes();
      for (const auto &outcome : outcomes) {
        std::cout << "  Outcome " << outcome.first << " with probability "
                  << outcome.second << "\n";
      }
      std::vector<double> probabilities;
      for (const auto &outcome : outcomes) {
        probabilities.push_back(outcome.second);
      }
      std::discrete_distribution<> dist(probabilities.begin(),
                                        probabilities.end());
      int index = dist(rng);
      auto action = outcomes[index].first;
      std::cout << "Selected chance action: " << action << "\n";
      state->ApplyAction(action);
    } else {
      // Current player
      open_spiel::Player player = state->CurrentPlayer();

      // Print information state and observation state
      PrintInformationState(*state, player);
      PrintObservationState(*state, player);

      // Print legal actions
      auto actions = state->LegalActions(player);
      PrintLegalActions(*state, player, actions);

      // Randomly select an action
      std::uniform_int_distribution<> dis(0, actions.size() - 1);
      auto action = actions[dis(rng)];
      std::cout << "Player " << player
                << " chooses action: " << state->ActionToString(player, action)
                << "\n";

      // Apply the action
      state->ApplyAction(action);
    }

    // Print terminal status
    if (state->IsTerminal()) {
      std::cout << "The game is now in a terminal state.\n";
    }
  }

  // Game has ended; print final returns
  auto returns = state->Returns();
  for (open_spiel::Player p = 0; p < game->NumPlayers(); ++p) {
    std::cout << "Final return for player " << p << ": " << returns[p] << "\n";
  }

  return 0;
}
