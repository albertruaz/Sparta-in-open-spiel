# Sparta in-open-spiel


Game: Tiny-Hanabi

Environment: [Open-Spiel](https://github.com/google-deepmind/open_spiel)


Main Agent: Single-Agent search (SearchBot)

BluePrint Agent: SAD model (TorchBot)

## Single-Agent Search 

Train SAD model
```
python python/train_sad_txt.py
```

Build and Run Agent in Tiny-Hanbi
```
g++ -std=c++17 -o ./build/tiny_hanabi_build tiny_hanabi_eval.cc agents/searchBot.cc agents/torchBot.cc -lopen_spiel

./build/tiny_hanabi_build
```