#pragma once

#include <cstdint>
#include <string>

namespace KhaosChess {
namespace datagen {

struct Options {
    std::uint64_t games = 1000;      // self-play games to run
    std::uint64_t nodes = 5000;      // fixed node budget per move
    std::uint64_t seed = 1;          // PRNG seed; vary it per shard
    std::int32_t random_plies = 8;   // random opening plies before scoring
    std::int32_t max_plies = 400;    // adjudicate as a draw past this
    std::string out = "data.txt";    // output shard path
    std::int32_t report_every = 50;  // progress lines to stderr; 0 disables
};

// Play `games` self-play games and append one line per retained position to
// `opt.out`:
//
//     <fen> | <score> | <result>
//
// `score` is the engine's search score in its own units (not centipawns),
// always from WHITE's point of view so the trainer never has to guess.
// `result` is the game result, also from White: 1.0 / 0.5 / 0.0.
//
// Returns false if the output file cannot be opened.
bool run(const Options& opt);

// Parse a UCI-style "datagen games 1000 nodes 5000 out shard.txt ..." tail.
// Unknown tokens are ignored so the command stays forgiving.
Options parse_options(const char* cmd);

}  // namespace datagen
}  // namespace KhaosChess
