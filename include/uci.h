#pragma once

#include <iosfwd>
#include <string>

#include "position.h"

namespace KhaosChess {
class Position;
class Move;
struct SearchLimits;

// parse user/GUI move string input (e.g. "e7e8q")
extern Move parse_move(std::string move_string, const Position& pos);
extern void parse_position(const char* cmd, Position& pos, InfoListPtr& infos);
// Fills `limits` from a "go" command; returns true if a search should be
// launched ("go perft" is handled inline and returns false).
extern bool parse_go(const char* cmd, Position& pos, SearchLimits& limits);
// Emits the "id name / id author / option ... / uciok" block to `os`.
extern void print_id_and_options(std::ostream& os);
// Applies a "setoption name <id> value <rest>" command line.
extern void handle_setoption(const std::string& line, Position& pos);
extern void uci_loop();
}  // namespace KhaosChess
