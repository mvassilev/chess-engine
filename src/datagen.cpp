#include "datagen.h"

#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "eval.h"
#include "movegen.h"
#include "position.h"
#include "search_engine.h"
#include "tt.h"

namespace KhaosChess {
namespace datagen {

namespace {

// Deliberately single-threaded. Parallelism comes from launching one process
// per core with different --seed values, each writing its own shard: that keeps
// the shared transposition table out of the picture entirely, so games never
// contaminate each other's scores. A threaded generator sharing the global TT
// would produce subtly noisier labels for no real gain.

constexpr Value MATE_THRESHOLD = VALUE_MATE - MAX_PLY;

// Reject a random opening that already decided the game. Expressed in engine
// eval units, where a tuned pawn is ~410, so this is roughly four pawns.
constexpr Value OPENING_BALANCE_LIMIT = 4 * 410;

// True when the side to move is in check. Mirrors the expression the search
// itself uses, rather than re-deriving it a different way.
bool in_check(const Position& pos) {
    const Color stm = pos.side_to_move();
    return (pos.get_attackers_to(pos.square<KING>(stm)) &
            pos.get_pieces_bb(~stm)) != 0;
}

std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One scored position awaiting the game result, which is only known at the end.
struct Sample {
    std::string fen;
    Value white_score;
};

enum class GameResult { WhiteWin, Draw, BlackWin };

const char* result_string(GameResult r) {
    switch (r) {
        case GameResult::WhiteWin:
            return "1.0";
        case GameResult::BlackWin:
            return "0.0";
        default:
            return "0.5";
    }
}

// Read a long option value out of a UCI-style token stream.
std::uint64_t token_u64(const char* cmd, const char* key, std::uint64_t fallback) {
    const char* at = strstr(cmd, key);
    if (at == nullptr) {
        return fallback;
    }
    return static_cast<std::uint64_t>(std::strtoull(at + strlen(key), nullptr, 10));
}

std::int32_t token_i32(const char* cmd, const char* key, std::int32_t fallback) {
    const char* at = strstr(cmd, key);
    if (at == nullptr) {
        return fallback;
    }
    return static_cast<std::int32_t>(std::strtol(at + strlen(key), nullptr, 10));
}

std::string token_string(const char* cmd, const char* key,
                         const std::string& fallback) {
    const char* at = strstr(cmd, key);
    if (at == nullptr) {
        return fallback;
    }

    at += strlen(key);
    while (*at == ' ') {
        ++at;
    }

    std::string value;
    while (*at != '\0' && *at != ' ' && *at != '\n' && *at != '\r') {
        value.push_back(*at++);
    }

    return value.empty() ? fallback : value;
}

}  // namespace

Options parse_options(const char* cmd) {
    Options opt;
    opt.games = token_u64(cmd, "games", opt.games);
    opt.nodes = token_u64(cmd, "nodes", opt.nodes);
    opt.seed = token_u64(cmd, "seed", opt.seed);
    opt.random_plies = token_i32(cmd, "randply", opt.random_plies);
    opt.max_plies = token_i32(cmd, "maxply", opt.max_plies);
    opt.report_every = token_i32(cmd, "report", opt.report_every);
    opt.out = token_string(cmd, "out", opt.out);
    return opt;
}

bool run(const Options& opt) {
    std::ofstream out(opt.out, std::ios::app);
    if (!out) {
        std::cerr << "datagen: cannot open '" << opt.out << "' for writing\n";
        return false;
    }

    std::uint64_t rng_state = opt.seed;
    std::uint64_t positions_written = 0;
    std::uint64_t games_played = 0;

    // One engine and board reused across games: cheaper than reconstructing the
    // history tables every game, and set() resets the board state anyway.
    Position pos;
    SearchEngine engine(pos, 0);
    engine.set_quiet(true);
    engine.set_max_nodes(opt.nodes);
    engine.set_max_time(std::chrono::milliseconds(24 * 60 * 60 * 1000));

    std::vector<Sample> samples;

    for (std::uint64_t game = 0; game < opt.games; ++game) {
        // Fresh TT and history per game so nothing leaks across games.
        tt::TT.clear();
        engine.clear_history();

        std::deque<MoveInfo> infos(1);
        pos.set(START_FEN, &infos.back());

        samples.clear();
        GameResult result = GameResult::Draw;
        bool aborted = false;

        // Random opening walk for diversity. These plies are never scored: the
        // positions they reach are often nonsense, and labelling them would
        // teach the net to evaluate lines no search would ever visit.
        bool opening_ok = true;
        for (std::int32_t ply = 0; ply < opt.random_plies; ++ply) {
            MoveList<GT_LEGAL> moves(pos);
            if (moves.size() == 0) {
                opening_ok = false;  // random walk stumbled into a terminal node
                break;
            }

            const std::size_t pick =
                splitmix64(rng_state) % static_cast<std::size_t>(moves.size());
            infos.emplace_back();
            pos.do_move(*(moves.begin() + pick), infos.back());
        }

        if (!opening_ok) {
            continue;
        }

        // Reject an opening that already handed someone a decisive advantage;
        // otherwise the dataset skews toward lost positions from ply one.
        {
            SearchInfo probe;
            // A node-limited search signals exhaustion through the shared abort
            // flag, and nothing lowers it again; without this every search after
            // the first would return instantly with an empty PV. The UCI "go"
            // path clears it the same way before each search.
            SearchEngine::clear_stop();
            engine.search(64, probe);
            if (probe.pv.empty()) {
                continue;
            }
            const Value stm_score = probe.score;
            const Value white_score =
                pos.side_to_move() == WHITE ? stm_score : -stm_score;
            if (std::abs(white_score) > OPENING_BALANCE_LIMIT) {
                continue;
            }
        }

        for (std::int32_t ply = 0; ply < opt.max_plies; ++ply) {
            MoveList<GT_LEGAL> moves(pos);

            if (moves.size() == 0) {
                // Checkmate is a loss for the side to move; stalemate a draw.
                if (in_check(pos)) {
                    result = pos.side_to_move() == WHITE ? GameResult::BlackWin
                                                         : GameResult::WhiteWin;
                } else {
                    result = GameResult::Draw;
                }
                break;
            }

            if (pos.is_draw()) {
                result = GameResult::Draw;
                break;
            }

            SearchInfo info;
            SearchEngine::clear_stop();  // see the probe search above
            engine.search(64, info);

            if (info.pv.empty()) {
                aborted = true;
                break;
            }

            const Move best = info.pv[0];
            const Value stm_score = info.score;
            const Value white_score =
                pos.side_to_move() == WHITE ? stm_score : -stm_score;

            // Resign / declare once the score is overwhelming: playing out a
            // decided game just burns nodes on positions the net gains nothing
            // from, and the result label is already certain.
            if (std::abs(white_score) >= MATE_THRESHOLD) {
                result = white_score > 0 ? GameResult::WhiteWin
                                         : GameResult::BlackWin;
                break;
            }

            // Retain only quiet, non-tactical positions. A static evaluation
            // trained on positions with a hanging queen learns to predict the
            // search, not the position -- these are exactly the positions the
            // engine's own quiescence search exists to resolve.
            const bool best_is_capture = !pos.is_empty(best.target_square()) ||
                                         best.move_type() == MT_EN_PASSANT;
            const bool best_is_promotion = best.move_type() == MT_PROMOTION;

            if (!in_check(pos) && !best_is_capture && !best_is_promotion) {
                samples.push_back(Sample{pos.get_fen(), white_score});
            }

            infos.emplace_back();
            pos.do_move(best, infos.back());
        }

        if (aborted) {
            continue;
        }

        for (const Sample& s : samples) {
            out << s.fen << " | " << s.white_score << " | "
                << result_string(result) << "\n";
        }
        positions_written += samples.size();
        ++games_played;

        if ((opt.report_every > 0) &&
            ((games_played % static_cast<std::uint64_t>(opt.report_every)) == 0)) {
            out.flush();
            std::cerr << "datagen[" << opt.seed << "]: " << games_played << "/"
                      << opt.games << " games, " << positions_written
                      << " positions\n"
                      << std::flush;
        }
    }

    out.flush();
    std::cerr << "datagen[" << opt.seed << "]: done, " << games_played
              << " games, " << positions_written << " positions -> " << opt.out
              << "\n"
              << std::flush;

    return true;
}

}  // namespace datagen
}  // namespace KhaosChess
