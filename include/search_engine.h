#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>
#include <atomic>

#include "defs.h"
#include "movegen.h"
#include "position.h"
#include "score.h"

namespace KhaosChess {
struct SearchInfo {
    std::vector<Move> pv;            // Principal variation
    std::uint64_t nodes;             // Number of nodes searched
    std::uint64_t q_nodes;           // Number of quiescence nodes searched
    std::int32_t depth;              // Current search depth
    std::int32_t completed_depth;    // Deepest fully-searched iteration
    Value score;                     // Root score at completed_depth
    std::chrono::milliseconds time;  // Time spent searching
    bool stopped;                    // Whether search was stopped early

    SearchInfo()
        : nodes(0),
          q_nodes(0),
          depth(0),
          completed_depth(0),
          score(0),
          stopped(false) {}
};

class SearchEngine {
   public:
    explicit SearchEngine(Position& pos, std::int32_t id = 0);

    // Search entry point
    Value search(std::int32_t depth, SearchInfo& info);
    // Set maximum time for search (hard limit: abort mid-iteration)
    void set_max_time(std::chrono::milliseconds max_time);

    // Set the soft time budget: once an iteration completes past this point,
    // no new iteration is started. 0 means no soft limit. Only the main
    // worker honours it; helpers run to the depth or hard limit.
    void set_soft_time(std::chrono::milliseconds soft_time);

    // Cap the search at a node budget; 0 means no limit. Node-limited
    // games are immune to timing noise from CPU contention
    void set_max_nodes(std::uint64_t nodes);

    // Ponder mode: search the predicted position on the opponent's clock,
    // ignoring the time limit until told otherwise.
    void set_ponder(bool on);

    // Suppress the per-iteration UCI "info" lines. Data generation runs
    // millions of searches on the main worker and wants none of them reported.
    void set_quiet(bool on);

    // Shared stop flag controls for Lazy SMP
    static void clear_stop();  // lower the flag before a search
    static void stop();        // raise it to halt every thread

    // A pondered guess was confirmed: convert the running infinite ponder
    // search into a normal timed one by arming the deadline from now.
    static void ponderhit(std::chrono::milliseconds budget);

    // Print a UCI "info" line for a completed iteration. Static because it uses
    // no per-engine state, so ThreadPool can emit the final line for a voted
    // best thread that is not the reporting (main) worker.
    static void report_iteration(const SearchInfo& info, std::int32_t depth,
                                 Value score);

    // Zero the history/continuation/countermove tables. Called at construction
    // and on ucinewgame; the per-search wipe is skipped while history is
    // retained across a game's moves.
    void clear_history();

   private:
    Position& pos;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::milliseconds max_time;
    std::chrono::milliseconds soft_time{0};  // 0 = no soft limit
    std::uint64_t max_nodes;  // 0 = no node limit
    bool ponder_ = false;     // searching on the opponent's clock
    bool quiet_ = false;      // suppress "info" reporting

    bool should_stop;
    std::uint64_t time_checks;

    // Absolute stop time shared by every worker, in steady-clock ms; a value
    // of INT64_MAX means "no deadline" (used while pondering). ponderhit()
    // arms it from the UCI thread, so it lives in a static like the stop flag.
    static std::atomic<std::int64_t> deadline_ms;

    // Core search functions
    Value negamax(std::int32_t depth, std::int32_t ply, Value alpha, Value beta,
                  SearchInfo& info, bool can_null = true,
                  std::int32_t num_extensions = 0,
                  Move prev_move = Move::invalid_move());
    Value quiescence(std::int32_t ply, Value alpha, Value beta,
                     SearchInfo& info);

    static std::atomic<bool> abort_search;
    int thread_id;  // 0 = main worker (reports, searches every depth)

    // Utility functions
    bool is_time_up();
    bool is_capture(Move move);
    bool null_move_cuts(std::int32_t depth, std::int32_t ply, Value beta,
                        SearchInfo& info, std::int32_t num_extensions);
    bool should_skip_quiet(std::int32_t depth, std::int32_t moves_searched,
                           Value alpha, Value static_eval);

    void score_moves(ScoredMoves* begin, ScoredMoves* end, std::int32_t ply,
                     Move tt_move = Move::invalid_move(),
                     bool score_quiets = true,
                     Move prev_move = Move::invalid_move());
    void update_quiet_stats(Move move, std::int32_t ply, std::int32_t depth,
                            Move prev_move, Move* searched_quiets,
                            std::int32_t num_quiets);
    void update_pv(Move move, std::int32_t ply);

    Value aspiration_search(std::int32_t depth, Value prev_score,
                            SearchInfo& info);
    Value pvs_search(std::int32_t depth, std::int32_t ply, Value alpha,
                     Value beta, SearchInfo& info, std::int32_t moves_searched,
                     bool in_check, bool is_quiet, std::int32_t num_extensions,
                     Move prev_move);

    std::int32_t lmr_reduction(std::int32_t depth, std::int32_t moves_searched,
                               bool in_check, bool is_quiet);
    std::int32_t calculate_extension_depth(bool in_check,
                                           std::int32_t num_extensions) const;

    // Quiet-move ordering: two killer slots per ply, and a butterfly
    // history table bumped by depth^2 on every quiet beta cutoff
    Move killers[MAX_PLY][2];
    std::int32_t history[BOTH][SQUARE_TOTAL][SQUARE_TOTAL];

    // Continuation history: how well a quiet move did as a reply to the
    // previous move, keyed by (prev piece, prev target) then (this piece,
    // this target). The colored Piece index folds in the side to move.
    std::int16_t continuation_history[PIECE_NB][SQUARE_TOTAL][PIECE_NB]
                                     [SQUARE_TOTAL];

    // Countermove: per side, the quiet move that last produced a beta cutoff
    // in reply to the (from -> to) move the opponent just played
    Move countermove[BOTH][SQUARE_TOTAL][SQUARE_TOTAL];

    // Triangular PV table: pv_table[ply] is the best line found from that
    // ply, ending at pv_length[ply]. Copied into SearchInfo at the root.
    Move pv_table[MAX_PLY][MAX_PLY];
    std::int32_t pv_length[MAX_PLY];
};

}  // namespace KhaosChess
