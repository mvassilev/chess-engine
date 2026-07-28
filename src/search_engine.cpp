#include "search_engine.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "thread.h"
#include "tt.h"

namespace KhaosChess {
// Move-ordering tiers
constexpr std::int32_t HISTORY_MAX = 16'000;
constexpr std::int32_t CONTINUATION_HISTORY_MAX = 16'000;
constexpr std::int32_t QUIET_SCORE_MAX = HISTORY_MAX + CONTINUATION_HISTORY_MAX;
constexpr std::int32_t SCORE_COUNTERMOVE = QUIET_SCORE_MAX + 1;

constexpr std::int32_t SCORE_KILLER_SECONDARY = SCORE_COUNTERMOVE + 1;
constexpr std::int32_t SCORE_KILLER_PRIMARY = SCORE_KILLER_SECONDARY + 1;
constexpr std::int32_t SCORE_CAPTURE = SCORE_KILLER_PRIMARY + 1;

// Kill-switch (dormant, pending an A/B match). When true, the history/
// continuation/countermove tables are retained across the moves of a game
// (Stockfish-style) instead of wiped every search, so move ordering carries
// context between moves; cleared on ucinewgame. Left false so the committed
// binary matches 2.17.0 — flip to true, rebuild, and measure vs the frozen
// base to bank it.
constexpr bool HISTORY_RETENTION = false;

// A capture's ceiling: queen victim + promotion bonus + promoted queen
constexpr std::int32_t SCORE_PROMOTION_BONUS = 900;
constexpr std::int32_t CAPTURE_CEILING =
    SCORE_CAPTURE + (2 * Material().piece_value[QUEEN].mg) +
    SCORE_PROMOTION_BONUS;

constexpr std::int32_t SCORE_TT_MOVE = CAPTURE_CEILING + 1;
constexpr std::int32_t SCORE_SKIP = -1'000'000;

// Bad-capture tier
constexpr std::int32_t SCORE_BAD_CAPTURE = -20'000;

// Null-move pruning
constexpr std::int32_t NULL_MOVE_REDUCTION = 3;
constexpr std::int32_t NULL_MOVE_MIN_DEPTH = 3;

// Late move reductions
constexpr std::int32_t LMR_MIN_MOVES = 4;
constexpr std::int32_t LMR_MIN_DEPTH = 3;
static std::int32_t LMR_TABLE[MAX_PLY][MAX_MOVES];

// Late move pruning
constexpr std::int32_t LMP_MAX_DEPTH = 3;
constexpr std::int32_t LMP_BASE = 3;

// Reverse futility pruning
constexpr std::int32_t RFP_MAX_DEPTH = 6;
constexpr std::int32_t RFP_MARGIN = 120;

// Futility pruning
constexpr std::int32_t FUTILITY_MAX_DEPTH = 2;
constexpr std::int32_t FUTILITY_MARGIN = 150;

// Poll the clock once per this many nodes; must stay a power of two
constexpr std::uint64_t TIME_CHECK_INTERVAL = 2048;
constexpr Value ASPIRATION_DELTA = 25;
constexpr std::int32_t ASPIRATION_MIN_DEPTH = 4;

// By how much we want to extend the depth search
constexpr std::int32_t MAX_EXTENSION = 16;

constexpr std::int32_t IIR_MIN_DEPTH = 4;

static Value score_to_tt(Value score, std::int32_t ply) {
    if (score >= VALUE_MATE - MAX_PLY) {
        return score + ply;
    }
    if (score <= -VALUE_MATE + MAX_PLY) {
        return score - ply;
    }
    return score;
}

static Value score_from_tt(Value score, std::int32_t ply) {
    if (score >= VALUE_MATE - MAX_PLY) {
        return score - ply;
    }
    if (score <= -VALUE_MATE + MAX_PLY) {
        return score + ply;
    }
    return score;
}

// Fill the late-move-reduction table once. Reduction grows with the log of
// both depth and move number, so late quiet moves at high depth are cut most.
static void init_lmr_table() {
    for (std::int32_t d = 1; d < MAX_PLY; ++d) {
        for (std::int32_t m = 1; m < MAX_MOVES; ++m) {
            LMR_TABLE[d][m] = static_cast<std::int32_t>(
                0.75 + std::log(static_cast<double>(d)) *
                           std::log(static_cast<double>(m)) / 2.25);
        }
    }
}

// Gravity update: nudge an entry toward +/- bonus while decaying stale values,
// so it self-limits within [-max, +max]. Unlike a hard cap it never saturates,
// which keeps the table responsive as the search revises what a move is worth.
template <typename T>
static void apply_gravity(T& entry, std::int32_t bonus, std::int32_t max) {
    entry += static_cast<T>(bonus - entry * std::abs(bonus) / max);
}

std::atomic<bool> SearchEngine::abort_search{false};
std::atomic<std::int64_t> SearchEngine::deadline_ms{
    std::numeric_limits<std::int64_t>::max()};

// Steady-clock "now" in milliseconds, the unit the shared deadline is kept in.
static std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void SearchEngine::clear_stop() {
    abort_search.store(false, std::memory_order_relaxed);
}

void SearchEngine::stop() {
    abort_search.store(true, std::memory_order_relaxed);
}

void SearchEngine::ponderhit(std::chrono::milliseconds budget) {
    // The guessed move was played: give the already-running ponder search a
    // real deadline measured from now, turning it into a normal timed search.
    deadline_ms.store(now_ms() + budget.count(), std::memory_order_relaxed);
}

SearchEngine::SearchEngine(Position& pos, std::int32_t id)
    : pos(pos),
      max_time(std::chrono::milliseconds(3000)),  // Default 3 seconds
      max_nodes(0),
      should_stop(false),
      time_checks(0),
      thread_id(id) {
    [[maybe_unused]] static bool lmr_ready = (init_lmr_table(), true);
    // Retention skips the per-search wipe, so the tables must start zeroed.
    clear_history();
}

void SearchEngine::clear_history() {
    std::memset(history, 0, sizeof(history));
    std::memset(continuation_history, 0, sizeof(continuation_history));
    std::memset(countermove, 0, sizeof(countermove));  // 0 == invalid move
}

Value SearchEngine::search(std::int32_t depth, SearchInfo& info) {
    // Reset search info
    info = SearchInfo();
    info.depth = depth;
    start_time = std::chrono::high_resolution_clock::now();
    should_stop = false;
    time_checks = 0;

    // Arm the shared deadline. While pondering there is none (ponderhit() sets
    // it later); otherwise the hard limit applies from now. Every worker sets
    // it to ~the same value, which is fine.
    deadline_ms.store(ponder_ ? std::numeric_limits<std::int64_t>::max()
                              : now_ms() + max_time.count(),
                      std::memory_order_relaxed);

    // Killers are ply-indexed, so they are always reset per search; the history
    // tables are retained across the game's moves unless retention is disabled.
    for (std::int32_t i = 0; i < MAX_PLY; ++i) {
        killers[i][0] = killers[i][1] = Move::invalid_move();
    }

    if constexpr (!HISTORY_RETENTION) {
        clear_history();
    }

    Value best_score = -VALUE_INFINITE;

    // Iterative deepening
    for (std::int32_t current_depth = 1; current_depth <= depth;
         ++current_depth) {
        // Lazy SMP staggering: helper threads skip a ply on a per-thread
        // schedule, so they run at different depths than the main thread
        // and fill the shared TT with a wider spread of results. Helpers
        // still search the final target depth.
        if ((thread_id > 0) && (current_depth < depth) &&
            (((current_depth + thread_id) % 2) == 0)) {
            continue;
        }

        Value score = aspiration_search(current_depth, best_score, info);

        if (should_stop && (current_depth > 1)) {
            break;
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        info.time = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time);

        if (!should_stop) {
            info.pv.assign(pv_table[0], pv_table[0] + pv_length[0]);
            best_score = score;
            info.completed_depth = current_depth;
            info.score = score;

            if (thread_id == 0) {
                report_iteration(info, current_depth, score);
            }
        }

        // Soft-limit cutoff (main worker only): the next iteration usually
        // costs more than every previous one combined, so once we have spent
        // the soft budget there is no point starting another one. The hard
        // limit still aborts an iteration already in flight.
        if ((thread_id == 0) && !ponder_ && (soft_time.count() > 0) &&
            (info.time >= soft_time)) {
            break;
        }
    }

    // A ponder search that exhausts its depth budget before the opponent moves
    // must not emit a move yet: UCI forbids reporting bestmove during pondering
    // until ponderhit or stop. Hold here until ponderhit arms the deadline or
    // stop raises the abort flag.
    while (ponder_ && !should_stop &&
           (deadline_ms.load(std::memory_order_relaxed) ==
            std::numeric_limits<std::int64_t>::max()) &&
           !abort_search.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return best_score;
}

Value SearchEngine::negamax(std::int32_t depth, std::int32_t ply, Value alpha,
                            Value beta, SearchInfo& info, bool can_null,
                            std::int32_t num_extensions, Move prev_move) {
    pv_length[ply] = ply;

    // Check if we're out of time
    if (is_time_up()) {
        info.stopped = should_stop = true;

        return Scorer<SC_ALL>().get_score(pos);
    }

    info.nodes++;

    // Draw by repetition or fifty-move rule
    if ((ply > 0) && pos.is_draw()) {
        return VALUE_DRAW;
    }

    // Mate distance pruning: cap the window by the fastest mate possible
    // from this ply; a collapsed window means nothing here can matter
    if (ply > 0) {
        alpha = std::max(alpha, -VALUE_MATE + ply);
        beta = std::min(beta, VALUE_MATE - (ply + 1));

        if (alpha >= beta) {
            return alpha;
        }
    }

    Color stm = pos.side_to_move();
    bool in_check =
        pos.get_attackers_to(pos.square<KING>(stm)) & pos.get_pieces_bb(~stm);

    // Check extension: forcing positions get one extra ply
    std::int32_t extension =
        calculate_extension_depth(in_check, num_extensions);
    depth += extension;
    num_extensions += extension;

    bool is_pv = (beta - alpha) > 1;

    // Transposition table probe
    bool is_tt_hit;
    tt::TTEntry* tte = tt::TT.probe(pos.key(), is_tt_hit);

    // PV nodes are exempt from the cutoff: returning the TT score here skips the
    // move loop, so update_pv never runs and the triangular PV collapses at this
    // ply (the "one-move PV" seen when re-searching a warm TT). The whole PV path
    // is PV nodes, so skipping their cutoffs rebuilds the full line for a trivial
    // node cost.
    if (is_tt_hit && (ply > 0) && !is_pv && (tte->depth >= depth)) {
        Value tt_score = score_from_tt(tte->score, ply);

        if ((tte->flag() == tt::Flag::F_EXACT) ||
            ((tte->flag() == tt::Flag::F_LOWER_BOUND) && (tt_score >= beta)) ||
            ((tte->flag() == tt::Flag::F_UPPER_BOUND) && (tt_score <= alpha))) {
            return tt_score;
        }
    }

    // Internal iterative reduction: no TT move at high depth means the node
    // is probably unimportant; search it shallower and let the TT fill in
    if ((depth >= IIR_MIN_DEPTH) &&
        (!is_tt_hit || (tte->move == Move::invalid_move()))) {
        depth--;
    }

    // At depth 0, switch to quiescence search to evade horizon effect
    if (depth <= 0) {
        return quiescence(ply, alpha, beta, info);
    }

    Value static_eval =
        in_check ? -VALUE_INFINITE : Scorer<SC_ALL>().get_score(pos);

    // Reverse futility pruning
    if (!is_pv && !in_check && (depth <= RFP_MAX_DEPTH) &&
        (std::abs(beta) < VALUE_MATE - MAX_PLY) &&
        (static_eval - RFP_MARGIN * depth >= beta)) {
        return beta;
    }

    // Null-move pruning
    if (can_null && !in_check && (ply > 0) && (depth >= NULL_MOVE_MIN_DEPTH) &&
        (std::abs(beta) < (VALUE_MATE - MAX_PLY)) &&
        (pos.count(stm, KNIGHT, BISHOP, ROOK, QUEEN) > 0)) {
        if (null_move_cuts(depth, ply, beta, info, num_extensions)) {
            return beta;
        }
        if (should_stop) {
            return 0;
        }
    }

    // Generate legal moves
    MoveList<GT_LEGAL> moves(pos);

    // If no legal moves, check for checkmate or stalemate
    if (moves.size() == 0) {
        return in_check ? -VALUE_MATE + ply
                        : VALUE_DRAW;  // Checkmate or Stalemate
    }

    // Score moves for better ordering
    score_moves(moves.begin(), moves.end(), ply,
                is_tt_hit ? tte->move : Move::invalid_move(), true, prev_move);

    // Local PV line for this level
    bool found_pv = false;
    Move best_move = Move::invalid_move();
    Move searched_quiets[MAX_MOVES];
    std::int32_t moves_searched = 0;
    std::int32_t num_quiets = 0;

    // Loop through all moves
    for (const ScoredMoves& scored_move : moves) {
        Move move = scored_move;
        MoveInfo move_info;

        // Read before doing a move, while any victim still sits on the target
        bool is_quiet = !is_capture(move) && (move.move_type() != MT_PROMOTION);

        // Forward pruning: LMP and futility
        if (is_quiet && !in_check &&
            should_skip_quiet(depth, moves_searched, alpha, static_eval)) {
            continue;
        }

        // Make the move
        pos.do_move(move, move_info);

        Value score = pvs_search(depth, ply, alpha, beta, info, moves_searched,
                                 in_check, is_quiet, num_extensions, move);

        moves_searched++;

        // Undo the move
        pos.undo_move(move);

        // Check if search was stopped
        if (should_stop) {
            return 0;
        }

        // Beta cutoff (fail-high)
        if (score >= beta) {
            // Quiet cutoff
            if (is_quiet) {
                update_quiet_stats(move, ply, depth, prev_move, searched_quiets,
                                   num_quiets);
            }

            tt::TT.store(pos.key(), score_to_tt(beta, ply), depth,
                         tt::Flag::F_LOWER_BOUND, move);
            return beta;
        }

        // Better move found
        if (score > alpha) {
            alpha = score;
            found_pv = true;
            best_move = move;

            update_pv(move, ply);
        }

        if (is_quiet && num_quiets < MAX_MOVES) {
            searched_quiets[num_quiets++] = move;
        }
    }

    tt::TT.store(pos.key(), score_to_tt(alpha, ply), depth,
                 found_pv ? tt::Flag::F_EXACT : tt::Flag::F_UPPER_BOUND,
                 best_move);

    return alpha;
}

Value SearchEngine::quiescence(std::int32_t ply, Value alpha, Value beta,
                               SearchInfo& info) {
    // Check if we're out of time
    if (is_time_up()) {
        info.stopped = should_stop = true;

        return alpha;  // discarded by callers once should_stop is set
    }

    // Increment quiescence node counter
    info.q_nodes++;

    if (pos.is_draw()) {
        return VALUE_DRAW;
    }

    // Hard safety net for any other pathologically long sequence
    if (ply >= MAX_PLY) {
        return Scorer<SC_ALL>().get_score(pos);
    }

    // Transposition table probe; any stored entry beats a depth-0 search
    bool is_tt_hit = false;
    tt::TTEntry* tte = nullptr;

    tte = tt::TT.probe(pos.key(), is_tt_hit);

    if (is_tt_hit) {
        Value tt_score = score_from_tt(tte->score, ply);

        if ((tte->flag() == tt::Flag::F_EXACT) ||
            ((tte->flag() == tt::Flag::F_LOWER_BOUND) && (tt_score >= beta)) ||
            ((tte->flag() == tt::Flag::F_UPPER_BOUND) && (tt_score <= alpha))) {
            return tt_score;
        }
    }

    Value orig_alpha = alpha;
    Move best_move = Move::invalid_move();

    Color stm = pos.side_to_move();
    bool in_check =
        pos.get_attackers_to(pos.square<KING>(stm)) & pos.get_pieces_bb(~stm);

    if (!in_check) {
        Value stand_pat = Scorer<SC_ALL>().get_score(pos);

        // Stand-pat cutoff
        if (stand_pat >= beta) {
            tt::TT.store(pos.key(), score_to_tt(beta, ply), 0,
                         tt::Flag::F_LOWER_BOUND, Move::invalid_move());
            return beta;
        }

        // Update alpha if the static eval is better
        if (stand_pat > alpha) {
            alpha = stand_pat;
        }
    }

    // Generate only capture moves
    MoveList<GT_LEGAL> legals(pos);

    // No legal moves while in check is checkmate
    if (in_check && (legals.size() == 0)) {
        return -VALUE_MATE + ply;
    }

    // Score legal moves for better ordering; a quiet TT move is harmless,
    // the capture filter below skips it anyway
    score_moves(legals.begin(), legals.end(), ply,
                is_tt_hit ? tte->move : Move::invalid_move(),
                false);

    // Loop through capture moves
    for (const ScoredMoves& scored_move : legals) {
        if (!in_check && !is_capture(scored_move)) {
            continue;
        }

        Move move = scored_move;

        // SEE pruning
        if (!in_check && !pos.see_ge(move, 0)) {
            continue;
        }

        MoveInfo move_info;

        // Make the move
        pos.do_move(move, move_info);

        // Recursively search with negated bounds
        Value score = -quiescence(ply + 1, -beta, -alpha, info);

        // Undo the move
        pos.undo_move(move);

        // Check if search was stopped
        if (should_stop) {
            return 0;
        }

        // Beta cutoff (fail-high)
        if (score >= beta) {
            tt::TT.store(pos.key(), score_to_tt(beta, ply), 0,
                         tt::Flag::F_LOWER_BOUND, move);
            return beta;
        }

        // Better move found
        if (score > alpha) {
            alpha = score;
            best_move = move;
        }
    }

    tt::Flag flag =
        (alpha > orig_alpha) ? tt::Flag::F_EXACT : tt::Flag::F_UPPER_BOUND;
    tt::TT.store(pos.key(), score_to_tt(alpha, ply), 0, flag, best_move);

    return alpha;
}

void SearchEngine::score_moves(ScoredMoves* begin, ScoredMoves* end,
                               std::int32_t ply, Move tt_move,
                               bool score_quiets, Move prev_move) {
    // The previous move is fixed for the whole node, so resolve its inner
    // continuation-history table once, before the move loop.
    std::int16_t (*continuation)[SQUARE_TOTAL] = nullptr;
    if (prev_move != Move::invalid_move()) {
        // The previous move is already made: its piece sits on its target.
        Piece prev_piece = pos.get_piece_on(prev_move.target_square());
        continuation =
            continuation_history[prev_piece][prev_move.target_square()];
    }

    // Simple move ordering with MVV-LVA
    for (ScoredMoves* it = begin; it != end; ++it) {
        Move move = *it;

        if (move == tt_move) {
            it->score = SCORE_TT_MOVE;
            continue;
        }

        // Captures and promotions
        if (is_capture(move) || (move.move_type() == MT_PROMOTION)) {
            PieceType victim =
                type_of_piece(pos.get_piece_on(move.target_square()));
            PieceType aggressor =
                type_of_piece(pos.get_piece_on(move.source_square()));

            bool is_winning = pos.see_ge(move, 0);

            it->score = (is_winning ? SCORE_CAPTURE : SCORE_BAD_CAPTURE) +
                        MATERIAL_SCORES.piece_value[victim].mg -
                        MATERIAL_SCORES.piece_value[aggressor].mg / 10;

            if (move.move_type() == MT_PROMOTION) {
                it->score += SCORE_PROMOTION_BONUS +
                             MATERIAL_SCORES.piece_value[move.promoted()].mg;
            }
        } else if (score_quiets) {
            if (move == killers[ply][0]) {
                it->score = SCORE_KILLER_PRIMARY;
            } else if (move == killers[ply][1]) {
                it->score = SCORE_KILLER_SECONDARY;
            } else if ((prev_move != Move::invalid_move()) &&
                       (move == countermove[pos.side_to_move()]
                                           [prev_move.source_square()]
                                           [prev_move.target_square()])) {
                it->score = SCORE_COUNTERMOVE;
            } else {
                Piece current = pos.get_piece_on(move.source_square());

                std::int32_t ch =
                    (continuation != nullptr)
                        ? continuation[current][move.target_square()]
                        : 0;
                it->score = history[pos.side_to_move()][move.source_square()]
                                   [move.target_square()] +
                            ch;
            }
        } else {
            // Quiescence discards quiets (except evasions)
            it->score = SCORE_SKIP;
        }
    }

    std::sort(begin, end, [](auto& a, auto& b) { return a.score > b.score; });
}

bool SearchEngine::is_capture(Move move) {
    return (pos.get_piece_on(move.target_square()) != NO_PIECE) ||
           (move.move_type() == MT_EN_PASSANT);
}

bool SearchEngine::is_time_up() {
    if (should_stop) {
        return true;
    }

    if (abort_search.load(std::memory_order_relaxed)) {
        return should_stop = true;
    }

    ++time_checks;

    if ((max_nodes != 0) && (time_checks >= max_nodes)) {
        abort_search.store(true, std::memory_order_relaxed);
        return should_stop = true;
    }

    // The clock read is costly; poll it once every (time check interval - 1)
    // calls
    if ((time_checks & (TIME_CHECK_INTERVAL - 1)) != 0) {
        return false;
    }

    // Stop once we pass the shared deadline. It is INT64_MAX while pondering,
    // so this never fires until ponderhit() arms it.
    if (now_ms() >= deadline_ms.load(std::memory_order_relaxed)) {
        abort_search.store(true, std::memory_order_relaxed);
        return should_stop = true;
    }

    return false;
}

void SearchEngine::set_max_time(std::chrono::milliseconds max_time) {
    this->max_time = max_time;
}

void SearchEngine::set_soft_time(std::chrono::milliseconds soft_time) {
    this->soft_time = soft_time;
}

void SearchEngine::set_max_nodes(std::uint64_t nodes) {
    this->max_nodes = nodes;
}

void SearchEngine::set_ponder(bool on) {
    ponder_ = on;
}

// Prepend `move` to the child's line: pv_table[ply] becomes move +
// pv_table[ply+1]
void SearchEngine::update_pv(Move move, std::int32_t ply) {
    pv_table[ply][ply] = move;

    for (std::int32_t i = ply + 1; i < pv_length[ply + 1]; ++i) {
        pv_table[ply][i] = pv_table[ply + 1][i];
    }

    pv_length[ply] = pv_length[ply + 1];
}

// A quiet move refuted this node
void SearchEngine::update_quiet_stats(Move move, std::int32_t ply,
                                      std::int32_t depth, Move prev_move,
                                      Move* searched_quiets,
                                      std::int32_t num_quiets) {
    if (move != killers[ply][0]) {
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = move;
    }

    if (prev_move != Move::invalid_move()) {
        countermove[pos.side_to_move()][prev_move.source_square()]
                   [prev_move.target_square()] = move;
    }

    std::int32_t bonus = std::min(depth * depth, HISTORY_MAX);
    Color stm = pos.side_to_move();

    // Resolve the previous move's continuation slot once for the whole update.
    bool has_prev = (prev_move != Move::invalid_move());
    Piece prev_piece = NO_PIECE;
    Square prev_target = NONE;
    if (has_prev) {
        prev_target = prev_move.target_square();
        prev_piece = pos.get_piece_on(prev_target);
    }

    // Reward the quiet that produced the cutoff, in both tables.
    apply_gravity(history[stm][move.source_square()][move.target_square()],
                  bonus, HISTORY_MAX);
    if (has_prev) {
        Piece curr_piece = pos.get_piece_on(move.source_square());
        apply_gravity(continuation_history[prev_piece][prev_target][curr_piece]
                                          [move.target_square()],
                      bonus, CONTINUATION_HISTORY_MAX);
    }

    // Penalize the quiets that were tried first and failed low, in both tables.
    for (std::int32_t i = 0; i < num_quiets; ++i) {
        Move q_move = searched_quiets[i];
        Square q_source = q_move.source_square();
        Square q_target = q_move.target_square();

        apply_gravity(history[stm][q_source][q_target], -bonus, HISTORY_MAX);
        if (has_prev) {
            Piece quiet_piece = pos.get_piece_on(q_source);
            apply_gravity(continuation_history[prev_piece][prev_target]
                                              [quiet_piece][q_target],
                          -bonus, CONTINUATION_HISTORY_MAX);
        }
    }
}

std::int32_t SearchEngine::lmr_reduction(std::int32_t depth,
                                         std::int32_t moves_searched,
                                         bool in_check, bool is_quiet) {
    if ((moves_searched < LMR_MIN_MOVES) || (depth < LMR_MIN_DEPTH) ||
        in_check || !is_quiet) {
        return 0;
    }

    if (pos.get_attackers_to(pos.square<KING>(pos.side_to_move())) &
        pos.get_pieces_bb(~pos.side_to_move())) {
        return 0;  // the move gives check: never reduce it
    }

    std::int32_t reduction =
        LMR_TABLE[std::min<std::int32_t>(depth, MAX_PLY - 1)]
                 [std::min<std::int32_t>(moves_searched, MAX_MOVES - 1)];

    return std::min(reduction, depth - 2);  // keep the scout at depth >= 1
}

bool SearchEngine::should_skip_quiet(std::int32_t depth,
                                     std::int32_t moves_searched, Value alpha,
                                     Value static_eval) {
    if ((depth <= LMP_MAX_DEPTH) &&
        (moves_searched >= LMP_BASE + depth * depth)) {
        return true;
    }

    return (moves_searched > 0) && (depth <= FUTILITY_MAX_DEPTH) &&
           (std::abs(alpha) < VALUE_MATE - MAX_PLY) &&
           (static_eval + FUTILITY_MARGIN * depth <= alpha);
}

Value SearchEngine::pvs_search(std::int32_t depth, std::int32_t ply,
                               Value alpha, Value beta, SearchInfo& info,
                               std::int32_t moves_searched, bool in_check,
                               bool is_quiet, std::int32_t num_extensions,
                               Move prev_move) {
    if (moves_searched == 0) {
        return -negamax(depth - 1, ply + 1, -beta, -alpha, info, true,
                        num_extensions, prev_move);
    }

    std::int32_t reduction =
        lmr_reduction(depth, moves_searched, in_check, is_quiet);

    // Null-window
    Value score = -negamax(depth - 1 - reduction, ply + 1, -alpha - 1, -alpha,
                           info, true, num_extensions, prev_move);

    if ((score > alpha) && (reduction > 0)) {
        score = -negamax(depth - 1, ply + 1, -alpha - 1, -alpha, info, true,
                         num_extensions, prev_move);
    }

    if ((score > alpha) && (score < beta)) {
        score = -negamax(depth - 1, ply + 1, -beta, -alpha, info, true,
                         num_extensions, prev_move);
    }

    return score;
}

bool SearchEngine::null_move_cuts(std::int32_t depth, std::int32_t ply,
                                  Value beta, SearchInfo& info,
                                  std::int32_t num_extensions) {
    MoveInfo null_info;
    pos.do_null_move(null_info);

    Value null_score = -negamax(depth - 1 - NULL_MOVE_REDUCTION, ply + 1, -beta,
                                -beta + 1, info, false, num_extensions);

    pos.undo_null_move();

    return !should_stop && (null_score >= beta);
}

Value SearchEngine::aspiration_search(std::int32_t depth, Value prev_score,
                                      SearchInfo& info) {
    Value delta = ASPIRATION_DELTA;
    Value alpha = -VALUE_INFINITE;
    Value beta = VALUE_INFINITE;

    if (depth >= ASPIRATION_MIN_DEPTH) {
        alpha = std::max(prev_score - delta, -VALUE_INFINITE);
        beta = std::min(prev_score + delta, VALUE_INFINITE);
    }

    while (true) {
        Value score = negamax(depth, 0, alpha, beta, info);

        if (should_stop) {
            return score;
        }

        if (score <= alpha) {
            alpha = std::max(alpha - delta, -VALUE_INFINITE);
        } else if (score >= beta) {
            beta = std::min(beta + delta, VALUE_INFINITE);
        } else {
            return score;  // landed inside: the true score
        }

        delta *= 2;
    }
}

void SearchEngine::report_iteration(const SearchInfo& info, std::int32_t depth,
                                    Value score) {
    std::lock_guard<std::mutex> io_lock(io_mutex);

    std::cout << "info depth " << depth;

    if (std::abs(score) >= VALUE_MATE - MAX_PLY) {
        std::int32_t mate_in = (VALUE_MATE - std::abs(score) + 1) / 2;

        std::cout << " score mate " << (score > 0 ? mate_in : -mate_in);
    } else {
        std::cout << " score cp " << score;
    }

    std::cout << " nodes " << (info.nodes + info.q_nodes) << " time "
              << info.time.count() << " pv";

    for (const Move& m : info.pv) {
        std::cout << " " << m.uci_move();
    }

    std::cout << std::endl;
}

std::int32_t SearchEngine::calculate_extension_depth(
    bool in_check, std::int32_t num_extensions) const {
    std::int32_t extension = 0;

    if (num_extensions < MAX_EXTENSION) {
        if (in_check) {
            extension = 1;
        }
    }

    return extension;
}

}  // namespace KhaosChess
