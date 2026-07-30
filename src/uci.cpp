#include "uci.h"

#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include "datagen.h"
#include "eval.h"
#include "move.h"
#include "movegen.h"
#include "nnue.h"
#include "perft.h"
#include "position.h"
#include "thread.h"
#include "tt.h"

namespace KhaosChess {
constexpr std::int64_t MOVE_OVERHEAD_MS = 30;
constexpr std::int64_t ENDGAME_MOVE_HORIZON = 20;
constexpr std::int64_t HARD_LIMIT_FACTOR = 4;
constexpr double INCREMENT_SPEND_FACTOR = 0.75;

Move parse_move(std::string move_string, const Position& pos) {
    Square source =
        Square((move_string[0] - 'a') + (8 - (move_string[1] - '0')) * 8);
    Square target =
        Square((move_string[2] - 'a') + (8 - (move_string[3] - '0')) * 8);

    for (auto& m : MoveList<GT_LEGAL>(pos)) {
        Square m_target = m.target_square();

        // Castling is encoded king->rook square; UCI speaks king->g/c file
        if (m.move_type() == MT_CASTLING) {
            m_target =
                make_square(m_target > m.source_square() ? FILE_G : FILE_C,
                            rank_of(m.source_square()));
        }

        if (m.source_square() == source && m_target == target) {
            if (m.move_type() == MT_PROMOTION) {
                // UCI promotion letters are always lowercase (e7e8q); BLACK
                // indexes the lowercase half of ascii_pieces
                if (move_string[4] ==
                    ascii_pieces[get_piece(BLACK, m.promoted())]) {
                    return m;
                }

                continue;
            }

            return m;
        }
    }

    return Move::invalid_move();
}

void parse_position(const char* cmd, Position& pos, InfoListPtr& infos) {
    // every position command rebuilds the game from scratch.
    // start a fresh history chain, because old one is dead
    infos->clear();
    infos->emplace_back();

    // shift to next token, because "position" is 8 characters and " " is 1,
    // hence shift 9
    cmd += 9;
    const char* current = cmd;

    if (strncmp(current, "startpos", 8) == 0) {
        pos.set(START_FEN, &infos->back());
    } else if (strncmp(current, "testpos", 7) == 0) {
        pos.set(TEST_FEN, &infos->back());
    } else  // UCI "fen" command
    {
        // fen command is available
        current = strstr(cmd, "fen");

        // fen command is not available
        if (current == NULL) {
            pos.set(START_FEN, &infos->back());
        } else  // found FEN
        {
            current += 4;

            // init board position from FEN command
            pos.set(current, &infos->back());
        }
    }

    // parse moves command
    current = strstr(cmd, "moves");

    // moves available
    if (current != NULL) {
        // shift to the moves
        current += 6;

        while (*current) {
            // parse next move
            Move move = parse_move(current, pos);

            // no move
            if (move == Move::invalid_move()) {
                break;
            }

            // make the move
            infos->emplace_back();
            pos.do_move(move, infos->back());

            while (*current && *current != ' ') {
                current++;
            }

            current++;
        }
    }
}

bool parse_go(const char* cmd, Position& pos, SearchLimits& limits) {
    const char* current;

    if ((current = strstr(cmd, "perft"))) {
        perft_debug(pos, atoi(current + 6));
        return false;
    }

    std::int32_t depth = 0;
    std::int64_t movetime = 0;   // fixed per-move time, in milliseconds
    std::int64_t soft_time = 0;  // optimum budget: don't start a new iteration
    std::int64_t hard_time = 0;  // ceiling: abort an iteration in flight

    // fixed depth search
    if ((current = strstr(cmd, "depth"))) {
        depth = atoi(current + 6);
    }

    // fixed nodes per move
    std::int64_t max_nodes = 0;
    if ((current = strstr(cmd, "nodes"))) {
        max_nodes = atoll(current + 6);
    }

    if ((current = strstr(cmd, "movetime"))) {
        // Explicit per-move time: spend it exactly, no soft cutoff.
        movetime = atoll(current + 9);
        hard_time = movetime;
    } else {
        // Allocate a slice of the remaining clock time.
        const char* time_token = pos.side_to_move() == WHITE ? "wtime" : "btime";
        const char* inc_token = pos.side_to_move() == WHITE ? "winc" : "binc";

        std::int64_t time_left = 0;
        std::int64_t increment = 0;
        std::int64_t moves_to_go = 0;

        if ((current = strstr(cmd, time_token))) {
            time_left = atoll(current + 6);
        }
        if ((current = strstr(cmd, inc_token))) {
            increment = atoll(current + 5);
        }
        if ((current = strstr(cmd, "movestogo"))) {
            moves_to_go = atoll(current + 10);
        }

        if (time_left > 0) {
            std::int64_t available =
                std::max<std::int64_t>(time_left - MOVE_OVERHEAD_MS, 1);

            std::int64_t moves_left =
                moves_to_go > 0 ? moves_to_go
                                : ENDGAME_MOVE_HORIZON + pos.game_phase();

            std::int64_t optimum =
                available / moves_left + static_cast<std::int64_t>(increment * INCREMENT_SPEND_FACTOR);

            soft_time = std::min(optimum, available);
            hard_time = std::min(optimum * HARD_LIMIT_FACTOR, available);
        }
    }

    // "go ponder ...": search the predicted position on the opponent's clock.
    // The clock/movetime is still parsed above and becomes the budget once
    // ponderhit arrives; until then the search ignores it.
    bool ponder = strstr(cmd, "ponder") != nullptr;
    bool infinite = strstr(cmd, "infinite") != nullptr;

    // bare "go": no explicit limit, fall back to a fixed depth for quick
    // interactive testing. "go infinite" and pondering instead deepen until
    // "stop" (bounded only by the depth-64 / one-day ceilings), so they skip
    // the fallback.
    if (!ponder && !infinite && !depth && !movetime && !soft_time &&
        !max_nodes) {
        depth = 6;
    }

    limits = SearchLimits{};
    constexpr std::int64_t ONE_DAY_MS = 24LL * 60 * 60 * 1000;
    limits.max_time =
        std::chrono::milliseconds(hard_time ? hard_time : ONE_DAY_MS);
    limits.soft_time = std::chrono::milliseconds(soft_time);  // 0 => no cutoff
    limits.node_limit = static_cast<std::uint64_t>(max_nodes);
    limits.depth = depth ? depth : 64;
    limits.ponder = ponder;

    return true;
}

// Runs a search to completion and reports the chosen move. Meant to run on a
// background thread so the UCI loop stays responsive to "stop"; it returns
// once the search hits its own limit or "stop" raises the abort flag.
void run_search(Position& pos, SearchLimits limits) {
    SearchInfo info = Threads.run(pos, limits);

    Move best = info.pv.empty() ? Move::invalid_move() : info.pv[0];

    // The second PV move is what we expect the opponent to reply; hand it to
    // the GUI as the ponder move so it can have us think on their clock.
    Move ponder = info.pv.size() >= 2 ? info.pv[1] : Move::invalid_move();

    // search was stopped before depth 1 completed; play any legal move
    if (best == Move::invalid_move()) {
        MoveList<GT_LEGAL> moves(pos);
        if (moves.size()) {
            best = *moves.begin();
        }
        ponder = Move::invalid_move();
    }

    std::lock_guard<std::mutex> io_lock(io_mutex);
    if (best == Move::invalid_move()) {
        std::cout << "bestmove (none)\n"
                  << std::flush;
    } else if (ponder == Move::invalid_move()) {
        std::cout << "bestmove " << best.uci_move() << "\n"
                  << std::flush;
    } else {
        std::cout << "bestmove " << best.uci_move() << " ponder "
                  << ponder.uci_move() << "\n"
                  << std::flush;
    }
}

// The identify-and-options block, emitted on startup and on every "uci".
void print_id_and_options() {
    std::cout << "id name " << NAME << " " << VERSION << "\n";
    std::cout << "id author " << AUTHOR << "\n";
    std::cout << "option name Threads type spin default 1 min 1 max 256\n";
    std::cout << "option name Hash type spin default 64 min 1 max 4096\n";
    std::cout << "option name Ponder type check default false\n";
    std::cout << "option name EvalFile type string default <empty>\n";
    std::cout << "option name UseNNUE type check default true\n";
    std::cout << "uciok\n";
}

/*
        GUI -> isready
        Engine -> readyok
        GUI -> ucinewgame
*/
void uci_loop() {
    constexpr auto INPUT_BUFFER = 10000;

    // rest stdin & stdout buffers
    std::setvbuf(stdin, NULL, _IONBF, 0);
    std::setvbuf(stdout, NULL, _IONBF, 0);

    // def user/GUI inout buffer
    char input_buffer[INPUT_BUFFER];

    print_id_and_options();

    InfoListPtr infos(new std::deque<MoveInfo>(1));
    Position pos;

    // Boot into a valid (empty) position so commands like "d" are safe
    // even before the GUI sends "position"
    pos.set(EMPTY_FEN, &infos->back());

    // The active search runs on its own thread so this loop can keep reading
    // "stop", "isready", "ponderhit" and "quit" while the engine is thinking.
    std::thread search_thread;
    SearchLimits current_limits;  // limits of the running search (for ponderhit)
    auto stop_and_join = [&search_thread]() {
        if (search_thread.joinable()) {
            SearchEngine::stop();  // raise the abort flag every worker polls
            search_thread.join();  // wait for run_search to print bestmove
        }
    };

    // main loop
    while (true) {
        // rest user/GUI input
        memset(input_buffer, 0, sizeof(input_buffer));

        // making sure output reacehes GUI
        fflush(stdout);

        // get user/GUI input
        if (!fgets(input_buffer, INPUT_BUFFER, stdin)) {
            break;
        }

        // available input
        if (input_buffer[0] == '\n') {
            continue;
        }

        // parse UCI "isready" command
        if (strncmp(input_buffer, "isready", 7) == 0) {
            {
                std::lock_guard<std::mutex> io_lock(io_mutex);
                std::cout << "readyok\n";
            }
            continue;
        }
        // parse UCI "setoption" command
        else if (strncmp(input_buffer, "setoption", 9) == 0) {
            const char* val = strstr(input_buffer, "value");
            if (val && strstr(input_buffer, "Threads")) {
                Threads.set_count(atoi(val + 6));
            } else if (val && strstr(input_buffer, "Hash")) {
                // Resize the transposition table to the requested MB. Wipes
                // its contents, so this is a between-games operation.
                tt::TT.resize(static_cast<std::size_t>(atoi(val + 6)));
            } else if (val && strstr(input_buffer, "EvalFile")) {
                // Trim the value: GUIs pad with spaces and a trailing newline,
                // and a path is taken verbatim otherwise.
                std::string path(val + 6);
                while (!path.empty() && (path.back() == '\n' ||
                                         path.back() == '\r' ||
                                         path.back() == ' ')) {
                    path.pop_back();
                }
                std::size_t start = path.find_first_not_of(' ');
                path = (start == std::string::npos) ? "" : path.substr(start);

                std::lock_guard<std::mutex> io_lock(io_mutex);
                if (path.empty() || path == "<empty>") {
                    nnue::unload();
                    std::cout << "info string EvalFile cleared, using the "
                                 "hand-crafted evaluation\n";
                } else {
                    std::string error;
                    if (nnue::load(path, error)) {
                        // The incremental updates were skipped while no net was
                        // loaded, so this position's accumulator is unbuilt.
                        // Worker positions are re-set from FEN each search and
                        // rebuild theirs automatically.
                        pos.refresh_accumulator();
                        std::cout << "info string loaded net " << path << "\n";
                    } else {
                        std::cout << "info string EvalFile error: " << error
                                  << "\n";
                    }
                }
            } else if (val && strstr(input_buffer, "UseNNUE")) {
                nnue::use_nnue = strstr(val, "true") != nullptr;
                std::lock_guard<std::mutex> io_lock(io_mutex);
                std::cout << "info string UseNNUE "
                          << (nnue::use_nnue ? "true" : "false") << "\n";
            }
        }
        // parse UCI "position" command
        else if (strncmp(input_buffer, "position", 8) == 0) {
            parse_position(input_buffer, pos, infos);
        }

        // NNUE training data generation. Blocks until finished -- this is a
        // batch job, not something a GUI drives, so it deliberately owns the
        // process for its duration. Parallelism is one process per core.
        else if (strncmp(input_buffer, "datagen", 7) == 0) {
            stop_and_join();
            datagen::run(datagen::parse_options(input_buffer));
        }

        // parse UCI "ucinewgame" command
        else if (strncmp(input_buffer, "ucinewgame", 10) == 0) {
            tt::TT.clear();
            Threads.clear_history();
            parse_position("position startpos", pos, infos);
        }

        // parse UCI "go" command: launch the search on a background thread
        else if (strncmp(input_buffer, "go", 2) == 0) {
            stop_and_join();  // finish any search already in progress
            if (parse_go(input_buffer, pos, current_limits)) {
                // Lower the abort flag here, on the UCI thread, before the
                // search thread exists. Clearing it inside the search thread
                // would race with a "stop" arriving on this thread.
                SearchEngine::clear_stop();
                search_thread =
                    std::thread(run_search, std::ref(pos), current_limits);
            }
        }

        // parse UCI "ponderhit": the opponent played our predicted move, so the
        // running ponder search becomes a normal timed move. Arm the deadline
        // with the soft budget if we have one, otherwise the hard limit.
        else if (strncmp(input_buffer, "ponderhit", 9) == 0) {
            SearchEngine::ponderhit(current_limits.soft_time.count() > 0
                                        ? current_limits.soft_time
                                        : current_limits.max_time);
        }

        // parse UCI "stop" command: halt the current search. run_search then
        // reports bestmove and the join reclaims the thread.
        else if (strncmp(input_buffer, "stop", 4) == 0) {
            stop_and_join();
        }

        // parse UCI "quit" command
        else if (strncmp(input_buffer, "quit", 4) == 0) {
            stop_and_join();  // never leave a joinable thread to destruct
            break;            // exit loop
        }

        // parse UCI "uci" command
        else if (strncmp(input_buffer, "uci", 3) == 0) {
            std::cout << "\n";
            print_id_and_options();
        }

        // parse debug "eval" command - static evaluation breakdown,
        // optionally for a single component
        else if (strncmp(input_buffer, "eval", 4) == 0) {
            const char* arg = input_buffer + 4;

            // The number the search actually uses, and which path produced it.
            // One parseable line, so the trainer's cross-check script can
            // compare it against its own reference implementation -- and skip
            // the positions where exact endgame knowledge pre-empted the net
            // rather than reporting them as a mismatch.
            {
                Value endgame = Endgames::score(pos);
                const char* source = (endgame != VALUE_NONE) ? "endgame"
                                     : nnue::active()        ? "nnue"
                                                             : "hce";

                std::lock_guard<std::mutex> io_lock(io_mutex);
                std::cout << "static eval: " << evaluate(pos) << " (" << source
                          << ")\n";
            }

            if (strstr(arg, "material")) {
                Scorer<SC_MATERIAL>().print_stats(pos);
            } else if (strstr(arg, "mobility")) {
                Scorer<SC_MOBILITY>().print_stats(pos);
            } else if (strstr(arg, "king")) {
                Scorer<SC_KING_SAFETY>().print_stats(pos);
            } else if (strstr(arg, "pawn")) {
                Scorer<SC_PAWN_STRUCTURE>().print_stats(pos);
            } else if (strstr(arg, "coord")) {
                Scorer<SC_PIECE_COORDINATION>().print_stats(pos);
            } else {
                Scorer<SC_ALL>().print_stats(pos);
            }
        }

        else if (!strncmp(input_buffer, "d", 1)) {
            std::cout << pos << std::endl;
        }
    }

    // Reached on "quit" or EOF. "quit" already stopped and joined the search
    // above. On EOF (stdin closed without a "quit") let an in-flight search
    // finish naturally and print its bestmove instead of aborting it, so a
    // piped "position ... / go depth N" still shows the full search.
    if (search_thread.joinable()) {
        search_thread.join();
    }
}
}  // namespace KhaosChess
