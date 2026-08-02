#include <gtest/gtest.h>

#include <unistd.h>
#include <cstdio>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>

#include "move.h"
#include "nnue.h"
#include "position.h"
#include "test_common.h"
#include "thread.h"
#include "uci.h"

namespace KhaosChess {
namespace {

// Run `f`, capturing everything it writes to std::cout and returning it. The
// setoption handlers report through std::cout, so this is how the tests read
// their "info string ..." replies.
template <typename F>
std::string capture_cout(F&& f) {
    std::ostringstream buffer;
    std::streambuf* previous = std::cout.rdbuf(buffer.rdbuf());
    f();
    std::cout.rdbuf(previous);
    return buffer.str();
}

class UciTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        init_engine_once();
    }

    // Every test starts from a clean NNUE state so load/toggle checks are
    // independent of order.
    void SetUp() override {
        nnue::unload();
        nnue::use_nnue = true;
    }
};

// ------------------------- parse_move -------------------------

TEST_F(UciTest, ParseMoveRoundTripsBasic) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    EXPECT_EQ(parse_move("e2e4", pos).uci_move(), "e2e4");
    EXPECT_EQ(parse_move("g1f3", pos).uci_move(), "g1f3");
}

TEST_F(UciTest, ParseMoveRejectsIllegal) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    EXPECT_EQ(parse_move("e2e5", pos), Move::invalid_move());
    EXPECT_EQ(parse_move("a1a3", pos), Move::invalid_move());
}

TEST_F(UciTest, ParseMovePromotion) {
    MoveInfo mi{};
    Position pos;
    pos.set("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", &mi);
    Move m = parse_move("a7a8q", pos);
    ASSERT_NE(m, Move::invalid_move());
    EXPECT_EQ(m.move_type(), MT_PROMOTION);
    EXPECT_EQ(m.promoted(), QUEEN);
    EXPECT_EQ(m.uci_move(), "a7a8q");
}

TEST_F(UciTest, ParseMoveCastling) {
    MoveInfo mi{};
    Position pos;
    pos.set(kKiwipete, &mi);
    Move m = parse_move("e1g1", pos);  // UCI king->g file
    ASSERT_NE(m, Move::invalid_move());
    EXPECT_EQ(m.move_type(), MT_CASTLING);
}

// ------------------------- parse_position -------------------------

TEST_F(UciTest, ParsePositionStartpos) {
    Position pos;
    InfoListPtr infos(new std::deque<MoveInfo>(1));
    parse_position("position startpos", pos, infos);

    EXPECT_EQ(pos.side_to_move(), WHITE);
    EXPECT_NE(pos.get_fen().find(
                  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"),
              std::string::npos);
}

TEST_F(UciTest, ParsePositionStartposWithMoves) {
    Position pos;
    InfoListPtr infos(new std::deque<MoveInfo>(1));
    parse_position("position startpos moves e2e4 e7e5", pos, infos);

    // Two plies played, so it is White's turn again.
    EXPECT_EQ(pos.side_to_move(), WHITE);
    EXPECT_NE(pos.get_fen().find("4p3/4P3"), std::string::npos);
}

TEST_F(UciTest, ParsePositionFromFen) {
    Position pos;
    InfoListPtr infos(new std::deque<MoveInfo>(1));
    parse_position(("position fen " + kEnPassantPins).c_str(), pos, infos);

    EXPECT_EQ(pos.side_to_move(), WHITE);
    EXPECT_NE(pos.get_fen().find("8/2p5/3p4/KP5r/1R3p1k"), std::string::npos);
}

TEST_F(UciTest, ParsePositionFenWithMoves) {
    Position pos;
    InfoListPtr infos(new std::deque<MoveInfo>(1));
    parse_position("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
                   "w KQkq - 0 1 moves d2d4",
                   pos, infos);

    EXPECT_EQ(pos.side_to_move(), BLACK);
    EXPECT_NE(pos.get_fen().find("3P4"), std::string::npos);
}

// ------------------------- parse_go -------------------------

TEST_F(UciTest, ParseGoDepth) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    SearchLimits limits;
    EXPECT_TRUE(parse_go("go depth 12", pos, limits));
    EXPECT_EQ(limits.depth, 12);
}

TEST_F(UciTest, ParseGoNodes) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    SearchLimits limits;
    EXPECT_TRUE(parse_go("go nodes 5000", pos, limits));
    EXPECT_EQ(limits.node_limit, 5000u);
    EXPECT_EQ(limits.depth, 64);
}

TEST_F(UciTest, ParseGoMovetime) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    SearchLimits limits;
    EXPECT_TRUE(parse_go("go movetime 1000", pos, limits));
    EXPECT_EQ(limits.max_time.count(), 1000);
    EXPECT_EQ(limits.soft_time.count(), 0);  // explicit movetime => no soft cut
}

TEST_F(UciTest, ParseGoClockAllocatesSoftAndHard) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);  // White to move, game phase 24
    SearchLimits limits;
    ASSERT_TRUE(parse_go("go wtime 60000 btime 60000 winc 1000 binc 1000", pos,
                         limits));

    // available = 60000 - 30 overhead; moves_left = 20 + 24; optimum =
    // available/44 + 1000*0.75; hard = 4 * optimum. Pinned so a change to the
    // allocation formula is caught here rather than only in a match.
    EXPECT_EQ(limits.soft_time.count(), 2112);
    EXPECT_EQ(limits.max_time.count(), 8448);
    EXPECT_GT(limits.soft_time.count(), 0);
    EXPECT_GT(limits.max_time.count(), limits.soft_time.count());
}

TEST_F(UciTest, ParseGoBareFallsBackToDepthSix) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    SearchLimits limits;
    EXPECT_TRUE(parse_go("go", pos, limits));
    EXPECT_EQ(limits.depth, 6);
}

TEST_F(UciTest, ParseGoInfiniteHasNoFallbackOrSoftLimit) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    SearchLimits limits;
    EXPECT_TRUE(parse_go("go infinite", pos, limits));
    EXPECT_EQ(limits.depth, 64);
    EXPECT_EQ(limits.soft_time.count(), 0);
}

TEST_F(UciTest, ParseGoPonderSetsFlag) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    SearchLimits limits;
    EXPECT_TRUE(parse_go("go ponder wtime 1000 btime 1000", pos, limits));
    EXPECT_TRUE(limits.ponder);
    EXPECT_EQ(limits.depth, 64);  // pondering skips the bare-go fallback
}

TEST_F(UciTest, ParseGoPerftReturnsFalse) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    SearchLimits limits;
    bool launch = true;
    // perft prints per-move counts; swallow them.
    capture_cout([&] { launch = parse_go("go perft 2", pos, limits); });
    EXPECT_FALSE(launch);
}

// ------------------------- print_id_and_options -------------------------

TEST_F(UciTest, PrintOptionsListsEveryOption) {
    std::ostringstream os;
    print_id_and_options(os);
    const std::string out = os.str();

    EXPECT_NE(out.find("id name "), std::string::npos);
    EXPECT_NE(out.find("id author "), std::string::npos);
    EXPECT_NE(out.find("option name Threads type spin default 1 min 1 max 256"),
              std::string::npos);
    EXPECT_NE(out.find("option name Hash type spin default 64 min 1 max 4096"),
              std::string::npos);
    EXPECT_NE(out.find("option name Ponder type check default false"),
              std::string::npos);
    EXPECT_NE(out.find("option name EvalFile type string default <empty>"),
              std::string::npos);
    EXPECT_NE(out.find("option name UseNNUE type check default true"),
              std::string::npos);
    EXPECT_NE(out.find("uciok"), std::string::npos);
}

TEST_F(UciTest, PrintOptionsCheckTypeHasNoMinMax) {
    std::ostringstream os;
    print_id_and_options(os);
    const std::string out = os.str();

    // The Ponder line is a "check" and must end without a min/max clause.
    const std::string line = "option name Ponder type check default false";
    const std::size_t at = out.find(line);
    ASSERT_NE(at, std::string::npos);
    EXPECT_EQ(out[at + line.size()], '\n');
}

// ------------------------- handle_setoption -------------------------

TEST_F(UciTest, SetOptionUseNnueToggles) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);

    std::string out = capture_cout(
        [&] { handle_setoption("setoption name UseNNUE value false", pos); });
    EXPECT_FALSE(nnue::use_nnue);
    EXPECT_NE(out.find("UseNNUE false"), std::string::npos);

    out = capture_cout(
        [&] { handle_setoption("setoption name UseNNUE value true", pos); });
    EXPECT_TRUE(nnue::use_nnue);
    EXPECT_NE(out.find("UseNNUE true"), std::string::npos);
}

// The value token, not a substring of the whole line, selects the option. A
// path that merely contains "Hash" must be handled as EvalFile, not silently
// swallowed by the Hash branch (which was the old strstr bug).
TEST_F(UciTest, SetOptionEvalFilePathContainingHashGoesToEvalFile) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);

    std::string out = capture_cout([&] {
        handle_setoption(
            "setoption name EvalFile value /no/such/Hash/net.nnue", pos);
    });

    EXPECT_NE(out.find("EvalFile error"), std::string::npos);
    EXPECT_FALSE(nnue::is_loaded());
}

TEST_F(UciTest, SetOptionEvalFileLoadsAndClears) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);

    const std::string net_path =
        "/tmp/khaos_uci_test_" + std::to_string(::getpid()) + ".nnue";
    ASSERT_TRUE(nnue::write_random_net(net_path, 0x1234ULL));

    std::string out = capture_cout([&] {
        handle_setoption("setoption name EvalFile value " + net_path, pos);
    });
    EXPECT_NE(out.find("loaded net"), std::string::npos);
    EXPECT_TRUE(nnue::is_loaded());

    out = capture_cout([&] {
        handle_setoption("setoption name EvalFile value <empty>", pos);
    });
    EXPECT_NE(out.find("cleared"), std::string::npos);
    EXPECT_FALSE(nnue::is_loaded());

    std::remove(net_path.c_str());
}

TEST_F(UciTest, SetOptionWithoutValueIsNoOp) {
    MoveInfo mi{};
    Position pos;
    pos.set(kStartPos, &mi);
    std::string out = capture_cout(
        [&] { handle_setoption("setoption name Threads", pos); });
    EXPECT_TRUE(out.empty());
}

}  // namespace
}  // namespace KhaosChess
