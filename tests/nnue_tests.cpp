#include <gtest/gtest.h>

#include <unistd.h>
#include <cstdio>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include "movegen.h"
#include "nnue.h"
#include "position.h"
#include "score.h"
#include "test_common.h"

namespace KhaosChess {
namespace {

// A net has to be loaded for the accumulator to be maintained at all, so every
// test here runs against a deterministic pseudo-random net written to a temp
// file. That exercises the real loader as well as the update path.
class NnueTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        init_engine_once();

        net_path = "/tmp/khaos_nnue_test_" + std::to_string(::getpid()) + ".nnue";
        ASSERT_TRUE(nnue::write_random_net(net_path, 0xC0FFEEULL));

        std::string error;
        ASSERT_TRUE(nnue::load(net_path, error)) << error;
    }

    static void TearDownTestSuite() {
        nnue::unload();
        std::remove(net_path.c_str());
    }

    static std::string net_path;
};

std::string NnueTest::net_path;

// Compare the incrementally maintained accumulator against one rebuilt from
// scratch. They must agree bit-for-bit, not approximately: the update is exact
// integer arithmetic, so any difference is a bug in the move/accumulator
// pairing rather than rounding.
::testing::AssertionResult AccumulatorMatchesRefresh(Position& pos) {
    nnue::Accumulator incremental = pos.accumulator();
    pos.refresh_accumulator();
    const nnue::Accumulator& fresh = pos.accumulator();

    for (Color p : {WHITE, BLACK}) {
        for (std::int32_t i = 0; i < nnue::HIDDEN; ++i) {
            if (incremental.v[p][i] != fresh.v[p][i]) {
                return ::testing::AssertionFailure()
                       << "perspective " << (p == WHITE ? "WHITE" : "BLACK")
                       << " neuron " << i << ": incremental "
                       << incremental.v[p][i] << " != refreshed "
                       << fresh.v[p][i] << "\nFEN: " << pos.get_fen();
            }
        }
    }

    return ::testing::AssertionSuccess();
}

TEST_F(NnueTest, LoadedNetIsReported) {
    EXPECT_TRUE(nnue::is_loaded());
}

TEST_F(NnueTest, RejectsMissingFile) {
    std::string error;
    EXPECT_FALSE(nnue::load("/nonexistent/path/to.nnue", error));
    EXPECT_FALSE(error.empty());
    // A failed load must leave the working net in place.
    EXPECT_TRUE(nnue::is_loaded());
}

TEST_F(NnueTest, RejectsGarbageFile) {
    std::string path = "/tmp/khaos_nnue_bad_" + std::to_string(::getpid());
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::fputs("not a network at all", f);
        std::fclose(f);
    }

    std::string error;
    EXPECT_FALSE(nnue::load(path, error));
    EXPECT_TRUE(nnue::is_loaded());
    std::remove(path.c_str());
}

// set() must seed the accumulator with the feature bias before placing pieces;
// if it did not, a freshly parsed FEN would already disagree with a refresh.
TEST_F(NnueTest, FenSetupMatchesRefresh) {
    for (const std::string& fen :
         {kStartPos, kKiwipete, kEnPassantPins, kPromotions, kTalkchess}) {
        std::deque<MoveInfo> infos(1);
        Position pos;
        pos.set(fen, &infos.back());
        EXPECT_TRUE(AccumulatorMatchesRefresh(pos)) << "fen: " << fen;
    }
}

TEST_F(NnueTest, EvaluationIsSymmetricUnderColourFlip) {
    // A mirrored position must evaluate to the same score for the side to move,
    // which is what proves the perspective indexing (the s^56 flip and the
    // own/their halves) is wired up consistently.
    struct Pair {
        std::string white;
        std::string black;
    };

    const std::vector<Pair> mirrors = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"},
        {"4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1",
         "4k3/8/2n5/8/3P4/8/8/4K3 b - - 0 1"},
        {"r3k2r/pp3ppp/8/8/8/8/PP3PPP/R3K2R w KQkq - 0 1",
         "r3k2r/pp3ppp/8/8/8/8/PP3PPP/R3K2R b KQkq - 0 1"},
    };

    for (const Pair& m : mirrors) {
        std::deque<MoveInfo> wi(1);
        std::deque<MoveInfo> bi(1);
        Position wp;
        Position bp;
        wp.set(m.white, &wi.back());
        bp.set(m.black, &bi.back());

        EXPECT_EQ(nnue::forward(wp.accumulator(), wp.side_to_move()),
                  nnue::forward(bp.accumulator(), bp.side_to_move()))
            << m.white << "  vs  " << m.black;
    }
}

// The core regression test. Walk random games and assert the accumulator still
// matches a refresh after every do_move and after every undo_move, so captures,
// promotions, castling and en passant are all covered by construction.
TEST_F(NnueTest, IncrementalUpdatesSurviveRandomGames) {
    std::mt19937_64 rng(0x5EEDULL);

    for (const std::string& fen :
         {kStartPos, kKiwipete, kEnPassantPins, kPromotions, kTalkchess}) {
        for (std::int32_t game = 0; game < 20; ++game) {
            std::deque<MoveInfo> infos(1);
            Position pos;
            pos.set(fen, &infos.back());

            std::vector<Move> played;

            for (std::int32_t ply = 0; ply < 60; ++ply) {
                MoveList<GT_LEGAL> moves(pos);
                if (moves.size() == 0) {
                    break;
                }

                const std::size_t pick =
                    rng() % static_cast<std::size_t>(moves.size());
                Move move = *(moves.begin() + pick);

                infos.emplace_back();
                pos.do_move(move, infos.back());
                played.push_back(move);

                ASSERT_TRUE(AccumulatorMatchesRefresh(pos))
                    << "after " << move.uci_move() << " from " << fen;
            }

            // Unwind the whole game: undo_move replays the inverse piece
            // mutations, so the accumulator must land back on the start
            // position's value with no drift accumulated along the way.
            while (!played.empty()) {
                pos.undo_move(played.back());
                played.pop_back();
                infos.pop_back();

                ASSERT_TRUE(AccumulatorMatchesRefresh(pos))
                    << "after undo, from " << fen;
            }
        }
    }
}

// A null move moves no pieces, so the accumulator must come out bit-identical;
// only which half feeds the "own" output weights changes.
//
// Note what is deliberately NOT asserted here: that the score negates. This
// architecture is not antisymmetric by construction -- the output layer has
// independent weights for the own and their halves plus a bias, so
// forward(acc, WHITE) and forward(acc, BLACK) are unrelated values for an
// arbitrary net. A trained net learns approximate antisymmetry from data; it is
// never exact, and nothing in the search relies on it being so.
TEST_F(NnueTest, NullMoveLeavesAccumulatorUntouched) {
    std::deque<MoveInfo> infos(1);
    Position pos;
    pos.set(kKiwipete, &infos.back());

    nnue::Accumulator before = pos.accumulator();
    const Color stm_before = pos.side_to_move();

    infos.emplace_back();
    pos.do_null_move(infos.back());

    ASSERT_EQ(pos.side_to_move(), ~stm_before);
    for (Color p : {WHITE, BLACK}) {
        for (std::int32_t i = 0; i < nnue::HIDDEN; ++i) {
            ASSERT_EQ(before.v[p][i], pos.accumulator().v[p][i]);
        }
    }

    pos.undo_null_move();
    ASSERT_EQ(pos.side_to_move(), stm_before);
    EXPECT_TRUE(AccumulatorMatchesRefresh(pos));
}

}  // namespace
}  // namespace KhaosChess
