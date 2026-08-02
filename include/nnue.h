#pragma once

#include <cstdint>
#include <string>

#include "defs.h"

namespace KhaosChess {
namespace nnue {

// ---------------------------------------------------------------------------
// Topology: (768 -> HIDDEN) x2 -> L2 -> 1, clipped ReLU. Only the feature
// transformer (the accumulator) is incremental; the rest is a dense matmul.
//
// The input is the plain "piece on square, from a perspective" feature set:
//
//     index(p, pc, pt, s) = (pc != p) * 384 + (pt - PAWN) * 64 + s^(p * 56)
//
// i.e. 2 (own / their piece) x 6 (piece type) x 64 (square, rank-flipped for
// the black perspective) = 768 inputs. Deliberately NOT HalfKP/HalfKA: without
// a king bucket in the index, a king move is an ordinary two-feature update,
// so the accumulator never needs a refresh mid-search. That is what makes the
// self-inverting update in Position (see accumulator_add/sub) sound.
//
// King buckets are the obvious generation-2 upgrade; they would require an
// accumulator stack and refresh-on-king-move, so they are out of scope here.
// ---------------------------------------------------------------------------

constexpr std::int32_t INPUTS = 768;
constexpr std::int32_t HIDDEN = 256;   // feature-transformer / accumulator width
constexpr std::int32_t L2 = 32;        // post-accumulator hidden layer width

// Quantization. A clipped activation in [0, QA] is a float in [0, 1]. Every
// post-accumulator layer stores weights as round(w * QB) and bias as
// round(b * QA * QB); dividing its int32 sum by QB returns to the QA scale.
constexpr std::int32_t QA = 255;
constexpr std::int32_t QB = 64;

// Engine eval units per 1.0 of raw network output. The net is trained so that
// sigmoid(raw) predicts the win probability, matching a training target of
// sigmoid(engine_score / EVAL_SCALE); this converts back. 1640 is four tuned
// pawns (Material::piece_value[PAWN] ~= 410), the usual "4 pawns" sigmoid
// scale expressed in this engine's units rather than centipawns.
constexpr std::int32_t EVAL_SCALE = 1640;

// Per-position hidden state: one accumulator per perspective. Maintained
// incrementally by Position; index it with the side whose turn it is to get
// the "own pieces" half.
struct Accumulator {
    alignas(64) std::int16_t v[BOTH][HIDDEN];
};

// The loaded network. Weights are public so the accumulator update in
// position.h can reach them without a call through a translation unit
// boundary; treat them as read-only after load.
struct Network {
    alignas(64) std::int16_t feature_weights[INPUTS][HIDDEN];  // feature-major
    alignas(64) std::int16_t feature_bias[HIDDEN];
    alignas(64) std::int16_t l2_weights[L2][2 * HIDDEN];  // output-major, own|their
    std::int32_t l2_bias[L2];
    alignas(64) std::int16_t output_weights[L2];
    std::int32_t output_bias;
};

// Non-null only while a net is loaded. Checked on every accumulator update, so
// keep it a plain pointer rather than anything with acquire semantics: nets are
// loaded from the UCI thread between searches, never concurrently with one.
extern const Network* net;

inline bool is_loaded() {
    return net != nullptr;
}

// True when the search should call nnue::evaluate() instead of the hand-crafted
// evaluation: a net is loaded and the UseNNUE toggle is on.
extern bool use_nnue;

inline bool active() {
    return use_nnue && (net != nullptr);
}

// Feature index of `pc`-coloured `pt` on square `s`, from `p`'s perspective.
inline std::int32_t feature_index(Color p, Color pc, PieceType pt, Square s) {
    return ((pc != p) ? 384 : 0) + (static_cast<std::int32_t>(pt) - PAWN) * 64 +
           static_cast<std::int32_t>(sq_relative_to_side(s, p));
}

// Add / subtract one piece's contribution to both perspectives. Callers must
// have checked is_loaded(). These are exact integer adds, so a sub() undoes the
// matching add() bit-for-bit no matter what order the updates happen in --
// which is why undo_move needs no saved accumulator: it simply replays the
// inverse piece mutations.
inline void accumulator_add(Accumulator& acc, Piece piece, Square s) {
    const Color pc = get_piece_color(piece);
    const PieceType pt = type_of_piece(piece);

    for (Color p : {WHITE, BLACK}) {
        const std::int16_t* row = net->feature_weights[feature_index(p, pc, pt, s)];
        for (std::int32_t i = 0; i < HIDDEN; ++i) {
            acc.v[p][i] += row[i];
        }
    }
}

inline void accumulator_sub(Accumulator& acc, Piece piece, Square s) {
    const Color pc = get_piece_color(piece);
    const PieceType pt = type_of_piece(piece);

    for (Color p : {WHITE, BLACK}) {
        const std::int16_t* row = net->feature_weights[feature_index(p, pc, pt, s)];
        for (std::int32_t i = 0; i < HIDDEN; ++i) {
            acc.v[p][i] -= row[i];
        }
    }
}

// Reset both perspectives to the feature bias (an empty board).
inline void accumulator_reset(Accumulator& acc) {
    if (net == nullptr) {
        return;
    }

    for (Color p : {WHITE, BLACK}) {
        for (std::int32_t i = 0; i < HIDDEN; ++i) {
            acc.v[p][i] = net->feature_bias[i];
        }
    }
}

// Run the output layer over an accumulator. `stm` selects which perspective
// supplies the "own" half, so the result is from the side to move's point of
// view -- the same convention as the hand-crafted evaluation.
Value forward(const Accumulator& acc, Color stm);

// Load a .nnue file. Returns false and leaves any previously loaded net in
// place on any failure (missing file, bad magic, topology mismatch, short
// read); `error` receives a human-readable reason.
bool load(const std::string& path, std::string& error);

// Drop the loaded net and fall back to the hand-crafted evaluation.
void unload();

// Serialize a deterministic pseudo-random net to `path`. Test-only helper: it
// lets the accumulator and loader be exercised without a trained file.
bool write_random_net(const std::string& path, std::uint64_t seed);

// Header layout, shared with the Python exporter (tools/export.py in the
// trainer repo). All little-endian.
//
//   char     magic[8]   "KHAOSNN1"
//   uint32   version    2
//   uint32   inputs     768
//   uint32   hidden     256
//   uint32   l2         32
//   int32    qa         255
//   int32    qb         64
//   int32    eval_scale 1640
//   int32    reserved[3]
//   int16    feature_weights[inputs * hidden]     (feature-major)
//   int16    feature_bias[hidden]
//   int16    l2_weights[l2 * (2 * hidden)]         (output-major)
//   int32    l2_bias[l2]
//   int16    output_weights[l2]
//   int32    output_bias
constexpr char MAGIC[8] = {'K', 'H', 'A', 'O', 'S', 'N', 'N', '1'};
constexpr std::uint32_t FORMAT_VERSION = 2;

}  // namespace nnue
}  // namespace KhaosChess
