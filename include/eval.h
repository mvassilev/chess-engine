#pragma once

#include "defs.h"
#include "endgame.h"
#include "nnue.h"
#include "position.h"
#include "score.h"

namespace KhaosChess {

// The search's single static-evaluation entry point: the network when one is
// loaded and enabled, otherwise the hand-crafted tapered evaluation. Both are
// signed from the side to move's point of view.
//
// Specialized endgame knowledge (the KPK bitbase, known mates) is consulted
// first either way -- those are exact results rather than estimates, and small
// nets are notoriously bad at exactly the positions they cover. Scorer already
// probes Endgames itself, so only the NNUE branch needs the explicit check.
//
// This lives here rather than in score.h because score.h and endgame.h include
// each other; only a header downstream of both can name Scorer and Endgames.
inline Value evaluate(const Position& pos) {
    if (!nnue::active()) {
        return Scorer<SC_ALL>().get_score(pos);
    }

    Value endgame = Endgames::score(pos);
    if (endgame != VALUE_NONE) {
        return endgame;
    }

    return nnue::forward(pos.accumulator(), pos.side_to_move());
}

}  // namespace KhaosChess
