#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "consts.h"
#include "defs.h"
#include "move.h"
#include "movegen.h"
#include "nnue.h"

namespace KhaosChess {
// Empty fen string
extern const std::string EMPTY_FEN;

// Starting fen string
extern const std::string START_FEN;

// Some example fen string
extern const std::string TEST_FEN;
extern const std::string TEST_ATTACKS_FEN;

/*
        binary representation of castling rights

        bin		dec		description

        0001	1		white king can castle to the king side
        0010	2		white king can castle to the queen side
        0100	4		black king can castle to the king side
        1000	8		black king can castle to the queen side
*/

/*
        examples

        1111 both can castle both directions

        1001	white king => king side
                        black king => queen side
*/

extern char ascii_pieces[PIECE_NB];

// char unicode_pieces[12] = { '♙','♘','♗','♖','♕','♔','♟','♞','♝','♜','♛','♚'
// };

// Info structure stores information needed to restore a position
// to its previous state when a move is taken back
struct MoveInfo {
    // Copied when making a move
    CastlingRights castling_rights;
    Square en_passant;
    PLY_TYPE fifty_move;  // halfmove clock
    Piece captured_piece;
    BITBOARD key;  // Zobrist hash of the position after the move

    // Not copied
    MoveInfo* next;
    MoveInfo* prev;
};

// List to keep track of position states along the setup
// std::deque is used because pointers to elements are not invalidated when list
// resizing
using InfoListPtr = std::unique_ptr<std::deque<MoveInfo>>;

// This class stores information to the board representation as pieces,
// side to move, castling info, etc.
class Position {
   public:
    // static void init();

    Position() = default;
    // Delete the copy constructor
    Position(const Position&) = delete;
    // Delete the copy assignment operator
    Position& operator=(const Position&) = delete;

    // FEN i/o
    Position& set(const std::string& fen, MoveInfo* mi);
    std::string get_fen() const;

    // Squares
    template <PieceType pt>
    Square square(Color c) const {
        return get_ls1b(get_pieces_bb(pt, c));
    }

    BITBOARD key() const {
        return move_info->key;
    }

    Square ep_square() const {
        return move_info->en_passant;
    }

    Square castling_rook_square(CastlingRights cr) const;

    // Bitboards
    template <typename... PieceTypes>
    inline BITBOARD get_pieces_bb(PieceType pt, PieceTypes... pts) const;
    inline BITBOARD get_pieces_bb(PieceType pt) const;
    inline BITBOARD get_pieces_bb(PieceType pt, Color c) const {
        return (type[pt] & occupancies[c]);
    }
    inline BITBOARD get_pieces_bb(Color c) const {
        return occupancies[c];
    }

    inline BITBOARD get_our_pieces_bb() const {
        return occupancies[side];
    }
    inline BITBOARD get_opponent_pieces_bb() const {
        return occupancies[~side];
    }
    inline BITBOARD get_all_pieces_bb() const {
        return occupancies[BOTH];
    }
    inline BITBOARD get_all_empty_squares_bb() const {
        return ~occupancies[BOTH];
    }

    inline BITBOARD get_attackers_to(Square s) const;
    inline BITBOARD get_attackers_to(Square s, BITBOARD occ) const;
    template <PieceType pt>
    inline BITBOARD get_attacks_by(Color c) const;
    inline BITBOARD get_attacked_squares(Color side) const;
    inline BITBOARD get_checked_squares(PieceType pt) const;
    inline BITBOARD get_threats(PieceType pt) const {
        return threats[pt];
    }
    inline BITBOARD get_king_blockers(Color c) const {
        return blocking_pieces[c];
    }
    inline BITBOARD get_pinners(Color c) const {
        return pinning_pieces[c];
    }

    // Booleans
    bool is_empty(Square s) const {
        return get_piece_on(s) == NO_PIECE;
    }
    bool gives_check(Move m) const;
    bool can_castle(CastlingRights cr) const {
        return move_info->castling_rights & cr;
    }
    bool is_castling_interrupted(CastlingRights cr) const;
    bool is_legal(Move m) const;
    bool is_square_attacked(Square s) const;
    bool is_square_attacked(Square s, Color c) const;
    bool is_draw() const;
    bool see_ge(Move m, Value threshold = 0) const;

    // Pieces
    Piece get_piece_on(Square s) const;
    Piece moved_piece(Move m) const;
    Piece captured_piece() const;

    // PLY
    PLY_TYPE game_ply() const {
        return fullmove_number;
    };
    PLY_TYPE fifty_move_count() const {
        return move_info->fifty_move;
    };

    // Piece count
    template <PieceType pt>
    std::int32_t count(Color c) const;
    std::int32_t count(Color c, PieceType pt) const;
    template <typename... PieceTypes>
    std::int32_t count(Color c, PieceType pt, PieceTypes... pts) const;

    std::int32_t game_phase() const;

    // Piece Count Vector
    PCV get_pcv() const;

    // Doing and undoing moves
    void do_move(const Move& m, MoveInfo& new_info);
    void undo_move(const Move& m);
    void do_null_move(MoveInfo& new_info);
    void undo_null_move();

    void update_blocks_and_pins(Color c);
    void remove_piece(Square s);
    void place_piece(Piece p, Square s);

    // NNUE hidden state for this position. Maintained incrementally by
    // place_piece/remove_piece/move_piece, so do_move updates it as a side
    // effect and undo_move -- which replays the inverse mutations -- restores
    // it exactly. No accumulator stack and no dirty-piece list: the board
    // mutators are the single source of truth, so the accumulator cannot drift
    // out of step with the move logic.
    const nnue::Accumulator& accumulator() const {
        return acc;
    }

    // Rebuild the accumulator from the piece list. Needed after loading a net
    // mid-game (the incremental updates were skipped while none was loaded)
    // and used by the tests as the reference the incremental path must match.
    void refresh_accumulator();
    void calculate_threats();
    void print_attacked_squares(Color c) const;

    // Castling & side
    CastlingRights castling_rights(Color c) const {
        return c & CastlingRights(move_info->castling_rights);
    }

    Color side_to_move() const {
        return side;
    }

    // Overrides
    friend std::ostream& operator<<(std::ostream& os, const Position& position);

   private:
    template <bool Do>
    void do_castle(Color us, Square source, Square& target, Square& r_source,
                   Square& r_target);
    void set_castling_rights(Color c, Square r_source);
    void move_piece(Square source, Square target);

    BITBOARD get_least_valuable_piece(BITBOARD attacks, Color by_side,
                                      PieceType& pt) const;
    BITBOARD compute_key() const;

    // Data
    BITBOARD occupancies[BOTH + 1]{};             // All pieces of the side to move
    BITBOARD type[PIECE_TYPE_NB]{};               // All the piece types
    BITBOARD castling_path[CASTLING_RIGHT_NB]{};  // Castling path depending on
                                                  // the castling side
    BITBOARD threats[PIECE_TYPE_NB]{};            // Piece type threads
    BITBOARD pinning_pieces[BOTH]{};              // Pieces that are pinning
    BITBOARD blocking_pieces[BOTH]{};             // Pieces that are blocking

    Piece piece_board[SQUARE_TOTAL]{};  // Board of pieces

    Square rook_source_sq[CASTLING_RIGHT_NB]{};

    Color side{};

    PLY_TYPE fullmove_number{};

    std::int32_t piece_count[PIECE_NB]{};  // Piece count

    // State info
    MoveInfo* move_info{};

    nnue::Accumulator acc{};
};

// Calculate game phase (0-24, where 24 is opening, 0 is endgame)
inline std::int32_t Position::game_phase() const {
    return std::min(MAX_PHASE_SCORE,
                    count<KNIGHT>(WHITE) + count<KNIGHT>(BLACK) +
                        count<BISHOP>(WHITE) + count<BISHOP>(BLACK) +
                        2 * (count<ROOK>(WHITE) + count<ROOK>(BLACK)) +
                        4 * (count<QUEEN>(WHITE) + count<QUEEN>(BLACK)));
}

template <PieceType pt>
inline std::int32_t Position::count(Color c) const {
    return piece_count[get_piece(c, pt)];
}

inline std::int32_t Position::count(Color c, PieceType pt) const {
    return piece_count[get_piece(c, pt)];
}

template <typename... PieceTypes>
inline std::int32_t Position::count(Color c, PieceType pt,
                                    PieceTypes... pts) const {
    return count(c, pt) + count(c, pts...);
}

inline PCV Position::get_pcv() const {
    return encode_pcv(count<PAWN>(WHITE), count<KNIGHT>(WHITE),
                      count<BISHOP>(WHITE), count<ROOK>(WHITE),
                      count<QUEEN>(WHITE), count<PAWN>(BLACK),
                      count<KNIGHT>(BLACK), count<BISHOP>(BLACK),
                      count<ROOK>(BLACK), count<QUEEN>(BLACK));
}

// Is square attacked by OPPONENT
inline bool Position::is_square_attacked(Square s, Color us) const {
    BITBOARD all = get_all_pieces_bb();

    bool is_pawn_attack =
        us == WHITE ? pawn_attacks_bb(BLACK, s) & get_pieces_bb(PAWN, WHITE)
                    : pawn_attacks_bb(WHITE, s) & get_pieces_bb(PAWN, BLACK);

    bool is_knight_attack = attacks_bb_by<KNIGHT>(s) & get_pieces_bb(KNIGHT, us);
    bool is_bishop_attack =
        attacks_bb_by<BISHOP>(s, all) & get_pieces_bb(BISHOP, us);
    bool is_rook_attack = attacks_bb_by<ROOK>(s, all) & get_pieces_bb(ROOK, us);
    bool is_queen_attack =
        attacks_bb_by<QUEEN>(s, all) & get_pieces_bb(QUEEN, us);
    bool is_king_attack = attacks_bb_by<KING>(s) & get_pieces_bb(KING, us);

    return is_pawn_attack || is_knight_attack || is_bishop_attack ||
           is_rook_attack || is_queen_attack || is_king_attack;
}

// Is square attacked by US
inline bool Position::is_square_attacked(Square s) const {
    assert(is_square_ok(s));
    return is_square_attacked(s, side);
}

inline Piece Position::get_piece_on(Square s) const {
    assert(is_square_ok(s));
    return piece_board[s];
}

inline Piece Position::moved_piece(Move m) const {
    assert(m.is_move_ok());
    return get_piece_on(m.source_square());
}

inline BITBOARD Position::get_pieces_bb(PieceType pt) const {
    return type[pt];
}

template <typename... PieceTypes>
inline BITBOARD Position::get_pieces_bb(PieceType pt, PieceTypes... pts) const {
    return get_pieces_bb(pt) | get_pieces_bb(pts...);
}

inline BITBOARD Position::get_attackers_to(Square s, BITBOARD occ) const {
    BITBOARD w_pawn_att = pawn_attacks_bb(WHITE, s) & get_pieces_bb(PAWN, BLACK);
    BITBOARD b_pawn_att = pawn_attacks_bb(BLACK, s) & get_pieces_bb(PAWN, WHITE);
    BITBOARD knight_att = attacks_bb_by<KNIGHT>(s) & get_pieces_bb(KNIGHT);
    BITBOARD horizontal = attacks_bb_by<ROOK>(s, occ) &
                          (get_pieces_bb(ROOK) | get_pieces_bb(QUEEN));
    BITBOARD diagonal = attacks_bb_by<BISHOP>(s, occ) &
                        (get_pieces_bb(BISHOP) | get_pieces_bb(QUEEN));
    BITBOARD king_att = attacks_bb_by<KING>(s) & get_pieces_bb(KING);

    return w_pawn_att | b_pawn_att | knight_att | horizontal | diagonal |
           king_att;
}

inline BITBOARD Position::get_attackers_to(Square s) const {
    return get_attackers_to(s, get_all_pieces_bb());
}

inline Piece Position::captured_piece() const {
    return move_info->captured_piece;
}

inline void Position::place_piece(Piece p, Square s) {
    piece_board[s] = p;

    set_bit(type[type_of_piece(p)], s);
    set_bit(occupancies[get_piece_color(p)], s);

    occupancies[BOTH] |= occupancies[WHITE] | occupancies[BLACK];

    piece_count[p]++;
    piece_count[get_piece(get_piece_color(p), ALL_PIECES)]++;

    if (nnue::is_loaded()) {
        nnue::accumulator_add(acc, p, s);
    }
}

inline void Position::remove_piece(Square s) {
    Piece p = piece_board[s];

    if (nnue::is_loaded()) {
        nnue::accumulator_sub(acc, p, s);
    }

    // Update bitboards
    rm_bit(type[type_of_piece(p)], s);
    rm_bit(occupancies[get_piece_color(p)], s);
    rm_bit(occupancies[BOTH], s);

    // Clear piece
    piece_board[s] = NO_PIECE;

    // Update piece counts
    piece_count[p]--;
    piece_count[get_piece(get_piece_color(p), ALL_PIECES)]--;
}

inline void Position::move_piece(Square source, Square target) {
    Piece p = piece_board[source];

    if (nnue::is_loaded()) {
        nnue::accumulator_sub(acc, p, source);
        nnue::accumulator_add(acc, p, target);
    }

    // Update bitboards
    BITBOARD dest = square_to_BB(source) | square_to_BB(target);

    type[type_of_piece(p)] ^= dest;           // remove source and target
    occupancies[get_piece_color(p)] ^= dest;  // remove occupancies of this colour
    occupancies[BOTH] ^= dest;                // remove the source and add the target

    piece_board[source] = NO_PIECE;
    piece_board[target] = p;
}

inline bool Position::is_castling_interrupted(CastlingRights cr) const {
    assert(cr == WK || cr == WQ || cr == BK || cr == BQ);
    return get_all_pieces_bb() & castling_path[cr];
}

inline Square Position::castling_rook_square(CastlingRights cr) const {
    assert(cr == WK || cr == WQ || cr == BK || cr == BQ);
    return rook_source_sq[cr];
}

template <PieceType pt>
inline BITBOARD Position::get_attacks_by(Color c) const {
    if (pt == PAWN) {
        return c == WHITE ? pawn_attacks_bb<WHITE>(get_pieces_bb(PAWN, WHITE))
                          : pawn_attacks_bb<BLACK>(get_pieces_bb(PAWN, BLACK));
    }

    BITBOARD attacks = 0ULL;
    BITBOARD attackers = get_pieces_bb(pt, c);

    while (attackers) {
        attacks |= attacks_bb_by<pt>(pop_ls1b(attackers), get_all_pieces_bb());
    }

    return attacks;
}

// Computes the attacks and returns a bitboard of all the attacked squares
BITBOARD Position::get_attacked_squares(Color side) const {
    BITBOARD attacks = 0ULL;

    attacks |= get_attacks_by<PAWN>(~side);
    attacks |= get_attacks_by<KNIGHT>(~side);
    attacks |= get_attacks_by<BISHOP>(~side);
    attacks |= get_attacks_by<ROOK>(~side);
    attacks |= get_attacks_by<QUEEN>(~side);
    attacks |= get_attacks_by<KING>(~side);

    return attacks;
}
}  // namespace KhaosChess
