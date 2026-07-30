#include "position.h"

#include <stddef.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

#include "bitboard.h"
#include "defs.h"
#include "zobrist.h"

namespace KhaosChess {
// Empty fen string
const std::string EMPTY_FEN = "8/8/8/8/8/8/8/8 b - - ";

// Starting fen string
const std::string START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Some example fen string
const std::string TEST_FEN =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 ";
const std::string TEST_ATTACKS_FEN = "8/8/8/3PN3/8/8/3p4/8 w - - ";

char ascii_pieces[PIECE_NB] = " PNBRQKpnbrqk";

// constexpr Piece Pieces[]{NO_PIECE,    WHITE_PAWN,   WHITE_KNIGHT,
// WHITE_BISHOP,
//                          WHITE_ROOK,  WHITE_QUEEN,  WHITE_KING,   NO_PIECE,
//                          BLACK_PAWN,  BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK,
//                          BLACK_QUEEN, BLACK_KING};

std::ostream& operator<<(std::ostream& os, const Position& position) {
    os << "\n";

    // loop on ranks
    for (Rank rank = RANK_8; rank <= RANK_1; ++rank) {
        // loop on files
        for (File file = FILE_A; file <= FILE_H; ++file) {
            Piece p = position.get_piece_on(make_square(file, rank));

            if (p != NO_PIECE) {
                os << " " << ascii_pieces[p];
            } else {
                os << " .";
            }
        }

        os << "  " << (8 - rank) << " \n";
    }

    // print files
    os << "\n a b c d e f g h \n\n";

    Color side = position.side_to_move();

    os << "\nSide:		" << ((side == WHITE) ? "white" : "black");

    Square en_passant = position.ep_square();
    os << "\nEn passant:	"
       << ((en_passant != NONE) ? squareToCoordinates[en_passant] : "no") << "\n";

    CastlingRights castle = position.move_info->castling_rights;

    std::string castling;
    (castle & WK) ? castling += 'K' : castling += '-';
    (castle & WQ) ? castling += 'Q' : castling += '-';
    (castle & BK) ? castling += 'k' : castling += '-';
    (castle & BQ) ? castling += 'q' : castling += '-';

    os << "Castling:	" << castling << "\n";

    os << "\nFEN: " << position.get_fen() << "\n";

    return os;
}

Position& Position::set(const std::string& fen, MoveInfo* mi) {
    /*
    FEN describes a Chess position, It is one line ASCII string.

    A FEN contains 6 fields separated by space

    1. Piece placement (from white's perspective).
    Each rank is separated by every / (slash). Each rank has letters and/or
    numbers. Letters  or  represent the pieces, small caps are the Black's
    pieces (pnbrkq) and upper case letters (PNBRKQ) are the White's pieces. The
    numbers (from 1 to 8) show how many empty spaces need to be empty

    2. Active colour or side to move. Represented as 'w' or 'b'

    3. Castling ability. If neither side can castle it is '-'.
    Otherwise FEN string indicates the castling rights using the letter 'KQ' and
    their representative lowercase kq. Uppercase letters mean castling rights for
    White and Lowercase letters mean castling rights for Black. 'K' stands for
    kingside and 'Q' stands for queen side.


    4. En passant target square. If there's no en passant target square then the
    notation for this is simply '-'. Otherwise the notation is the name of the
    square. If a pawn just made a 2 square move, this is the position "behind"
    the pawn.

    5. Halfmove clock. This is the number of halfmoves since the last pawn
    advance or capture. This is used to determine if a draw can be claimed under
    the 50-move rule.

    6. Fullmove number. The number of the full move.
    It starts at 1 and is incremented after Black's move.
    */

    Rank rank = RANK_8;
    File file = FILE_A;

    // reset boards and state variables
    memset(this, 0, sizeof(Position));
    memset(mi, 0, sizeof(MoveInfo));
    move_info = mi;

    // Seed the accumulator with the feature bias before any place_piece call
    // below adds to it -- the memset above zeroed it, and an all-zero
    // accumulator would silently drop the bias term.
    nnue::accumulator_reset(acc);

    side = WHITE;
    size_t idx = 0;

    // 1. Piece placement
    // loop over char elements
    while (idx < fen.length() && fen[idx] != ' ') {
        char c = fen[idx++];

        if (c == '/')  // new line
        {
            file = FILE_A;
            ++rank;
        } else if (isdigit(c)) {
            // leave this many empty spaces
            file += c - '0';
        } else  // it's a piece
        {
            Square sq = make_square(file, rank);
            size_t piece_idx = std::string(ascii_pieces).find(c);

            place_piece(Piece(piece_idx), sq);
            ++file;
        }
    }

    idx++;

    // 2. Side to move
    // inc pointer to castling rights and check the side to move
    side = ((idx < fen.length()) && (fen[idx] == 'w')) ? WHITE : BLACK;

    // go to castling rights
    idx += 2;

    // 3. Castling rights
    while ((idx < fen.length()) && (fen[idx] != ' ')) {
        Square r_sq = NONE;
        Color c = islower(fen[idx]) ? BLACK : WHITE;
        Piece rook = get_piece(c, ROOK);

        std::uint8_t token = toupper(fen[idx++]);

        if (token == 'K') {
            for (r_sq = sq_relative_to_side(H1, c); get_piece_on(r_sq) != rook;
                 --r_sq) {
                ;
            }
        } else if (token == 'Q') {
            for (r_sq = sq_relative_to_side(A1, c); get_piece_on(r_sq) != rook;
                 ++r_sq) {
                ;
            }
        } else if (token >= 'A' && token <= 'H') {
            r_sq = make_square(File(token - 'A'), rank_relative_to_side(c, RANK_1));
        } else {
            continue;  // invalid token
        }

        set_castling_rights(c, r_sq);
    }

    idx++;

    // 3. Enpassant square
    if (idx < fen.length() && fen[idx] != '-') {
        // init en_passant suqare
        move_info->en_passant =
            make_square(File(fen[idx] - 'a'), Rank(8 - (fen[idx] - '0')));

        idx += 2;  // skip the en passant square and space
    } else {
        move_info->en_passant = NONE, idx++;
    }

    // Skip to halfmove and fullmove
    idx++;

    // 5 & 6. Halfmove clock and fullmove number
    // Get the rest of the string for parsing numbers
    std::string remainder = fen.substr(idx);
    std::istringstream ss(remainder);

    // Read halfmove and fullmove
    ss >> move_info->fifty_move >> fullmove_number;

    // Convert from fullmove starting from one to gamePly starting from 0
    // handles also incorrect FEN's with fullmove = 0
    fullmove_number = std::max(2 * (fullmove_number - 1), 0) + (side == BLACK);

    // Calculate threats
    calculate_threats();
    move_info->key = compute_key();

    return *this;
}

// Get FEN function. Exists for debugging purposes only
std::string Position::get_fen() const {
    std::int32_t empty_sq_count;
    std::ostringstream ss;

    // 1. Piece placement section
    for (Rank rank = RANK_8; rank <= RANK_1; ++rank) {
        for (File file = FILE_A; file <= FILE_H; ++file) {
            for (empty_sq_count = 0;
                 file <= FILE_H && is_empty(make_square(file, rank)); ++file) {
                ++empty_sq_count;
            }

            // means print empty square count
            if (empty_sq_count) {
                ss << empty_sq_count;
            }

            // print piece on the square
            if (file <= FILE_H) {
                ss << ascii_pieces[get_piece_on(make_square(file, rank))];
            }
        }

        if (rank < RANK_1) {
            ss << '/';
        }
    }

    // Side to move
    ss << (side == WHITE ? " w " : " b ");

    // Castling rights
    if (can_castle(WK)) {
        ss << 'K';
    }
    if (can_castle(WQ)) {
        ss << 'Q';
    }
    if (can_castle(BK)) {
        ss << 'k';
    }
    if (can_castle(BQ)) {
        ss << 'q';
    }

    if (!can_castle(ANY)) {
        ss << '-';
    }

    ss << " "
       << (ep_square() == NONE  // En passant square
               ? "- "
               : squareToCoordinates[ep_square()])
       << " " << move_info->fifty_move << " "           // Rule 50
       << 1 + (fullmove_number - (side == BLACK)) / 2;  // Current move count

    return ss.str();
}

void Position::print_attacked_squares(Color c) const {
    std::cout << "\nAttacked squares:\n\n";

    for (Rank rank = RANK_8; rank <= RANK_1; ++rank) {
        for (File file = FILE_A; file <= FILE_H; ++file) {
            Square sq = make_square(file, rank);
            std::string s = is_square_attacked(sq, c) ? " 1" : " 0";

            std::cout << s;
        }

        std::cout << "  " << (8 - rank) << " \n";
    }

    // print files
    std::cout << "\n a b c d e f g h \n\n";
}

// Builds the hash from scratch; used to seed set() and, in debug builds,
// to verify the incremental updates in do_move
BITBOARD Position::compute_key() const {
    BITBOARD k = 0;

    for (Square s = A8; s <= H1; ++s) {
        if (get_piece_on(s) != NO_PIECE) {
            k ^= Zobrist::psq[get_piece_on(s)][s];
        }
    }

    if (ep_square() != NONE) {
        k ^= Zobrist::en_passant[file_of(ep_square())];
    }

    k ^= Zobrist::castling[move_info->castling_rights];

    if (side == BLACK) {
        k ^= Zobrist::side;
    }

    return k;
}

// Shows a bitboard of the possible pieces that can give check to the opposite
// king in a given position
BITBOARD Position::get_checked_squares(PieceType pt) const {
    Square ksq = square<KING>(~side);
    BITBOARD all = get_all_pieces_bb();

    switch (pt) {
        case PAWN:
            return pawn_attacks_bb(~side, ksq);
        case KNIGHT:
            return attacks_bb_by<KNIGHT>(ksq);
        case BISHOP:
            return attacks_bb_by<BISHOP>(ksq, all);
        case ROOK:
            return attacks_bb_by<ROOK>(ksq, all);
        case QUEEN:
            return attacks_bb_by<QUEEN>(ksq, all);

        default:  // KING
            return 0ULL;
    }
}

// This function helps the move generation to determine
// if move in the current position gives a check
// Tests if pseudo-legal move gives check
bool Position::gives_check(Move m) const {
    assert(m.is_move_ok());
    assert(get_piece_color(moved_piece(m)) == side);

    Color us = side;

    Square source = m.source_square();
    Square target = m.target_square();

    Square opp_king_square = square<KING>(~us);

    // Cases are calculated after making a move and test whether a
    // pseudo-legal move gives a check

    // Direct check
    // Get the attacks from the piece that is on the source square
    // then intersect with the target
    // if its king, result will be greater than 0, hence a check
    // else result will be 0, hence not a check
    if (get_threats(type_of_piece(get_piece_on(source))) & target) {
        return true;
    }

    // Discovered check
    //  Get the pieces that block cheks (that are pinned)
    //  then return true if they are not aligned or if we are castling

    //	Already checked if a possible true result
    //  is not caused by direct check of sliding capture
    if (get_king_blockers(~us) & source) {
        return !are_squares_aligned(source, target, opp_king_square) ||
               m.move_type() == MT_CASTLING;
    }

    // On move types
    // In case of NORMAL move (a check cannot be given)
    // In case of PROMOTION
    // In case of EN_PASSANT
    //		could be en_passant capture with check
    //		(may fall into the discovered checks category)
    //		could be a discovered check with a captutured pawn
    // In case of CASLTLE

    switch (m.move_type()) {
            // If the move is any move that does not result in attack
            // that means the move is not giving check
        case MT_NORMAL:
            return false;

            // Handle the case where the promoted piece checks the king
        case MT_PROMOTION:
            return attacks_bb_by(m.promoted(), target, get_all_pieces_bb() ^ source) &
                   opp_king_square;

            // Handle the en passant capture with check
            // Direct and discovered checks are already handled above
            // so the only thing left to do is
            // handling discovered check through captured pawn
        case MT_EN_PASSANT: {
            Square csq = make_square(file_of(target), rank_of(source));
            BITBOARD bb = (get_all_pieces_bb() ^ source ^ csq) | target;

            BITBOARD rook_from_king_bb =
                attacks_bb_by<ROOK>(opp_king_square, bb) &
                (get_pieces_bb(QUEEN, us) | get_pieces_bb(ROOK, us));
            BITBOARD bishop_from_king_bb =
                attacks_bb_by<BISHOP>(opp_king_square, bb) &
                (get_pieces_bb(QUEEN, us) | get_pieces_bb(BISHOP, us));

            // Return true if sliders are attacking king after the capture
            return rook_from_king_bb | bishop_from_king_bb;
        }
        default:  // CASTLING
        {
            // Castling is encoded as king captures the rook
            Square rook_target = sq_relative_to_side(target > source ? F1 : D1, us);

            return get_threats(ROOK) & rook_target;
        }
    }
}

// Makes a move and saves the information in the Info
// Move is assumed to be legal
void Position::do_move(const Move& m, MoveInfo& new_info) {
    assert(m.is_move_ok());
    assert(&new_info != move_info);

    // Copy the old struct into the new one up to the captured_piece field
    // prev and next are not copied
    std::memcpy(&new_info, move_info, offsetof(struct MoveInfo, captured_piece));

    // Much like a linked list
    // Assign then previous block to be equal to the old struct
    new_info.prev = move_info;

    // then the next on the old to be the current new
    move_info->next = &new_info;

    // and the one we are at to be the current new
    move_info = &new_info;

    fullmove_number++;        // increment on every move, displayed correctly on
                              // get_fen()
    ++move_info->fifty_move;  // will be reset ot 0 in case of capture or pawn
                              // move

    // Incrementally build the hash; side to move always flips
    BITBOARD k = move_info->prev->key ^ Zobrist::side;

    Color us = side;
    Color them = ~us;

    Square source = m.source_square();
    Square target = m.target_square();

    MoveType m_type = m.move_type();

    Piece on_source = get_piece_on(source);
    Piece on_target = get_piece_on(target);
    Piece captured =
        m_type == MT_EN_PASSANT
            ? get_piece(them, PAWN)  // captured piece is opponents pawn
            : on_target;             // can be NO_PIECE or every other piece
                                     // (without KING)

    assert(get_piece_color(on_source) ==
           us);  // moving our piece instead of enemy piece
    assert(captured == NO_PIECE ||
           get_piece_color(captured) == (m_type != MT_CASTLING ? them : us));
    assert(type_of_piece(captured) != KING);  // make sure king is not captured

    // Castling
    if (m_type == MT_CASTLING) {
        assert(on_source == get_piece(us, KING));  // source piece is our king
        // since castling is encoded as "King captures rook"
        assert(captured == get_piece(us, ROOK));  // captured piece is rook

        Square r_source, r_target;
        do_castle<true>(us, source, target, r_source, r_target);

        k ^= Zobrist::psq[on_source][source] ^ Zobrist::psq[on_source][target];
        k ^= Zobrist::psq[captured][r_source] ^ Zobrist::psq[captured][r_target];

        captured = NO_PIECE;
    }

    // Captures
    if (captured) {
        Square capture_sq = target;

        if (type_of_piece(captured) == PAWN && m_type == MT_EN_PASSANT) {
            capture_sq -= pawn_push_direction(us);

            assert(on_source == get_piece(us, PAWN));
            assert(target == move_info->en_passant);
            assert(rank_relative_to_side(us, rank_of(target)) == RANK_6);
            assert(on_target == NO_PIECE);
            assert(get_piece_on(capture_sq) == get_piece(them, PAWN));
        }

        remove_piece(capture_sq);
        k ^= Zobrist::psq[captured][capture_sq];

        move_info->fifty_move = 0;
    }

    // Reset en passant square
    if (move_info->en_passant != NONE) {
        k ^= Zobrist::en_passant[file_of(move_info->en_passant)];
        move_info->en_passant = NONE;
    }

    k ^= Zobrist::castling[move_info->castling_rights];
    // Update castling rights if needed
    move_info->castling_rights =
        move_info->castling_rights & CASTLING_RIGHTS_TABLE[source];
    move_info->castling_rights =
        move_info->castling_rights & CASTLING_RIGHTS_TABLE[target];

    k ^= Zobrist::castling[move_info->castling_rights];

    if (m_type != MT_CASTLING) {
        move_piece(source, target);
        k ^= Zobrist::psq[on_source][source] ^ Zobrist::psq[on_source][target];
    }

    if (type_of_piece(on_source) == PAWN) {
        // Set an en passant square if the moved pawn can be captured
        if ((std::int32_t(target) ^ std::int32_t(source)) == 16  // double push
            && (pawn_attacks_bb(us, target - pawn_push_direction(us)) &
                get_pieces_bb(PAWN, them))) {
            move_info->en_passant = target - pawn_push_direction(us);
            k ^= Zobrist::en_passant[file_of(move_info->en_passant)];
        } else if (m_type == MT_PROMOTION) {
            Piece promoted = get_piece(us, m.promoted());

            assert(rank_relative_to_side(us, rank_of(target)) ==
                   RANK_8);  // promoting on the correct rank
            assert(type_of_piece(promoted) >= KNIGHT &&
                   type_of_piece(promoted) <= QUEEN);

            // erase the old piece and put the new one
            remove_piece(target);
            place_piece(promoted, target);

            k ^= Zobrist::psq[on_source][target] ^ Zobrist::psq[promoted][target];
        }

        move_info->fifty_move = 0;
    }

    move_info->captured_piece = captured;

    side = ~side;

    move_info->key = k;

    assert(move_info->key == compute_key());

    calculate_threats();
}

void Position::undo_move(const Move& m) {
    assert(m.is_move_ok());

    side = ~side;  // flip side

    Color us = side;

    Square source = m.source_square();
    Square target = m.target_square();

    Piece on_target = get_piece_on(target);

    MoveType mt = m.move_type();

    assert(is_empty(source) || mt == MT_CASTLING);
    assert(type_of_piece(move_info->captured_piece) != KING);

    if (mt == MT_PROMOTION) {
        assert(rank_relative_to_side(us, rank_of(target)) == RANK_8);
        assert(type_of_piece(on_target) == m.promoted());
        assert(type_of_piece(on_target) >= KNIGHT &&
               type_of_piece(on_target) <= QUEEN);

        remove_piece(target);
        on_target = get_piece(us, PAWN);
        place_piece(on_target,
                    target);  // place pawn on the 8th rank, will be moved later
    }

    if (mt == MT_CASTLING) {
        Square r_source, r_target;
        do_castle<false>(us, source, target, r_source, r_target);
    } else {
        move_piece(target, source);

        if (move_info->captured_piece) {
            Square cap_sq = target;

            if (mt == MT_EN_PASSANT) {
                cap_sq -= pawn_push_direction(us);

                assert(type_of_piece(on_target) == PAWN);
                assert(target == move_info->prev->en_passant);
                assert(rank_relative_to_side(us, rank_of(target)) == RANK_6);
                assert(get_piece_on(cap_sq) == NO_PIECE);
                assert(move_info->captured_piece == get_piece(~us, PAWN));
            }

            assert(get_piece_color(move_info->captured_piece) == ~us);
            place_piece(move_info->captured_piece, cap_sq);
        }
    }

    // return to the previous state
    move_info = move_info->prev;
    fullmove_number--;

    calculate_threats();
}

// Passes the turn without moving: only side to move and en passant
// change. Used by null-move pruning; never call while in check
void Position::do_null_move(MoveInfo& new_info) {
    assert(&new_info != move_info);

    std::memcpy(&new_info, move_info, offsetof(struct MoveInfo, captured_piece));

    new_info.prev = move_info;
    move_info->next = &new_info;
    move_info = &new_info;

    fullmove_number++;
    move_info->fifty_move = 0;

    BITBOARD k = move_info->prev->key ^ Zobrist::side;

    if (move_info->en_passant != NONE) {
        k ^= Zobrist::en_passant[file_of(move_info->en_passant)];
        move_info->en_passant = NONE;
    }

    move_info->captured_piece = NO_PIECE;

    side = ~side;
    move_info->key = k;

    assert(move_info->key == compute_key());

    calculate_threats();
}

void Position::undo_null_move() {
    side = ~side;

    move_info = move_info->prev;
    fullmove_number--;

    calculate_threats();
}

bool Position::is_legal(Move m) const {
    assert(m.is_move_ok());

    Color us = side;
    Square source = m.source_square();
    Square target = m.target_square();
    Square ksq = square<KING>(us);

    MoveType mt = m.move_type();

    BITBOARD opp_queen = get_pieces_bb(QUEEN, ~us);

    assert(get_piece_color(moved_piece(m)) == us);
    assert(get_piece_on(square<KING>(us)) == get_piece(us, KING));

    // This case is tested after removing the enemy pawn and moving our
    // pawn Checking for discovered check
    if (mt == MT_EN_PASSANT) {
        assert(target == move_info->en_passant);
        assert(moved_piece(m) == get_piece(us, PAWN));

        Square cap_sq = target - pawn_push_direction(us);

        assert(get_piece_on(cap_sq) == get_piece(~us, PAWN));
        assert(get_piece_on(target) == NO_PIECE);

        // Remove our pawn and cap square. Add the target square to occupancies
        BITBOARD occ = get_all_pieces_bb() ^ source ^
                       cap_sq ^ target;

        BITBOARD r_att =
            attacks_bb_by<ROOK>(ksq, occ) & (get_pieces_bb(ROOK, ~us) | opp_queen);
        BITBOARD b_att = attacks_bb_by<BISHOP>(ksq, occ) &
                         (get_pieces_bb(BISHOP, ~us) | opp_queen);

        // if such attack after removing the pieces does not exist
        // then the move is legal, hence return true
        return !(r_att | b_att);
    }
    // Not castling into check
    // And castling path is clear
    if (mt == MT_CASTLING) {
        target = sq_relative_to_side(target > source ? G1 : C1, us);
        Direction step = target > source ? LEFT : RIGHT;

        for (Square s = target; s != source; s += step) {
            if (is_square_attacked(s, ~us)) {
                return false;
            }
        }

        return true;
    }

    // moving the king
    if (type_of_piece(get_piece_on(source)) == KING) {
        // Check if the target square is attacked by the enemy
        if (is_square_attacked(target, ~side)) {
            return false;
        }

        // Check if moving the king exposes it to attacks from sliding pieces
        BITBOARD occ = get_all_pieces_bb() ^ source;  // remove the king
        // Get the rook and bishop attacks
        BITBOARD b1 = attacks_bb_by<ROOK>(target, occ) &
                      (get_pieces_bb(ROOK, ~us) | opp_queen);
        BITBOARD b2 = attacks_bb_by<BISHOP>(target, occ) &
                      (get_pieces_bb(BISHOP, ~us) | opp_queen);

        return !(b1 | b2);
    }

    // Since king exposing to checks is handled
    // Other cases are:

    // Capture of checking piece. The captured piece is NOT absoliutely pinned
    // Moving along the direction, towards or away from the king
    return !(get_king_blockers(us) & source) ||
           are_squares_aligned(source, target, ksq);
}

// Helper for do/undo castling move
template <bool Do>
void Position::do_castle(Color us, Square source, Square& target,
                         Square& r_source, Square& r_target) {
    bool king_side = target > source;

    r_source = target;
    r_target = sq_relative_to_side(king_side ? F1 : D1, us);
    target = sq_relative_to_side(king_side ? G1 : C1, us);

    // Remove both pieces
    remove_piece(Do ? source : target);
    remove_piece(Do ? r_source : r_target);

    // Remove piece does not do this
    piece_board[Do ? source : target] = NO_PIECE;
    piece_board[Do ? r_source : r_target] = NO_PIECE;

    place_piece(get_piece(us, KING), Do ? target : source);
    place_piece(get_piece(us, ROOK), Do ? r_target : r_source);
}

BITBOARD Position::get_least_valuable_piece(BITBOARD attacks, Color by_side,
                                            PieceType& pt) const {
    for (pt = PAWN; pt <= KING; ++pt) {
        // subset contains the attacks of the pieces
        // intersected with the current piece
        BITBOARD subset = attacks & get_pieces_bb(pt, by_side);

        // Order of the pieces in the enumerator are
        // from least to most valuable piece
        if (subset) {
            // return this piece's bitboard
            return subset & -subset;  // single bit
        }
    }

    // the set is empty
    return 0;
}

void Position::set_castling_rights(Color c, Square r_source) {
    Square k_source = square<KING>(c);

    CastlingRights cr = c & (k_source < r_source ? KINGSIDE : QUEENSIDE);

    move_info->castling_rights |= cr;

    rook_source_sq[cr] = r_source;

    Square r_target = sq_relative_to_side(cr & KINGSIDE ? F1 : D1, c);
    Square k_target = sq_relative_to_side(cr & KINGSIDE ? G1 : C1, c);

    BITBOARD k_and_r = square_to_BB(k_source) | r_source;

    castling_path[cr] =
        (in_between_bb(r_source, r_target) | in_between_bb(k_source, k_target)) &
        ~k_and_r;
}

void Position::refresh_accumulator() {
    if (!nnue::is_loaded()) {
        return;
    }

    nnue::accumulator_reset(acc);

    BITBOARD occupied = get_all_pieces_bb();
    while (occupied) {
        Square s = pop_ls1b(occupied);
        nnue::accumulator_add(acc, get_piece_on(s), s);
    }
}

void Position::calculate_threats() {
    update_blocks_and_pins(WHITE);
    update_blocks_and_pins(BLACK);

    Square ksq = square<KING>(~side);

    BITBOARD all = get_all_pieces_bb();

    threats[PAWN] = pawn_attacks_bb(~side, ksq);
    threats[KNIGHT] = attacks_bb_by<KNIGHT>(ksq);
    threats[BISHOP] = attacks_bb_by<BISHOP>(ksq, all);
    threats[ROOK] = attacks_bb_by<ROOK>(ksq, all);
    threats[QUEEN] = threats[BISHOP] | threats[ROOK];
    threats[KING] = 0;  // Can't have threats by king
}

void Position::update_blocks_and_pins(Color c) {
    blocking_pieces[c] = 0ULL;
    pinning_pieces[~c] = 0ULL;

    Square ksq = square<KING>(c);

    BITBOARD color_pieces = get_pieces_bb(c);

    // snipers are calculated such that there are no pieces on the board
    // to get the line between the king and the slider
    BITBOARD snipers =
        ((attacks_bb_by<ROOK>(ksq) & get_pieces_bb(ROOK, QUEEN)) |
         (attacks_bb_by<BISHOP>(ksq) & get_pieces_bb(BISHOP, QUEEN))) &
        get_pieces_bb(~c);

    // All pieces without the snipers
    BITBOARD occ = get_all_pieces_bb() ^ snipers;

    // looping through all the snipers
    while (snipers) {
        // getting the sniper's square and popping its bit
        Square sniper_sq = pop_ls1b(snipers);
        BITBOARD btw = in_between_bb(ksq, sniper_sq) & occ;

        // if there is space between the king square and the slider
        // and is still left space after removing a bit
        // then there is greater than or equal to one blocker between them
        if (btw && !has_bit_after_pop(btw)) {
            blocking_pieces[c] |= btw;

            // If the blocking piece is of the opposite colour
            // the blocker is pinned by the pinner (the sniper)
            if (btw & color_pieces) {
                pinning_pieces[~c] |= sniper_sq;
            }
        }
    }
}

// Fifty-move rule and (single) repetition detection. Scoring the first
// repetition as a draw is intentional: if repeating once is best play,
// repeating twice more changes nothing
bool Position::is_draw() const {
    if (move_info->fifty_move >= 100) {
        return true;
    }

    // Only positions since the last irreversible move (capture or pawn
    // move) can repeat — fifty_move is exactly that distance. Step 2 plies
    // at a time: a repetition needs the same side to move
    MoveInfo* info = move_info;

    for (std::int32_t i = 2; i <= move_info->fifty_move; i += 2) {
        info = info->prev ? info->prev->prev : nullptr;

        if (!info) {
            break;
        }

        if (info->key == move_info->key) {
            return true;
        }
    }

    return false;
}

// Exchange values for SEE. Index 0 is NO_PIECE_TYPE, so an empty target
// square (a quiet move) naturally prices at zero. The king's value is
// never read: a king "capture" exits the swap loop before pricing it
constexpr Value SEE_VALUE[PIECE_TYPE_NB] = {0, 100, 300, 300, 500, 900, 0, 0};

// Plays out the capture sequence on the target square, both sides always
// recapturing with their least valuable attacker. Sliders uncovered by a
// departing attacker join the exchange (x-rays). Pins are ignored - the
// rare misjudged exchange is the price of keeping the loop this cheap
bool Position::see_ge(Move m, Value threshold) const {
    // Castling never wins material; en passant and promotions are rare
    // enough to price as "breaks even"
    if (m.move_type() != MT_NORMAL) {
        return threshold <= 0;
    }

    Square from = m.source_square();
    Square to = m.target_square();

    // Best case: capture the victim and never get recaptured. Failing
    // the threshold even then is an immediate no
    Value swap = SEE_VALUE[type_of_piece(get_piece_on(to))] - threshold;
    if (swap < 0) {
        return false;
    }

    // Worst case: the mover is lost for nothing further. Winning even
    // then is an immediate yes
    swap = SEE_VALUE[type_of_piece(get_piece_on(from))] - swap;
    if (swap <= 0) {
        return true;
    }

    BITBOARD occupied =
        get_all_pieces_bb() ^ square_to_BB(from) ^ square_to_BB(to);
    Color stm = get_piece_color(get_piece_on(from));
    BITBOARD attackers = get_attackers_to(to, occupied);
    std::int32_t res = 1;  // 1 while the side that just moved stands to win

    while (true) {
        stm = ~stm;
        attackers &= occupied;  // spent attackers no longer participate

        BITBOARD stm_attackers = attackers & get_pieces_bb(stm);

        // No attacker left: stm loses the exchange
        if (!stm_attackers) {
            break;
        }

        res ^= 1;

        // Capture with the least valuable attacker, remove it from the
        // board, and let any slider hiding behind it join in (x-ray)
        BITBOARD bb;
        if ((bb = stm_attackers & get_pieces_bb(PAWN))) {
            if ((swap = SEE_VALUE[PAWN] - swap) < res) {
                break;
            }
            occupied ^= square_to_BB(get_ls1b(bb));
            attackers |= attacks_bb_by<BISHOP>(to, occupied) &
                         get_pieces_bb(BISHOP, QUEEN);
        } else if ((bb = stm_attackers & get_pieces_bb(KNIGHT))) {
            if ((swap = SEE_VALUE[KNIGHT] - swap) < res) {
                break;
            }
            occupied ^= square_to_BB(get_ls1b(bb));
        } else if ((bb = stm_attackers & get_pieces_bb(BISHOP))) {
            if ((swap = SEE_VALUE[BISHOP] - swap) < res) {
                break;
            }
            occupied ^= square_to_BB(get_ls1b(bb));
            attackers |= attacks_bb_by<BISHOP>(to, occupied) &
                         get_pieces_bb(BISHOP, QUEEN);
        } else if ((bb = stm_attackers & get_pieces_bb(ROOK))) {
            if ((swap = SEE_VALUE[ROOK] - swap) < res) {
                break;
            }
            occupied ^= square_to_BB(get_ls1b(bb));
            attackers |= attacks_bb_by<ROOK>(to, occupied) &
                         get_pieces_bb(ROOK, QUEEN);
        } else if ((bb = stm_attackers & get_pieces_bb(QUEEN))) {
            if ((swap = SEE_VALUE[QUEEN] - swap) < res) {
                break;
            }
            occupied ^= square_to_BB(get_ls1b(bb));
            attackers |= (attacks_bb_by<BISHOP>(to, occupied) &
                          get_pieces_bb(BISHOP, QUEEN)) |
                         (attacks_bb_by<ROOK>(to, occupied) &
                          get_pieces_bb(ROOK, QUEEN));
        } else {
            // Only the king is left to recapture: legal only if the other
            // side has no attacker remaining, else it walks into a capture
            return (attackers & ~get_pieces_bb(stm)) ? (res ^ 1) != 0
                                                     : res != 0;
        }
    }

    return res != 0;
}

}  // namespace KhaosChess
