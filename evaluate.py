import chess
import chess
import numpy as np

# Define piece values
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Define piece-square tables
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

PAWN_ENDGAME_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -10,  0,  5,  0,  0,  5,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_START = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

KING_END = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

# Convert piece-square tables to numpy arrays for faster access
PAWN_TABLE = np.array(PAWN_TABLE)
PAWN_ENDGAME_TABLE = np.array(PAWN_ENDGAME_TABLE)
KNIGHT_TABLE = np.array(KNIGHT_TABLE)
BISHOP_TABLE = np.array(BISHOP_TABLE)
ROOK_TABLE = np.array(ROOK_TABLE)
QUEEN_TABLE = np.array(QUEEN_TABLE)
KING_START = np.array(KING_START)
KING_END = np.array(KING_END)
# --- Piece-square tables from SebLague's PieceSquareTable.cs ---
PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]

PAWN_ENDGAME_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
    0, 0, 0, 0, 0, 0, 0, 0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,   0,  5,  5,  5,  5,  0, -5,
     0,   0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_START = [
    -80, -70, -70, -70, -70, -70, -70, -80,
    -60, -60, -60, -60, -60, -60, -60, -60,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,  -5,  -5,  -5,  -5,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20
]

KING_END = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    -10, -5,  20,  30,  30,  20,  -5, -10,
    -15, -10, 35,  45,  45,  35, -10, -15,
    -20, -15, 30,  40,  40,  30, -15, -20,
    -25, -20, 20,  25,  25,  20, -20, -25,
    -30, -25,  0,   0,   0,   0, -25, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]

piece_square_tables = {
    chess.PAWN: (PAWN_TABLE, PAWN_ENDGAME_TABLE),
    chess.KNIGHT: (KNIGHT_TABLE, KNIGHT_TABLE),
    chess.BISHOP: (BISHOP_TABLE, BISHOP_TABLE),
    chess.ROOK: (ROOK_TABLE, ROOK_TABLE),
    chess.QUEEN: (QUEEN_TABLE, QUEEN_TABLE),
    chess.KING: (KING_START, KING_END)
}

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def get_material_info(board, color):
    pieces = board.piece_map()
    value = 0
    num = {pt: 0 for pt in piece_values}
    for sq, pc in pieces.items():
        if pc.color == color:
            num[pc.piece_type] += 1
            value += piece_values[pc.piece_type]
    endgame_score = (
        num[chess.QUEEN] * 45 + num[chess.ROOK] * 20 +
        num[chess.BISHOP] * 10 + num[chess.KNIGHT] * 10
    )
    endgame_weight = max(0, 1 - endgame_score / 152)
    return value, endgame_weight, num

def evaluate_piece_square(board, color, endgame_t):
    value = 0
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc and pc.color == color:
            t_early, t_end = piece_square_tables[pc.piece_type]
            table_val = t_early[sq] * (1 - endgame_t) + t_end[sq] * endgame_t
            value += table_val
    return value

def evaluate_pawns(board, color):
    bonus = 0
    isolated_penalty = [-10, -25, -50, -75, -75, -75, -75, -75, -75]
    passed_bonus = [0, 120, 80, 50, 30, 15, 15, 15]
    pawns = list(board.pieces(chess.PAWN, color))
    other_pawns = board.pieces(chess.PAWN, not color)
    file_pawn_map = {chess.square_file(sq): [] for sq in pawns}
    for sq in pawns:
        file_pawn_map[chess.square_file(sq)].append(sq)
    for sq in pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
        if not any(chess.square_file(op) in [file - 1, file, file + 1] for op in other_pawns):
            bonus += passed_bonus[rank]
        if not any(f in file_pawn_map for f in [file - 1, file + 1] if 0 <= f <= 7):
            bonus += isolated_penalty[len(pawns)]
    return bonus

def king_pawn_shield(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    shield_squares = []
    file = chess.square_file(king_sq)
    rank = chess.square_rank(king_sq)
    dirs = [-1, 0, 1]
    for df in dirs:
        for dr in [1, 2]:
            f = file + df
            r = rank + dr if color == chess.WHITE else rank - dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                shield_squares.append(chess.square(f, r))
    penalty = 0
    for i, sq in enumerate(shield_squares[:3]):
        if board.piece_at(sq) != chess.Piece(chess.PAWN, color):
            penalty += [4, 7, 4][i]
    return -penalty

def mop_up_eval(board, color, my_mat, opp_mat, endgame_t):
    if my_mat > opp_mat + 200 and endgame_t > 0:
        my_king = board.king(color)
        opp_king = board.king(not color)
        if my_king is None or opp_king is None:
            return 0
        dist = abs(chess.square_file(my_king) - chess.square_file(opp_king)) + \
               abs(chess.square_rank(my_king) - chess.square_rank(opp_king))
        center_dist = abs(chess.square_file(opp_king) - 3.5) + abs(chess.square_rank(opp_king) - 3.5)
        return int(((14 - dist) * 4 + center_dist * 10) * endgame_t)
    return 0

def evaluate_board(board):
    if board.is_checkmate():
        return -999999 if board.turn == chess.WHITE else 999999

    white_mat, white_end, white_count = get_material_info(board, chess.WHITE)
    black_mat, black_end, black_count = get_material_info(board, chess.BLACK)

    white_score = white_mat
    black_score = black_mat

    white_score += evaluate_piece_square(board, chess.WHITE, white_end)
    black_score += evaluate_piece_square(board, chess.BLACK, black_end)

    # white_score += evaluate_pawns(board, chess.WHITE)
    # black_score += evaluate_pawns(board, chess.BLACK)

    # white_score += king_pawn_shield(board, chess.WHITE)
    # black_score += king_pawn_shield(board, chess.BLACK)

    # white_score += mop_up_eval(board, chess.WHITE, white_mat, black_mat, black_end)
    # black_score += mop_up_eval(board, chess.BLACK, black_mat, white_mat, white_end)

    eval_score = white_score - black_score
    return eval_score if board.turn == chess.WHITE else -eval_score
