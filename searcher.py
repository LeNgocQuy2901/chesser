import chess
import time
from evaluate import evaluate_board

IMMEDIATE_MATE_SCORE = 100000
POS_INF = 9999999
NEG_INF = -POS_INF
MAX_DEPTH = 64

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

class TranspositionEntry:
    def __init__(self, zobrist, value, depth, flag, move):
        self.zobrist = zobrist
        self.value = value
        self.depth = depth
        self.flag = flag
        self.move = move

class TranspositionTable:
    EXACT, LOWER, UPPER = 0, 1, 2

    def __init__(self, size=2**20):
        self.table = {}
        self.size = size

    def get(self, key):
        return self.table.get(key)

    def store(self, key, value, depth, flag, move):
        if len(self.table) >= self.size:
            self.table.clear()
        self.table[key] = TranspositionEntry(key, value, depth, flag, move)

class Searcher:
    def __init__(self):
        self.best_move = None
        self.best_eval = 0
        self.stop_search = False
        self.tt = TranspositionTable()
        self.killer_moves = {ply: [] for ply in range(64)}
        self.history = {}
        self.repetition_table = []

        self.start_time = 0
        self.time_limit = 9.5

    def is_mate_score(self, score):
        return abs(score) > IMMEDIATE_MATE_SCORE - 1000

    def score_to_ply(self, score):
        return IMMEDIATE_MATE_SCORE - abs(score)

    def iterative_deepening(self, board, max_depth=5, time_limit=9.5):
        self.stop_search = False
        self.best_eval = 0
        self.best_move = None
        self.start_time = time.time()
        self.time_limit = time_limit

        last_completed_best_move = None
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]  # Nếu chỉ còn 1 nước thì chơi luôn
        for depth in range(1, max_depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break

            self.current_depth = depth
            eval = self.alpha_beta(board, depth, 0, NEG_INF, POS_INF)

            if self.stop_search or (time.time() - self.start_time > self.time_limit):
                break

            self.best_eval = eval
            last_completed_best_move = self.best_move

            if self.is_mate_score(eval) and self.score_to_ply(eval) <= depth:
                break

        return last_completed_best_move if last_completed_best_move else self.best_move

    def alpha_beta(self, board, depth, ply, alpha, beta):
        if time.time() - self.start_time > self.time_limit:
            self.stop_search = True
            return 0

        zobrist = chess.polyglot.zobrist_hash(board)

        if zobrist in self.repetition_table or board.is_repetition(3):
            return 0

        entry = self.tt.get(zobrist)
        if entry and entry.depth >= depth:
            if entry.flag == self.tt.EXACT:
                return entry.value
            elif entry.flag == self.tt.LOWER and entry.value >= beta:
                return entry.value
            elif entry.flag == self.tt.UPPER and entry.value <= alpha:
                return entry.value

        if board.is_checkmate():
            return -IMMEDIATE_MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if depth == 0:
            return self.quiescence(board, alpha, beta)

        legal_moves = list(board.legal_moves)
        best_val = NEG_INF
        best_move = None

        self.repetition_table.append(zobrist)

        tt_move = self.tt.get(zobrist).move if self.tt.get(zobrist) else None

        def move_order_key(move):
            score = 0
            move_uci = move.uci()

            # 1. Hash move
            if tt_move == move:
                return 10_000_000

            # 2. MVV-LVA
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    score += 10_000 + piece_values[captured.piece_type]*10 - piece_values[attacker.piece_type]

            # 3. Promotion
            if move.promotion:
                score += 8_000 if move.promotion == chess.QUEEN else 6_000

            # 4. Killer move
            if move in self.killer_moves.get(ply, []):
                score += 4_000

            # 5. History heuristic
            score += self.history.get(move_uci, 0)

            return score

        ordered_moves = sorted(legal_moves, key=move_order_key, reverse=True)


        for move in ordered_moves:
            if time.time() - self.start_time > self.time_limit:
                self.stop_search = True
                break

            board.push(move)
            val = -self.alpha_beta(board, depth - 1, ply + 1, -beta, -alpha)
            board.pop()

            if self.stop_search:
                break

            if val > best_val:
                best_val = val
                best_move = move
                if ply == 0:
                    self.best_move = move

            alpha = max(alpha, val)
            if alpha >= beta:
                if not board.is_capture(move):
                    if move not in self.killer_moves[ply]:
                        self.killer_moves[ply].append(move)
                        if len(self.killer_moves[ply]) > 2:
                            self.killer_moves[ply] = self.killer_moves[ply][-2:]
                self.history[move.uci()] = self.history.get(move.uci(), 0) + depth * depth
                break

        self.repetition_table.pop()

        flag = self.tt.EXACT
        if best_val <= alpha:
            flag = self.tt.UPPER
        elif best_val >= beta:
            flag = self.tt.LOWER
        self.tt.store(zobrist, best_val, depth, flag, best_move)

        return best_val

    def quiescence(self, board, alpha, beta):
        if time.time() - self.start_time > self.time_limit:
            return evaluate_board(board)

        stand_pat = evaluate_board(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in board.legal_moves:
            if time.time() - self.start_time > self.time_limit:
                return stand_pat

            if board.is_capture(move) or move.promotion:
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha)
                board.pop()

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

        return alpha

    def move_score(self, board, move, ply):
        score = 0
        move_uci = move.uci()

        # Transposition Table Move Boost
        tt_entry = self.tt.get(chess.polyglot.zobrist_hash(board))
        if tt_entry and tt_entry.move == move:
            return 10_000_000

        # Capture Heuristics (MVV-LVA)
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            capturing_piece = board.piece_at(move.from_square)
            if captured_piece and capturing_piece:
                captured_value = piece_values[captured_piece.piece_type]
                attacker_value = piece_values[capturing_piece.piece_type]
                score += 10_000 + captured_value * 10 - attacker_value

        # Promotion Bonus
        if move.promotion:
            score += 8000 if move.promotion == chess.QUEEN else 6000

        # Killer Move Bonus
        if move in self.killer_moves.get(ply, []):
            score += 4000

        # History Heuristic
        score += self.history.get(move_uci, 0)

        return score