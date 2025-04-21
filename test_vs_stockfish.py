import chess
import chess.engine
import time
from ai import AI
from queue import Queue
import math

# === CONFIG ===
STOCKFISH_PATH = r"E:\CCO VUAAAAAAAAAAAAAAA\stockfish\stockfish-windows-x86-64-avx2.exe"
NUM_GAMES = 20
STOCKFISH_ELO = 1800  # 👈 ELO đối thủ cố định
STOCKFISH_TIME_LIMIT = 0.1  # mỗi nước (giây)

# === THỐNG KÊ ===
ai_wins = 0
draws = 0
sf_wins = 0

for game_num in range(1, NUM_GAMES + 1):
    board = chess.Board()
    ai = AI()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})

    ai_color = chess.WHITE if game_num % 2 == 1 else chess.BLACK
    print(f"\n🔄 Game {game_num}: AI chơi {'trắng' if ai_color == chess.WHITE else 'đen'}")

    while not board.is_game_over():
        if board.turn == ai_color:
            # AI đánh
            q = Queue()
            ai.run_search_process(board.copy(), q)
            try:
                move = q.get(timeout=15)
                if move not in board.legal_moves:
                    print("❌ AI chọn nước không hợp lệ!")
                    break
                board.push(move)
            except:
                print("❌ AI không chọn được nước đi!")
                break
        else:
            # Stockfish đánh
            result = engine.play(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
            board.push(result.move)

    engine.quit()

    # Kết quả
    result = board.result()
    print("📌 Kết quả:", result)

    if result == "1-0":
        if ai_color == chess.WHITE:
            ai_wins += 1
        else:
            sf_wins += 1
    elif result == "0-1":
        if ai_color == chess.BLACK:
            ai_wins += 1
        else:
            sf_wins += 1
    else:
        draws += 1

# === THỐNG KẾ CUỐI CÙNG ===
print("\n===== 📊 TỔNG KẾT SAU 20 VÁN =====")
print(f"✅ AI thắng: {ai_wins}")
print(f"🤝 Hòa    : {draws}")
print(f"❌ Thua   : {sf_wins}")

# === ƯỚC LƯỢNG ELO ===
games_played = ai_wins + sf_wins + draws
if games_played > 0:
    win_rate = (ai_wins + 0.5 * draws) / games_played

    def estimate_elo(opponent_elo, win_rate):
        if win_rate <= 0:
            return opponent_elo - 400
        elif win_rate >= 1:
            return opponent_elo + 400
        return int(opponent_elo + 400 * math.log10(win_rate / (1 - win_rate)))

    estimated_elo = estimate_elo(STOCKFISH_ELO, win_rate)
    print(f"\n📈 ELO ước tính của AI so với Stockfish {STOCKFISH_ELO}: ~{estimated_elo}")
else:
    print("\n⚠️ Không có đủ dữ liệu để ước tính ELO.")
