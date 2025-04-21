import sys
import chess
import chess.engine
from ai import get_best_move


class ChessEngine:
    def __init__(self):
        self.board = chess.Board()

    def handle_uci_command(self, command):
        """
        Xử lý các lệnh UCI.
        """
        if command.startswith("uci"):
            self.uci_ready()
        elif command.startswith("isready"):
            self.is_ready()
        elif command.startswith("position"):
            self.set_position(command)
        elif command.startswith("go"):
            self.go(command)
        elif command.startswith("quit"):
            self.quit()
        else:
            print(f"UCI: Không nhận diện được lệnh: {command}")

    def uci_ready(self):
        """Lệnh khi engine đã sẵn sàng."""
        print("uciok")

    def is_ready(self):
        """Lệnh kiểm tra nếu engine sẵn sàng để nhận lệnh mới."""
        print("readyok")

    def set_position(self, command):
        """Lệnh để thiết lập vị trí cờ."""
        parts = command.split(" ")
        if "startpos" in parts:
            self.board = chess.Board()
        elif "fen" in parts:
            fen_position = parts[2]
            self.board.set_fen(fen_position)

    def go(self, command):
        """Lệnh yêu cầu engine tìm nước đi tốt nhất."""
        print("go")
        best_move_found = get_best_move(self.board, depth=10)
        print(f"bestmove {best_move_found}")

    def quit(self):
        """Lệnh để kết thúc engine."""
        sys.exit()


def main():
    engine = ChessEngine()

    # Đọc đầu vào từ STDIN
    for line in sys.stdin:
        line = line.strip()
        engine.handle_uci_command(line)


if __name__ == "__main__":
    main()
