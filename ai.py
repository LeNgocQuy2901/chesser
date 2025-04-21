import chess
import time
import chess.polyglot
from threading import Thread
from queue import Queue
from searcher import Searcher  
class AI:
    def __init__(self):
        self.move = None

    def run_search_process(self, board_state, return_queue):
            def search_and_update():
                try:
                    start = time.time()

                    # ⚡ Ưu tiên tìm trong sách khai cuộc trước
                    move = None
                    try:
                        with chess.polyglot.open_reader("baron30.bin") as reader:
                            try:
                                entry = reader.find(board_state)
                                move = entry.move
                                print(f"[AI] Sử dụng sách khai cuộc: {move}")
                            except IndexError:
                                pass
                    except FileNotFoundError:
                        print("[Lỗi] Không tìm thấy baron30.bin")

                    # ❌ Nếu không tìm thấy, dùng Searcher
                    if move is None:
                        searcher = Searcher()
                        move = searcher.iterative_deepening(board_state, max_depth=6, time_limit=9)
                        print(f"[AI] Đã chọn nước đi: {move} trong {time.time() - start:.2f} giây")

                    self.move = move
                    if return_queue:
                        return_queue.put(move)

                except Exception as e:
                    print("[Lỗi] Lỗi trong tìm kiếm:", e)
                    self.move = None
                    if return_queue:
                        return_queue.put(None)

            Thread(target=search_and_update).start()

    def update_ai_move(self, game, board_state):
        self.move = None
        return_queue = Queue()

        try:
            with chess.polyglot.open_reader("baron30.bin") as reader:
                try:
                    entry = reader.find(board_state)
                    self.move = entry.move
                    print(f"[AI] Sử dụng sách khai cuộc: {self.move}")
                    game.board.push(self.move)
                    game.history_index = len(game.board.move_stack)
                    game.view_board = game.board.copy()
                    game.check_game_end()
                    if hasattr(game, 'root'):
                        game.root.title("Cờ vua - Đã xong")
                    return
                except IndexError:
                    print("[AI] Không tìm thấy nước đi trong sách khai cuộc. Dùng Searcher.")
        except FileNotFoundError:
            print("[Lỗi] Không tìm thấy tệp baron30.bin. Dùng Searcher.")
        except Exception as e:
            print("[Lỗi] Lỗi khi mở sách khai cuộc:", e)

        self.run_search_process(board_state, return_queue)

        def check_result():
            if self.move and self.move in game.board.legal_moves:
                game.board.push(self.move)
                game.history_index = len(game.board.move_stack)
                game.view_board = game.board.copy()
                game.check_game_end()
                if hasattr(game, 'root'):
                    game.root.title("Cờ vua - Đã xong")
            elif not self.move:
                if hasattr(game, 'root'):
                    game.root.after(50, check_result)
                else:
                    time.sleep(0.05)
                    Thread(target=check_result).start()

        if hasattr(game, 'root'):
            game.root.title("AI đang nghĩ...")
            game.root.after(50, check_result)
        else:
            check_result()