import pygame
from ui import draw_board, draw_pieces, highlight_moves
from game import Game

# Kiểm tra xem module đã được nhập chưa để tránh chạy lại
if __name__ == "__main__":
    WIDTH, HEIGHT = 600, 600
    pygame.init()
    # Kiểm tra xem cửa sổ đã tồn tại chưa
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Game")
    except pygame.error as e:
        print(f"[Lỗi] Không thể tạo cửa sổ mới: {e}")
        pygame.quit()
        exit()

    game = Game()
    running = True

    while running and game.running:  # Kiểm tra cả trạng thái game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                game.handle_event(event)

        screen.fill((0, 0, 0))  # Xóa màn hình trước khi vẽ
        draw_board(screen)  # Vẽ bàn cờ
        draw_pieces(screen, game.view_board)  # Vẽ quân cờ từ view_board
        if game.selected_square is not None and game.history_index == len(game.board.move_stack):
            highlight_moves(screen, game.view_board, game.selected_square)

        game.update_ai_move()

        if game.history_index < len(game.board.move_stack):
            pygame.display.set_caption(f"Chess Game - Đang xem lại bước {game.history_index}/{len(game.board.move_stack)}")
        else:
            pygame.display.set_caption("Chess Game")

        pygame.display.flip()
        game.clock.tick(60)

    pygame.quit()