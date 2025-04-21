import pygame
import chess

WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)

PIECE_IMAGES = {}
PIECES = ['p', 'r', 'n', 'b', 'q', 'k', 'p1', 'r1', 'n1', 'b1', 'q1', 'k1']
for piece in PIECES:
    try:
        PIECE_IMAGES[piece] = pygame.image.load(f'assets/{piece}.png')
        PIECE_IMAGES[piece] = pygame.transform.scale(PIECE_IMAGES[piece], (SQUARE_SIZE, SQUARE_SIZE))
    except pygame.error:
        print(f"Không thể load ảnh cho quân {piece}")

def draw_board(screen):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col, row = chess.square_file(square), 7 - chess.square_rank(square)
            symbol = piece.symbol()
            if symbol.isupper():
                symbol = symbol.lower() + "1"
            screen.blit(PIECE_IMAGES[symbol], (col * SQUARE_SIZE, row * SQUARE_SIZE))

def highlight_moves(screen, board, square):
    legal_moves = [move for move in board.legal_moves if move.from_square == square]
    for move in legal_moves:
        col, row = chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square)
        pygame.draw.circle(screen, HIGHLIGHT,
                           (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 10)
        
def draw_promotion_choices(screen, color):
    options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    names = ['q', 'r', 'b', 'n']
    if color == chess.WHITE:
        names = [n + '1' for n in names]

    for i, name in enumerate(names):
        rect = pygame.Rect(i * 75 + 150, 225, 75, 75)
        pygame.draw.rect(screen, (200, 200, 200), rect)
        screen.blit(PIECE_IMAGES[name], (i * 75 + 150, 225))
def get_promotion_choice(pos, color):
    x, y = pos
    if 225 <= y <= 300 and 150 <= x <= 450:
        index = (x - 150) // 75
        options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        return options[index]
    return None
