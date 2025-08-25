import pygame
import sys

# Cấu hình
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 40
SCREEN_SIZE = CELL_SIZE * BOARD_SIZE + MARGIN * 2

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_COLOR = (50, 50, 50)
X_COLOR = (255, 100, 100)
O_COLOR = (100, 100, 255)
BG_COLOR = (240, 217, 181)

# Đọc file log
def read_moves_from_file(path):
    moves = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                x, y = int(parts[0]), int(parts[1])
                moves.append((x - 1, y - 1))  # Chuyển về 0-index
            except ValueError:
                continue
    return moves

# Vẽ bàn cờ
def draw_board(screen):
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR,
                         (MARGIN, MARGIN + i * CELL_SIZE),
                         (MARGIN + CELL_SIZE * (BOARD_SIZE - 1), MARGIN + i * CELL_SIZE), 1)
        pygame.draw.line(screen, LINE_COLOR,
                         (MARGIN + i * CELL_SIZE, MARGIN),
                         (MARGIN + i * CELL_SIZE, MARGIN + CELL_SIZE * (BOARD_SIZE - 1)), 1)

list_of_move = []

# Vẽ nước đi
def draw_moves(screen, moves):
    font = pygame.font.SysFont(None, 20)
    for idx, (x, y) in enumerate(moves):
        if (x,y) in list_of_move:
            print("bi trung")
        list_of_move.append((x,y))
        center = (MARGIN + x * CELL_SIZE, MARGIN + y * CELL_SIZE)
        color = X_COLOR if idx % 2 == 0 else O_COLOR
        pygame.draw.circle(screen, color, center, CELL_SIZE // 3)

        # Số thứ tự
        text = font.render(str(idx + 1), True, BLACK)
        text_rect = text.get_rect(center=center)
        screen.blit(text, text_rect)
        pygame.display.flip()
        clock.tick(2)


pygame.init()
clock = pygame.time.Clock()

# Hàm chính
def main():
    
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Gomoku Replay")
    

    moves = read_moves_from_file("dataTrain/Freestyle15_1/11_9_10_1.psq")

    running = True
    while running:
        screen.fill(BG_COLOR)
        draw_board(screen)
        draw_moves(screen, moves)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Thêm chức năng tạm dừng bằng phím cách (Space)
        # Tạm dừng chương trình tại đây, không cần ấn phím
        pygame.display.update()
        print("Đã phát lại xong. Nhấn Enter để thoát...")
        input()
        running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
