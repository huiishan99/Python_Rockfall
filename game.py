import pygame
import random
import json

# 初始化pygame
pygame.init()
game_time = 0
game_data = []  # 用于存放游戏数据

# 设置屏幕尺寸
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 进度条设置
difficulty_level = 1
max_difficulty_level = 10
progress_bar_length = 200  # 进度条长度，单位为像素
progress_bar_height = 20
progress_color = (0, 0, 255)  # 蓝色进度条
progress_bar_x = screen_width - 210  # 右上角位置，考虑到长度和一点边距
progress_bar_y = 10  # 顶部边距

# 玩家设置
player_width = 50
player_height = 50
player_x = screen_width // 2 - player_width // 2
player_y = screen_height - player_height - 10
player_color = (255, 255, 255)
player_speed = 5
lives = 10  # 玩家初始生命值

# 障碍物设置
obstacle_width = 50
obstacle_height = 50
obstacle_color = (255, 0, 0)
obstacles = []
obstacle_speed = 5
obstacle_frequency = 25

frame_count = 0

# 设置字体用于显示生命值
font = pygame.font.Font(None, 36)

# 设置字体用于显示进度条标签
level_text = font.render("Level:", True, (255, 255, 255))  # 创建标签文本

# 游戏主循环
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    current_state = {
        'player_x': player_x,
        'obstacles': [(obstacle[0], obstacle[1]) for obstacle in obstacles]
    }

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        player_x -= player_speed
        game_data.append({'state': current_state, 'action': 'left'})

    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        player_x += player_speed
        game_data.append({'state': current_state, 'action': 'right'})

    # 更新游戏时间和障碍物速度
    game_time += 1
    if game_time % 600 == 0:
        obstacle_speed += 1
        difficulty_level = min(difficulty_level + 1, max_difficulty_level)

    player_rect = pygame.Rect(player_x, player_y, player_width, player_height)

    # 更新障碍物生成
    frame_count += 1
    if frame_count % obstacle_frequency == 0:
        obstacles.append([random.randint(0, screen_width - obstacle_width), -obstacle_height])

    # 更新障碍物位置并检测碰撞
    obstacles = [obstacle for obstacle in obstacles if obstacle[1] < screen_height]
    for obstacle in obstacles:
        obstacle[1] += obstacle_speed
        obstacle_rect = pygame.Rect(obstacle[0], obstacle[1], obstacle_width, obstacle_height)
        if player_rect.colliderect(obstacle_rect):
            lives -= 1
            obstacles.remove(obstacle)
            if lives == 0:
                running = False

    # 清空屏幕
    screen.fill((0, 0, 0))

    # 绘制玩家
    pygame.draw.rect(screen, player_color, player_rect)

    # 绘制障碍物
    for obstacle in obstacles:
        pygame.draw.rect(screen, obstacle_color, pygame.Rect(obstacle[0], obstacle[1], obstacle_width, obstacle_height))

    # 显示生命值
    lives_text = font.render(f"Lives: {lives}", True, (255, 255, 255))
    screen.blit(lives_text, (10, 10))

    # 绘制进度条和标签显示难度
    progress = (difficulty_level / max_difficulty_level) * progress_bar_length
    pygame.draw.rect(screen, (255, 255, 255),
                     (progress_bar_x, progress_bar_y, progress_bar_length, progress_bar_height), 1)
    pygame.draw.rect(screen, progress_color, (progress_bar_x, progress_bar_y, progress, progress_bar_height))
    screen.blit(level_text, (progress_bar_x - level_text.get_width() - 10, progress_bar_y))

    # 更新屏幕显示
    pygame.display.flip()

    # 控制游戏更新速率
    clock.tick(30)

# 游戏主循环结束后保存数据
with open('game_data.json', 'w') as f:
    json.dump(game_data, f)
    print("Data has been saved to game_data.json")

# 退出pygame
pygame.quit()
