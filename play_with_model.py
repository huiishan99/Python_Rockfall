import pygame
import random
import joblib
import numpy as np

# 初始化pygame
pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 加载模型
model = joblib.load('game_model.pkl')

# 玩家设置
player_width = 50
player_height = 50
player_x = screen_width // 2 - player_width // 2
player_y = screen_height - player_height - 10
player_speed = 5

# 障碍物设置
obstacle_width = 50
obstacle_height = 50
obstacles = []
obstacle_speed = 5
obstacle_frequency = 25
frame_count = 0

# 游戏主循环控制
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 障碍物生成逻辑
    frame_count += 1
    if frame_count % obstacle_frequency == 0:
        obstacles.append([random.randint(0, screen_width - obstacle_width), -obstacle_height])

    # 更新障碍物位置
    obstacles = [[ob[0], ob[1] + obstacle_speed] for ob in obstacles if ob[1] < screen_height]

    # 这里我们假设有一个障碍物，你需要根据实际游戏逻辑调整这里的代码
    if obstacles:
        input_features = np.array([[player_x, obstacles[0][0]]])
    else:
        input_features = np.array([[player_x, 0]])  # 如果没有障碍物，使用0作为占位符

    # 使用模型预测玩家的行动
    predicted_action = model.predict(input_features)[0]

    # 根据预测结果更新玩家位置
    if predicted_action == 1:  # 假设1代表向右
        player_x = min(player_x + player_speed, screen_width - player_width)
    elif predicted_action == 0:  # 假设0代表向左
        player_x = max(player_x - player_speed, 0)

    # 清空屏幕
    screen.fill((0, 0, 0))

    # 绘制玩家
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(player_x, player_y, player_width, player_height))

    # 绘制障碍物
    for ob in obstacles:
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(ob[0], ob[1], obstacle_width, obstacle_height))

    # 更新屏幕显示
    pygame.display.flip()

    # 控制游戏更新速率
    clock.tick(30)

# 退出pygame
pygame.quit()
