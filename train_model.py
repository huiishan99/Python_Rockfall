import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# 加载和预处理数据
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    features = []
    labels = []

    for entry in data:
        # 假设状态中包含玩家x位置和障碍物列表
        player_x = entry['state']['player_x']
        obstacles = entry['state']['obstacles']

        # 为简单起见，我们只考虑第一个障碍物的位置
        if obstacles:  # 确保障碍物列表不为空
            first_obstacle_x, first_obstacle_y = obstacles[0]
        else:
            first_obstacle_x, first_obstacle_y = 0, 0  # 如果没有障碍物，使用默认值

        # 构建特征列表
        features.append([player_x, first_obstacle_x, first_obstacle_y])

        # 将动作添加到标签列表
        labels.append(entry['action'])

    return np.array(features), np.array(labels)


# 构建模型
def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 主函数
def main():
    X, y = load_data('game_data.json')

    # 将标签转换为整数
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)
    model = build_model(X_train.shape[1], y_train.shape[1])
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    model.save('model.h5')


if __name__ == "__main__":
    main()
