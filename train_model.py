# train_model.py
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    features = []
    labels = []
    for entry in data:
        features.append([entry['state']['player_x'], entry['state']['obstacles'][0][0]])
        labels.append(1 if entry['action'] == 'right' else 0)
    return np.array(features), np.array(labels)

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def main():
    X, y = load_data('game_data.json')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_model(X_train, y_train)
    joblib.dump(model, 'game_model.pkl')

if __name__ == "__main__":
    main()
