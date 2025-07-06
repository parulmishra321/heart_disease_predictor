import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

df = pd.read_csv("heart.csv")
print(df.drop('HeartDisease', axis=1).columns.tolist())
data = pd.read_csv('heart.csv')
data.columns = data.columns.str.strip()

data = pd.get_dummies(data, drop_first=True)

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)

with open('model/heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to model/heart_model.pkl")
