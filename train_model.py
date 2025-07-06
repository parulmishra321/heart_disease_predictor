import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load and clean the data
data = pd.read_csv('heart.csv')
data.columns = data.columns.str.strip()

# One-hot encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Split into features and target
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Save the model
with open('model/heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model/heart_model.pkl")
