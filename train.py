import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv("data/raw/Titanic-Dataset.csv")

# ── Feature Engineering ────────────────────────────────────
# Drop columns that are not useful for prediction
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode text columns into numbers (ML models need numbers)
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])        # male=1, female=0
df["Embarked"] = le.fit_transform(df["Embarked"])  # C=0, Q=1, S=2

# ── Split Features and Target ──────────────────────────────
X = df.drop(columns=["Survived"])
y = df["Survived"]

# ── Train / Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train Model ────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f" Model trained successfully!")
print(f" Test Accuracy: {accuracy:.2%}")

# ── Save Model ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print(" Model saved to models/model.joblib")