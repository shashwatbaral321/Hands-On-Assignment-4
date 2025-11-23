# autoencoder_fraud.py
# Author: Shashwat Baral
# Course: MSCS-633 – Advanced AI
# Assignment: Fraud Detection using AutoEncoder with PyOD

import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Dataset loaded.")
print(df.head())

# -----------------------------
# Scaling Data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -----------------------------
# AutoEncoder Model
# -----------------------------
print("Training AutoEncoder model...")

model = AutoEncoder(
    hidden_neurons=[64, 32, 32, 64],
    epochs=30,
    batch_size=32,
    verbose=1
)

model.fit(X_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

print("\n---- CONFUSION MATRIX ----")
print(confusion_matrix(y_test, y_pred))

print("\n---- CLASSIFICATION REPORT ----")
print(classification_report(y_test, y_pred))

print("\nExperiment Completed Successfully.")
