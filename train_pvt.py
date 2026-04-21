import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the cleaned .npy file
data_path = os.path.join("real_data", "cleaned_master_pvt.npy")

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Run the cleaning script first.")
    exit()

pvt_array = np.load(data_path, allow_pickle=True)
print(f"Loaded PVT array of shape: {pvt_array.shape}")

# Columns after dropping duration: [onset, response_time]
# Pull only the response_time (Index 1) and force it to float
rt_sequence = pvt_array[:, 1].astype(float)

# 1. Feature windowing for Deep Learning
def create_rolling_windows(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        # Extract a window of 10 consecutive reaction times
        window = data[i : (i + window_size)]
        X.append(window)
        
        # Target Label (y): If the subsequent response is a lapse (>250ms), mark as 1
        next_rt = data[i + window_size]
        label = 1 if next_rt > 250 else 0
        y.append(label)
        
    return np.array(X), np.array(y)

X, y = create_rolling_windows(rt_sequence, window_size=10)


unique, counts = np.unique(y, return_counts=True)
print("\n--- DATASET DISTRIBUTION ---")
for val, count in zip(unique, counts):
    label = "Normal (<250ms)" if val == 0 else "Lapse (>250ms)"
    print(f"{label}: {count} samples ({count/len(y)*100:.2f}%)")
print("-----------------------------\n")

# Reshape input to be [samples, timesteps, features] for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the LSTM Deep Learning Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(10, 1), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Binary classifier for fatigue/lapse risk
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nStarting LSTM Training...")
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("pvt_lstm_model.h5")
print("Model saved as pvt_lstm_model.h5")