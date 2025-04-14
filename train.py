import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

def load_and_prepare_dataset(dataset_path):
    dataset = tf.data.Dataset.load(dataset_path)
    
    mfcc = []
    stoi = []
    
    for sample in dataset:
        mfcc.append(sample['mfcc'].numpy())
        stoi.append(sample['stoi'].numpy())
    
    X = np.array(mfcc)
    y = np.array(stoi)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=[metrics.MeanAbsoluteError()]
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest R2 Score: {r2:.4f}")
    
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    
    return r2

if __name__ == "__main__":
    dataset_path = "/home/danya/develop/datasets/CMU-MOSEI/Audio/tf_dataset/"
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_dataset(dataset_path)
    
    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Validation shapes: {X_val.shape}, {y_val.shape}")
    print(f"Test shapes: {X_test.shape}, {y_test.shape}")
    
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    model.summary()
    
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=100)
    
    r2_score = evaluate_model(model, X_test, y_test)
    
    model.save("/home/danya/develop/models/stoi_predictor.h5")
    print("Model saved successfully.")
