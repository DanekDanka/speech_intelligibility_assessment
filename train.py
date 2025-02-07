import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from datetime import date
import pickle
import os

import model as Model


def train_model(train_dataset, epochs=10, batch_size=32, learning_rate=0.001):
    model = Model.create_model()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.R2Score()]
    )

    history = model.fit(
        train_dataset.batch(batch_size),
        epochs=epochs,
        verbose=1
    )

    return model, history


def save_model(model: tf.keras.Model):
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('saved_models/' + str(date.today()), exist_ok=True)
    path = 'saved_models/' + str(date.today()) + '/'

    model.save_weights(path + 'weights_speech_intelligibility_assessment' + '.weights.h5')
    model.save(path + 'model_speech_intelligibility_assessment' + '.keras')


def save_history(history):
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('saved_models/' + str(date.today()), exist_ok=True)
    path = 'saved_models/' + str(date.today()) + '/'
    history_filename = path + 'history_speech_intelligibility_assessment' + '.pkl'
    with open(history_filename, 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    train_dataset_path = 'dataset'
    epochs = 1000
    batch_size = 32
    learning_rate = 0.001

    dataset = tf.data.Dataset.load(train_dataset_path)
    train_dataset = dataset

    model, history = train_model(train_dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    save_model(model)
    save_history(history)
