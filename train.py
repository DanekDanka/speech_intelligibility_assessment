import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import model as Model


def train_model(train_dataset, epochs=10, batch_size=32, learning_rate=0.001):
    # Компиляция модели
    model = Model.create_model()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.R2Score]
    )

    # Обучение модели
    history = model.fit(
        train_dataset.batch(batch_size),
        epochs=epochs,
        verbose=1
    )

    return model, history

if __name__ == '__main__':
    train_dataset_path = 'dataset'
    epochs = 500
    batch_size = 32
    learning_rate = 0.001

    dataset = tf.data.Dataset.load(train_dataset_path)
    # dataset = dataset.shuffle(dataset.cardinality())
    train_dataset = dataset
    print(train_dataset)

    for samples, power in train_dataset.take(1):
        print(f'Power: {power.numpy()}')
        print(f'Samples: {samples.numpy()}')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.test.is_built_with_cuda())

    # with tf.device('/GPU:0'):
    train_model(train_dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)