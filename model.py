import tensorflow as tf
from numpy.ma.core import shape
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model():
    model = Sequential()

    model.add(Input(shape=(22050,)))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(1, activation='exponential'))

    return model
