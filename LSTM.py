#!/usr/bin/env python
# coding: utf-8


# All includes
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 2.1.x
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Load input data
def load_X(x_signals_paths):
    X_signals = []

    for signal_type_path in x_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))


# Load output data
def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

DATA_PATH = "data/"
os.chdir(DATA_PATH)
os.chdir("..")
TRAIN = "train/"
TEST = "test/"

# Create data paths
X_train_signals_paths = [DATA_PATH + "UCI HAR Dataset/" + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
X_test_signals_paths = [DATA_PATH + "UCI HAR Dataset/" + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]
y_train_path = DATA_PATH + "UCI HAR Dataset/" + TRAIN + "y_train.txt"
y_test_path = DATA_PATH + "UCI HAR Dataset/" + TEST + "y_test.txt"

# Prepare input and output data
X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)
y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_features = 9
n_input = len(X_train[0][0])

# LSTM structure
n_hidden = 32  # Hidden layer num of features
n_classes = 6  # Total classes (should go up, or should go down)
n_epochs = 20

# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iterations = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 128
display_iterations = 30000

# TODO: what is batch_size ???
# TODO: find how to save the model
# Create the model
# from https://adventuresinmachinelearning.com/keras-lstm-tutorial/ example
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(50, activation='sigmoid', return_sequences=True))
model.add(tf.keras.layers.LSTM(50, activation='sigmoid', return_sequences=False))
# to get one output and not 50
model.add(tf.keras.layers.Dense(1))

# TODO: checkout what the compile method is doing
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
# TODO: convert train data correct

history1 = model.fit(X_train, y_train, epochs=2)

# Plot loss
# TODO: show accuracy as well
var = history1.history['loss']
plt.plot(var)
plt.show()


# Prediction - Use the model
test = X_test[1666]
test = test.reshape((1, n_steps, n_features))
prediction = model.predict(test)
print(prediction+1)
