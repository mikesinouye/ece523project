from tensorflow.keras.layers import Flatten, Activation, Input, Lambda, concatenate, Dropout, Conv2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, regularizers

import numpy as np
import tensorflow as tf

from convolution import *


# the model would be defined here, with a Conv2D layer
output = concatenate([c0, c1, c2, c3])
dropout = output
lc = Tea(units=64, name='tea_2')(dropout)
cl = AdditivePooling(8)(lc)
predictions = Activation('softmax')(cl)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, verbose=1, validation_split=0.0)


# this call constrains convolution kernel weights to ternary values: -1, 0, or 1.
# we use a statistical method to round the weights to -1, 0,or 1, incurring some accuracy penalty.

orig_weights, constrained_weights = constrain_weights(model)
model.set_weights(constrained_weights)
		

print("Post-Ternary Constraint Accuracy: ")

# Evaluate the constained weight model
score = model.evaluate(x_test, y_test, verbose=0)

test_predictions = model.predict(x_test)

print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])

# Restore the original weights temporarily so we can evaluate them
model.set_weights(orig_weights)

print("Original Floating-Point Accuracy: ")

# Evaluate the original, floating-point weight model
float_score = model.evaluate(x_test, y_test, verbose=0)

print("Test Loss: ", float_score[0])
print("Test Accuracy: ", float_score[1])

print("Accuracy loss due to train-then-constrain: " ,float_score[1] - score[1])

# Set the weights back to the constrained/rounded versions so we can export them to the simulator
model.set_weights(constrained_weights)