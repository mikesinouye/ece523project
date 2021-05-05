from tensorflow.keras.layers import Flatten, Activation, Input, Lambda, concatenate, Dropout, Conv2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, regularizers

from skimage.filters import threshold_otsu

import numpy as np
import tensorflow as tf
from convolution import *
from additivepooling import *

(X_train, y_train), (X_test, y_test) = mnist.load_data()

for i in np.arange(len(X_train)):
  thresh = threshold_otsu(X_train[i])
  X_train[i] = X_train[i] > thresh

for i in np.arange(len(X_test)):
  thresh = threshold_otsu(X_test[i])
  X_test[i] = X_test[i] > thresh
			
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

inputs = Input(shape=(28,28,1))
			
'''
conv = Conv2D(filters=20,
			  kernel_size=(11,11),
			  strides=(1,1),
			  activation='relu',
			  kernel_regularizer=regularizers.l1(l=0.1),
			  use_bias=True,
			  )(inputs)
'''

conv = CWTConv2D(filters=20,
				  kernel_size=(11,11),
				  strides=(1,1),
				  #activation='relu',
				  #kernel_regularizer=regularizers.l1(l=0.1),
				  #use_bias=True,
				  )(inputs)

flattened_inputs = Flatten()(conv)

network = AdditivePooling(10)(flattened_inputs)

predictions = Activation('softmax')(network)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model.fit(X_train, y_train, batch_size=128, epochs=75, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=0)



# this call constrains convolution kernel weights to ternary values: -1, 0, or 1.
# we use a statistical method to round the weights to -1, 0,or 1, incurring some accuracy penalty.

orig_weights, constrained_weights = constrain_weights_cwt(model)
model.set_weights(constrained_weights)
		

print("Post-Ternary Constraint Accuracy: ")

# Evaluate the constained weight model
score = model.evaluate(X_test, y_test, verbose=0)

test_predictions = model.predict(X_test)

print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])

# Restore the original weights temporarily so we can evaluate them
model.set_weights(orig_weights)

print("Original Floating-Point Accuracy: ")

# Evaluate the original, floating-point weight model
float_score = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss: ", float_score[0])
print("Test Accuracy: ", float_score[1])

print("Accuracy loss due to train-then-constrain: " ,float_score[1] - score[1])


#view_weights = model.get_weights()
#names = [weight.name for layer in model.layers for weight in layer.weights]
#print(view_weights)
#print(names)

print_weights(model)

model.set_weights(constrained_weights)

print_weights(model)