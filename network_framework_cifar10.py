from tensorflow.keras.layers import Flatten, Activation, Input, Lambda, concatenate, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import Model, regularizers

from skimage.filters import threshold_otsu


import numpy as np
import tensorflow as tf

from convolution import *
from additivepooling import *
from tea import *



### CHANGE THESE ####
cwt = True
dump_weights = False
### CHANGE THESE ####



if (cwt) :
	epochs = 100
else :
	epochs = 100

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = opt_thresh_color_three(X_train)
X_test = opt_thresh_color_three(X_test)
	
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

inputs = Input(shape=(32,32,3))
			

if (cwt) :
	conv = CWTConv2D(filters=15,
						kernel_size=(11,11),
						strides=(1,1),
						use_bias=True,
						)(inputs)
						
	pool = CWTMaxPooling2D(pool_size=(2,2), strides=(2,2))(conv)
					  
	conv_output = Dropout(0.5)(pool)

else :
	conv = Conv2D(filters=15,
					kernel_size=(11,11),
					strides=(1,1),
					activation='relu',
					use_bias=True,
					)(inputs)
					
					
	pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv)
	
	conv_output = Dropout(0.5)(pool)


flattened_conv = Flatten()(conv_output)

lc = Tea(units=120, name='dense_tea')(flattened_conv)

network = AdditivePooling(10)(lc)

predictions = Activation('softmax')(network)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()
plot_model(model, to_file='cifar10_model.png', show_shapes=True, show_layer_names=True)

X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

model.fit(X_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=0)



# this call constrains convolution kernel weights to ternary values: -1, 0, or 1.
# we use a statistical method to round the weights to -1, 0,or 1, incurring some accuracy penalty.

if (cwt) :
	orig_weights, constrained_weights = constrain_weights_cwt(model)
else :
	orig_weights, constrained_weights = constrain_weights(model)


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

print("Accuracy loss due to train-then-constrain: " , float_score[1] - score[1])


if (dump_weights) :
	names = [weight.name for layer in model.layers for weight in layer.weights]
	print(names)
	
	model.set_weights(orig_weights)
	print_weights(model)

	model.set_weights(constrained_weights)
	print_weights(model)
	