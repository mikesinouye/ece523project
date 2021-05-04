import numpy as np

# General Imports:
import tensorflow as tf
import numpy as np
import copy
import math
from scipy import signal

# From Imports:
from tensorflow.keras import layers, activations, initializers, regularizers, constraints, Model


class CWTConv2D(layers.Layer) :
	
	def __init__(self, kernel, filters, stride, **kwargs):
		super(CWTConv2D, self).__init__(**kwargs)
		
		self.kernel = kernel
		self.filters = filters
		self.stride = stride

	def build(self, input_shape):
		
		super(CWTConv2D, self).build(input_shape)
											  
		self.kernel = self.add_weight(name='kernel',
										   shape=(filters, kernel[0], kernel[1]),
										   initializer=initializers.TruncatedNormal(mean=0.5))

		self.biases = self.add_weight(name='biases',
										  shape=(filters, kernel[0], kernel[1]),
										  initializer='zeros')
										  
	def call(self, inputs):
	
		with tf.compat.v1.get_default_graph().gradient_override_map({"Round":"CustomRound"}):
		
			inputs = tf.keras.backend.round(inputs)
			kernel = self.kernel
			kernel = tf.keras.backend.round(kernel)
			kernel = tf.keras.backend.clip(kernel, -1, 1)
			
			
			# Multiply Connections with Weights:
			weighted_connections = kernel * self.static_weights
			
			
			# conduct convolution in Python as a reference and to check result later
			for filter in range(self.filters) :
				#result.append(signal.convolve2d(inputs, np.fliplr(np.flipud(kernel[filter])), mode='valid'))
				output = np.append(signal.convolve2d(inputs, kernel[filter], mode='valid'))
			
			biases = tf.keras.backend.round(self.biases)
			
			output = tf.keras.backend.bias_add(output, biases, data_format='channels_last')
			
			output = tf.keras.backend.in_train_phase(self.activation(output), 
                                                     tf.dtypes.cast(tf.math.greater_equal(output, 0.0), tf.float32))
	

# rounds a Keras Conv2D layer's weights to a ternary value (-1, 0, 1) to 
# be compatible with deployment onto a SNN based simulator/enviornment
# finds the average weight, and then uses the mean as a reference point for where it should go
# this should be deprecated once the train while constrain approach is implemented
def constrain_weights(model) :

	# Slam the convolutional kernel weights to -1, 0, or 1
	weights = model.get_weights()
	orig_weights = copy.deepcopy(weights)
	names = [weight.name for layer in model.layers for weight in layer.weights]
	
	layer_num = -1
	for weight, name in zip(weights, names):
		layer_num += 1
		data_type = name.split('/')[1]

		if 'kernel' in data_type:	
			old_shape = weight.shape
			flattened_weights = weight.flatten()
			absolute_weights = np.absolute(flattened_weights)
			mean = np.mean(absolute_weights)
			for i in range(len(flattened_weights)) :
				if (flattened_weights[i] > mean) :
					flattened_weights[i] = 1
				elif (flattened_weights[i] < -1 * mean) :
					flattened_weights[i] = -1
				else :
					flattened_weights[i] = 0
			weight = flattened_weights.reshape(old_shape)
			weights[layer_num] = weight
			
	return orig_weights, weights
	
	
	
def print_weights(model) :
	view_weights = model.get_weights()
	names = [weight.name for layer in model.layers for weight in layer.weights]
	for weight, name in zip(view_weights, names) :
		if name == 'conv2d/kernel:0' :
			print(name + " weights (rounded): ")
			print(weight.shape)
			print_weights = weight
			
	
	for f in range(print_weights.shape[3]) :
		for y in range(print_weights.shape[0]) :
			line = ""
			for x in range(print_weights.shape[1]) :
				for c in range (print_weights.shape[2]) :
					line += str(print_weights[y][x][c][f])
				line += " "
			print(line)
		print("Next Filter: ")