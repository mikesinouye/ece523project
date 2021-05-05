import numpy as np

# General Imports:
import tensorflow as tf
import numpy as np
import copy
import math

# From Imports:
from tensorflow.keras import layers, activations, initializers, regularizers, constraints, Model

@tf.custom_gradient
def custom_round(x):
	output = tf.keras.backend.round(x)
	def grad(dy):
		return dy#*(tf.keras.backend.maximum(0*x, 1-2*tf.keras.backend.abs(x-0.5)) + tf.keras.backend.maximum(0*x, 1-2*tf.keras.backend.abs(x+0.5)))
	return output, grad

class CWTConv2D(layers.Layer) :
	
	def __init__(self, filters, kernel_size, strides, use_bias=True, **kwargs):
		super(CWTConv2D, self).__init__(**kwargs)
		
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.use_bias = use_bias

	def build(self, input_shape):
		
		self.kernel = self.add_weight(name='kernel',
										   shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
										   initializer=initializers.RandomNormal(mean=0.0, stddev=0.5, seed=0),
										   trainable=True)

		if (self.use_bias) :							   
			self.biases = self.add_weight(name='biases',
											  shape=(self.filters),
											  initializer='zeros',
											  trainable=True)
										  
		super(CWTConv2D, self).build(input_shape)
										  
	def call(self, inputs):
		
		#inputs = tf.keras.backend.in_train_phase(inputs, tf.keras.backend.round(inputs))
		
		kernel = self.kernel
		
		#kernel = tf.keras.backend.round(kernel)
		kernel = custom_round(kernel)
		
		kernel = tf.keras.backend.clip(kernel, -1, 1)
		
		output = tf.keras.backend.conv2d(inputs, kernel=kernel, strides=self.strides, padding='valid', data_format="channels_last")
		
		if (self.use_bias) :
			biases = self.biases
			
			#biases = tf.keras.backend.round(biases)
			biases = custom_round(biases)
			
			biases = tf.keras.backend.clip(biases, -128, 127)
			
			output = tf.keras.backend.bias_add(output, biases, data_format='channels_last')
		
		output = tf.keras.backend.in_train_phase(activations.sigmoid(output), tf.dtypes.cast(tf.math.greater_equal(output, 0.0), tf.float32))
		
		#output = tf.keras.backend.clip(output, 0, 1)
		
		#output = tf.keras.backend.round(output)
		#output = custom_round(output)
		
		return output
												 
	def compute_output_shape(self, input_shape):
		
		return ((inputs.shape[0] - self.kernel_size[0])/self.strides[0] + 1, (inputs.shape[1] - self.kernel_size[1])/self.strides[1] + 1, self.filters)
	

# custom cwt constrain function
def constrain_weights_cwt(model) :

	# Slam the convolutional kernel weights to -1, 0, or 1
	weights = model.get_weights()
	orig_weights = copy.deepcopy(weights)
	names = [weight.name for layer in model.layers for weight in layer.weights]
	
	layer_num = -1
	for weight, name in zip(weights, names):
		layer_num += 1

		if 'cwt' in name and 'kernel' in name:
			weight = tf.keras.backend.round(weight)
			weight = tf.keras.backend.clip(weight, -1, 1)
			weights[layer_num] = weight
			
			
		if 'cwt' in name and 'bias' in name:	
			weight = tf.keras.backend.round(weight)
			weight = tf.keras.backend.clip(weight, -128, 127)
			weights[layer_num] = weight
			
	return orig_weights, weights
	
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

		if 'conv2d/kernel' in name:	
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
			
			
		if 'conv2d/bias' in name:	
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
		if 'kernel' in name:	
			print(name + " weights: ")
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
		
	for weight, name in zip(view_weights, names) :
		if 'bias' in name:	
			print(name + " weights : ")
			print(weight)