import tensorflow as tf
import numpy as np

class NeuralNetwork:

	def __init__(self, hidden1_units, hidden2_units, num_classes):
		self.hidden1_units = hidden1_units
		self.hidden2_units = hidden2_units
		self.num_classes = num_classes
	
	def hidden_layer(self, inputs, num_neurons, name, activation=None):
		with tf.name_scope(name):
			n_inputs = int(inputs.shape[1])
			stddev = 2 / np.sqrt(n_inputs)
			init = tf.truncated_normal((n_inputs, num_neurons), stddev=stddev)
			W = tf.Variable(init, name="kernel")
			b = tf.Variable(tf.zeros([num_neurons]), name="bias")
			Z = tf.matmul(inputs, W) + b
			if activation is not None:
				return activation(Z)
			else:
				return Z

	def feature_extractor(self, X, name):
		with tf.name_scope(name):
			hidden1 = self.hidden_layer(X, self.hidden1_units, 'hidden1',
						    activation=tf.nn.sigmoid)
			hidden2 = self.hidden_layer(hidden1, self.hidden2_units, 'hidden2',
						    activation=tf.nn.sigmoid)
			return hidden1, hidden2

	def linear_classifier(self, features, name):
		with tf.name_scope(name):
			logits = self.hidden_layer(inputs=features, num_neurons=self.num_classes,
					      name='outputs')
		return logits


class ConvolutionalNeuralNetwork:
	
	def __init__(self, num_classes, train):
		self.num_classes = num_classes
		self.train = True

	def conv_layer(self):
		return None
	
	def hidden_layer(self):
		return None
	
	def feature_extractor(self):
		return None

	def linear_classifier(self):
		return None
