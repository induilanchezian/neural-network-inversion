import os
import sys
import argparse

import tensorflow as tf
import numpy as np

import model
from tensorflow.examples.tutorials.mnist import input_data

FLAGS=None

def read_random_mnist_sample():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
	X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
	sample_index = np.random.choice(range(X_train.shape[0]))
	return np.array([X_train[sample_index, :]])

def extract_features(image, hidden1, hidden2, num_classes, checkpoint_dir):
	X = tf.placeholder(tf.float32, image.shape)
	nn = model.NeuralNetwork(hidden1, hidden2, num_classes)
	features1, features2 = nn.feature_extractor(X, name='feature_extractor_MNIST')
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir +'/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		feature_vector1, feature_vector2 = sess.run([features1, features2], feed_dict={X : image})
	return feature_vector1, feature_vector2

if __name__ == '__main__':
	rnd_img = read_random_mnist_sample()
	feature = extract_features(rnd_img, 500, 100, 10, 'models/fully_connected_mnist')
	print feature

