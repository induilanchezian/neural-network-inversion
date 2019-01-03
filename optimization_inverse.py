import os
import sys
import argparse

import numpy as np 
import tensorflow as tf
import scipy.optimize as opt

from tensorflow.python import pywrap_tensorflow
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from scipy import misc

import restore_weights
import features

def make_minimize_cb(path=[], image=[]):
	def minimize_cb(xk):
		# note that we make a deep copy of xk
		path.append(L2_difference_objective(np.copy(xk[500:]), image))
	return minimize_cb

def sigmoid_inverse(x):
	return np.log(x/(1-x))

def sigmoid(x):
	return (1/(1+np.exp(-1*x)))

def L2_difference_objective(x, y):
	shp = x.shape
	assert shp == y.shape
	assert len(shp) == 1 
	return np.square(np.linalg.norm(x-y))

def read_random_mnist_sample(label):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
	X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
	y_train = mnist.train.labels
	label_indices = np.where(y_train==label)
	label_indices = label_indices[0]
	sample_index = np.random.choice(label_indices)
	return np.array([X_train[sample_index, :]])
	
def invert_features(false_class_image, real_class_image, features1, features2, weights, num_units, max_iters, eps=1e-6):	
	#####Get indices where original image is sero to get the boundaries of the input image. This works for MNIST...
	##### General method for other image datasets
	#zero_indices_real = np.where(real_class_image == 0.0)
	#zero_indices_real = zero_indices_real[0]
	#zero_indices_real = np.ones(zero_indices_real.shape[0])*500 + zero_indices_real 
	#zero_indices_real = zero_indices_real.astype(int)
	
	#z = sigmoid_inverse(features2).clip(eps, 1-eps)
	
	bounds = opt.Bounds(np.ones(shape=(num_units,)) * eps, np.ones(shape=(num_units,))* (1-eps))
	weights1 = weights['feature_extractor_MNIST/hidden1/kernel']
	bias1 = weights['feature_extractor_MNIST/hidden1/bias']
	weights2 = weights['feature_extractor_MNIST/hidden2/kernel']
	bias2 = weights['feature_extractor_MNIST/hidden2/bias']
	#eq_cons = {'type' : 'eq',
	#	   'fun'  : lambda x: np.concatenate((np.dot(np.transpose(weights2), x[:500]) + bias2 - z, np.dot(np.transpose(weights1), x[500:]) + bias1 - sigmoid_inverse(x[:500])						     ,x[zero_indices] - np.ones(zero_indices.shape[0])*eps))
	#	  }
	eq_cons = {'type' : 'eq',
		   'fun'  : lambda x: np.concatenate((sigmoid(np.dot(np.transpose(weights2), x[:500]) + bias2) - features2, sigmoid(np.dot(np.transpose(weights1), x[500:]) + bias1) - x[:500]))
		  }
	#ineq_cons = {'type' : 'ineq',
	#	     'fun'  : lambda x: np.concatenate((0.1 - features1 + x[:500], 0.1 - image + x[500:]))}
	#x0 = np.append((np.random.randn(num_units-784)).clip(eps, 1-eps), image_vector.clip(eps,1-eps))
	#x0 = np.zeros(num_units) 
	x0 = np.append(features1, false_class_image)
	path_ = []
	res = opt.minimize(lambda x: L2_difference_objective(x[500:], real_class_image), x0, method='SLSQP', 
	               constraints=[eq_cons], options={'ftol':1e-3, 'maxiter': max_iters, 'disp': True},
		       bounds=bounds, callback=make_minimize_cb(path_, real_class_image))
	print path_ 	
	return res.x
	

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_dir',
			     type=str,
			     help='Checkpoint directory of tensorflow model')
	parser.add_argument('--original_image_file',
			     type=str,
			     help='File path for original image')
	parser.add_argument('--inverted_image_file',
			     type=str,
			     help='File path to store the result of inversion')
	parser.add_argument('--real_class',
			     type=int,
			     help='Class label of original image')
	parser.add_argument('--false_class',
			     type=int,
			     help='Class label of the false class image')
	parser.add_argument('--max_iters',
			     type=int,
			     default=10,
			     help='Maximum number of iterations to run the non-linear optimization problem')
	#checkpoint_dir = 'models/fully_connected_mnist'
	args = parser.parse_args()
	
	latest_ckp = tf.train.latest_checkpoint(args.checkpoint_dir)
	reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
	
	false_class_image = read_random_mnist_sample(args.false_class)
	false_class_image = false_class_image[0]
	real_class_image = read_random_mnist_sample(args.real_class)
	real_class_image = real_class_image[0]
	
	misc.imsave(args.original_image_file, false_class_image.reshape((28,28)))
	#######Fixed parameters must be modifiable for different datasets and models....use a metadata file for model...
	#######Store a meta data file with format while training
	feature_array1, feature_array2 = features.extract_features(np.array([false_class_image]), 500, 100, 10, args.checkpoint_dir)
	feature_vec1 = feature_array1[0]
	feature_vec2 = feature_array2[0]
	weights = restore_weights.restore_all_vars(reader)
	num_units = 28*28+500

	inverted_ip = invert_features(false_class_image, real_class_image, feature_vec1, feature_vec2, weights, num_units, args.max_iters)
	inverted_img = inverted_ip[500:]
	inverted_img = inverted_img.reshape((28,28))
	misc.imsave(args.inverted_image_file, inverted_img)

	fig = plt.figure()
	fig.add_subplot(1,2,1)
        plt.axis('off')
	plt.imshow(false_class_image.reshape(28,28))
	fig.add_subplot(1,2,2)
        plt.axis('off')
	plt.imshow(inverted_img)
	plt.show()

	

	


