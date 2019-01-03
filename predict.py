import os
import sys
import argparse

import numpy as np
import tensorflow as tf

from scipy import misc
from matplotlib import pyplot as plt

import model

FLAGS=None

def prediction(image, checkpoint_dir, hidden1, hidden2, num_classes):
	image = tf.cast(image, tf.float32)
	nn = model.NeuralNetwork(hidden1, hidden2, num_classes)
	_, features = nn.feature_extractor(image, name='feature_extractor_MNIST')
	logits = nn.linear_classifier(features,name='classifier_MNIST')
	preds = tf.nn.softmax(logits)
	pred_class = tf.argmax(preds, 1)

	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.checkpoint_dir +'/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		predicted_class, feats = sess.run([pred_class, features])
	print feats
	return predicted_class

def main(unused_argv):
	image = misc.imread(FLAGS.image_file)
	image = image.reshape((784,))
	image = np.array([image])
	image = tf.cast(image, tf.float32)
	image = image/(255.0) 
	pred = prediction(image, FLAGS.checkpoint_dir, 500, 100, 10)
	print pred


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_dir',
			    type=str,
			    default='models/fully_connected_mnist',
			    help='Checkpoint directory')
	parser.add_argument('--image_file',
			    type=str,
			    default='inverted5.png',
			    help='Image file path')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
