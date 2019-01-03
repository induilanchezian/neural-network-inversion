import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist 

FLAGS = None

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, output_file):
	images = data_set.images
	labels = data_set.labels
	num_examples = data_set.num_examples

	assert images.shape[0] == num_examples
	height = images.shape[1]
	width = images.shape[2]
	channels = images.shape[3]

	filename = os.path.join(FLAGS.directory, output_file+'.tfrecords')
	writer = tf.python_io.TFRecordWriter(filename) 
	
	for index in range(num_examples):
		image_raw = images[index].tostring()
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'height':_int64_feature(height),
					'width':_int64_feature(width),
					'channels':_int64_feature(channels),
					'label':_int64_feature(labels[index]),
					'image':_bytes_feature(image_raw)
				}))
		writer.write(example.SerializeToString())
	
	writer.close()

def main(unused_argv):
	data_sets = mnist.read_data_sets(FLAGS.directory,
					 dtype=tf.uint8,
					 reshape=False,
					 validation_size=FLAGS.validation_size)
	convert_to(data_sets.train, 'train_mnist')
	convert_to(data_sets.test, 'test_mnist')
	convert_to(data_sets.validation, 'validation_mnist')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--directory',
		type=str,
		default='data/mnist/',
		help='Directory to download data and write the converted file'
		)
	parser.add_argument(
		'--validation_size',
		type=int,
		default=5000,
		help='Number of examples in validation set'
		)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



