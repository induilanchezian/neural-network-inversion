import os
import random 
import sys

import numpy as np
import tensorflow as tf
from six.moves import xrange

tf.flags.DEFINE_string('data_directory','data/shapes/', 'Data Directory')
tf.flags.DEFINE_string('output_directory', 'data/TFRecords/', 'Output Data Dierctory')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label, height, width):
	colorspace = 'RGB'
	channels = 3
	image_format = 'png'

	example = tf.train.Example(features=tf.train.Features(feature={
		  'image/height': _int64_feature(height),
		  'image/width': _int64_feature(width),
		  'image/colorspace': _bytes_feature(colorspace),
		  'image/channels': _int64_feature(channels),
		  'image/label': _int64_feature(label),
		  'image/format': _bytes_feature(image_format),
		  'image/filename': _bytes_feature(filename),
		  'image/encoded': _bytes_feature(image_buffer)})) 
	
	return example

class ImageCoder(object):
	
	def __init__(self):
		self._sess = tf.Session()

		self._png_data = tf.placeholder(dtype=tf.string)
		image = tf.image.decode_png(self._png_data, channels=3)
		self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb',quality=100)
			

		self._cmyk_data = tf.placeholder(dtype=tf.string)
		image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
		self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb',quality=100)

		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

	def png_to_jpeg(self, image_data):
		return self._sess.run(self._png_to_jpeg,
			              feed_dict={self._png_data: image_data})

	def cmyk_to_rgb(self, image_data):
		return self._sess.run(self._cmyk_to_rgb, 
				      feed_dict={self._cmyk_data: image_data})

	def decode_jpeg(self, image_data):
		image = self._sess.run(self._decode_jpeg,
				       feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image


def _process_image(filename, coder):
	
	image_data = tf.gfile.FastGFile(filename, 'r').read()

	image_data = coder.png_to_jpeg(image_data)
	image = coder.decode_jpeg(image_data)

	assert len(image.shape) == 3
	height = image.shape[0]
	width = image.shape[1]
	assert image.shape[2] == 3

	return image_data, height, width

def _process_image_files(name, filenames, labels):
	output_filename = 'shapes'
	output_file = os.path.join(FLAGS.data_directory, output_filename+'.tfrecords')
	writer = tf.python_io.TFRecordWriter(output_file)
	coder = ImageCoder()

	assert len(filenames) == len(labels)

	for i,filename in enumerate(filenames):
		image_buffer, height, width = _process_image(filename, coder)
		example = _convert_to_example(filename, image_buffer, labels[i], height, width)
		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()

def _find_image_files(data_dir):
	labels = []
	filenames = []

	label_ind = 0
	classes = os.listdir(data_dir)
	for cls in classes:
		filepath = '%s/%s/*.png' % (data_dir, cls)
		matching_files = tf.gfile.Glob(filepath)
		
		labels.extend([label_ind] * len(matching_files))
		filenames.extend(matching_files)

	shuffled_index = range(len(filenames))
	random.seed(12345)
	random.shuffle(shuffled_index)

	filenames = [filenames[i] for i in shuffled_index]
	labels = [labels[i] for i in shuffled_index]

	return filenames, labels

def _process_dataset(name, directory):
	filenames, labels = _find_image_files(directory)
	_process_image_files(name, filenames, labels)


def main(unused_argv):
	_process_dataset('data', FLAGS.data_directory)
	
if __name__=='__main__':
	tf.app.run()

	



	




