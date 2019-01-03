import os 
import sys
import argparse

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

FLAGS = None

def restore_all_vars(reader):
	names_to_shape_map = reader.get_variable_to_shape_map()
	names_to_values_map = {}
	for key in sorted(names_to_shape_map):
		names_to_values_map[key] = reader.get_tensor(key)
	return names_to_values_map
		
def restore_var(reader, var_name):
	var_value = reader.get_tensor(var_name)
	return var_value

def main(unused_argv):
	latest_ckp = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
	reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
	var_values = []
	for var in FLAGS.vars:
		var_values.append(restore_var(reader, var))
	for var, val in zip(FLAGS.vars, var_values):
		print var
		print val

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_dir',
			    type=str,
			    default='models/fully_connected_mnist/',
			    help='Checkpoints storage directory'
			    )
	parser.add_argument('--vars', 
			    type=list,
			    default=['feature_extractor_MNIST/hidden1/kernel','feature_extractor_MNIST/hidden2/kernel'],
			    help='Variable to be restored'
			    )
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

	


	
