import os
import sys
import time

import tensorflow as tf
import model
import argparse

FLAGS = None

def parse_record(serialized_example):
	features = tf.parse_single_example(serialized_example,
					   features={
						     'image': tf.FixedLenFeature([], tf.string),
						     'label': tf.FixedLenFeature([], tf.int64),
					   })
	image = tf.decode_raw(features['image'], tf.uint8)
	image.set_shape((28*28))
	image = tf.cast(image, tf.float32)
	image = image/255.0
	label = tf.cast(features['label'], tf.int32)

	return image, label 

def read_and_parse_TFRecords(record_path, train):
	dataset = tf.data.TFRecordDataset(record_path)
	dataset = dataset.map(parse_record)

	if train:
		dataset = dataset.shuffle(10000)
		dataset = dataset.repeat(FLAGS.num_epochs)
	dataset = dataset.batch(FLAGS.batch_size)
	
	return dataset

def data_init():
	train_dataset = read_and_parse_TFRecords(FLAGS.train_file, True)
	validation_dataset = read_and_parse_TFRecords(FLAGS.validation_file, False)
	test_dataset = read_and_parse_TFRecords(FLAGS.test_file, False)
	
	iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
	image, label = iterator.get_next()

	train_init = iterator.make_initializer(train_dataset)
	validation_init = iterator.make_initializer(validation_dataset)
	test_init = iterator.make_initializer(test_dataset)

	return image, label, train_init, validation_init, test_init

def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
								       name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	return cross_entropy_mean

def accuracy(logits, labels):
	preds = tf.nn.softmax(logits)
	labels = tf.cast(labels, tf.int64)
	pred_classes = tf.argmax(preds,1)
	correct_preds = tf.equal(pred_classes, labels)
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
	return accuracy

def optimize(loss, learning_rate, gstep):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss, global_step=gstep)
	return train_op

def train():
	gstep = tf.Variable(0, name='global_step', trainable=False)

	image, label, train_init, validation_init, test_init = data_init()
	nn = model.NeuralNetwork(FLAGS.hidden1, FLAGS.hidden2, FLAGS.num_classes)
	features = nn.feature_extractor(image, name='feature_extractor_MNIST')
	logits = nn.linear_classifier(features,name='classifier_MNIST')
	batch_loss = loss(logits, label)
	train_op = optimize(batch_loss, FLAGS.learning_rate, gstep)
	acc_batch = accuracy(logits, label)

	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(train_init)
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.checkpoint_dir +'/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		try:
			step=gstep.eval()
			while True:
				start_time = time.time()
				_, loss_value = sess.run([train_op, batch_loss])

				duration = time.time() - start_time

				if step%FLAGS.log_freq == 0:
					print ('Step %d: loss = %.2f (%.3f sec)'%(step, loss_value, duration))
				
				step+=1
		except tf.errors.OutOfRangeError:
			print ('Done training for %d epochs, %d steps' % (FLAGS.num_epochs, step))
			saver.save(sess, FLAGS.checkpoint_dir+'/fc_mnist', step)

		sess.run(test_init)
		total_acc = 0 
		n_examples = 0 
		start_time = time.time()
		try:
			while True:
				accuracy_value = sess.run(acc_batch)
				n_examples += FLAGS.batch_size
				total_acc += accuracy_value
		except tf.errors.OutOfRangeError:
			print('Test set accuracy: %.2f (%.3f sec)' %(total_acc/n_examples*100, time.time()-start_time))


def main(unused_argv):
	train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', 
			    type=float,
			    default=0.01,
			    help='Initial learning rate')
	parser.add_argument('--num_epochs',
			     type=int,
			     default=2,
			     help='Number of epochs to run trainer')
	parser.add_argument('--hidden1',
			     type=int,
			     default=128,
			     help='Number of units in hidden layer 1')
	parser.add_argument('--hidden2',
			     type=int,
			     default=32,
			     help='Number of units in hidden layer 2')
	parser.add_argument('--batch_size',
			     type=int,
			     default=64,
			     help='Batch size')
	parser.add_argument('--num_classes',
			     type=int,
			     default=10,
			     help='Number of classes')
	parser.add_argument('--train_file',
			     type=str,
			     default='data/mnist/train_mnist.tfrecords',
			     help='Path to training data tfrecord file')
	parser.add_argument('--test_file',
			     type=str,
			     default='data/mnist/test_mnist.tfrecords',
			     help='Path to test data tfrecord file')
	parser.add_argument('--validation_file',
			     type=str,
			     default='data/mnist/validation_mnist.tfrecords',
			     help='Path to validation data tfrecord file')
	parser.add_argument('--log_freq',
			     type=int,
			     default=100,
			     help='Frequency for logging loss and time metrics')
	parser.add_argument('--checkpoint_dir',
			     type=str,
			     default='models/fully_connected_mnist',
			     help='Checkpoint directory')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
