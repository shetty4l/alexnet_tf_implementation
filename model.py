from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import tensorflow as tf
import alexnet_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch""")
tf.app.flags.DEFINE_string('data_dir','data', """Directory of CIFAR-10 images""")

IMAGE_SIZE = alexnet_input.IMAGE_SIZE
NUM_CLASSES = alexnet_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = alexnet_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = alexnet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.999
NUM_EPOCHS_PER_DECAY = 350
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

def _activation_summary(x):
	tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name, shape, stddev, wd):
	var = tf.get_variable(name, shape, initializer = tf.truncated_normal_initializer(stddev=stddev))

	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)

	return var

def distorted_inputs():
	images, labels = alexnet_input.distorted_inputs(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
	return images, labels

def inputs():
	images, labels = alexnet_input.inputs(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
	return images, labels

# def fire_module(input,input_channels,s1,e1,e3,number):
# 	with tf.variable_scope('fire-%d'%number) as scope:
# 		squeeze_kernel = _variable_with_weight_decay('squeeze_weights', [1,1,input_channels,s1],stddev=0.05,wd=0.0)
# 		squeeze = tf.nn.relu(tf.nn.conv2d(input,squeeze_kernel,strides=[1,1,1,1],padding='SAME'), name='squeeze')

# 		expand_kernel_1x1 = _variable_with_weight_decay('expand_1x1_weights', [1,1,s1,e1],stddev=0.05,wd=0.0)
# 		expand_1x1 = tf.nn.relu(tf.nn.conv2d(squeeze, expand_kernel_1x1,strides=[1,1,1,1],padding='SAME'),name='expand_1x1')

# 		# padded_squeeze = tf.pad(squeeze, [[1,1]], "CONSTANT")
# 		expand_kernel_3x3 = _variable_with_weight_decay('expand_3x3_weights', [3,3,s1,e3],stddev=0.05,wd=0.0)
# 		expand_3x3 = tf.nn.relu(tf.nn.conv2d(squeeze, expand_kernel_3x3, strides=[1,1,1,1],padding='SAME'),name='expand_3x3')

# 		return tf.concat(3, [expand_1x1, expand_3x3], name=scope.name)

def inference(images,train=True):
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights', [11,11,3,96], stddev=0.05, wd=0.0)
		conv = tf.nn.conv2d(images, kernel, strides=[1,4,4,1], padding='SAME', name=scope.name)
		biases = tf.get_variable('biases', shape=[96], initializer=tf.constant_initializer(0.1))
		conv = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(conv, name=scope.name)
		_activation_summary(conv1)

	with tf.variable_scope('lrn1') as scope:
		norm1 = tf.nn.local_response_normalization(conv1, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope.name)
		_activation_summary(norm1)

	with tf.variable_scope('pool1') as scope:
		pool1 = tf.nn.max_pool(norm1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=scope.name)
		_activation_summary(pool1)

	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights', [5,5,96,256], stddev=0.05, wd=0.0)
		conv = tf.nn.conv2d(pool1, kernel, strides=[1,1,1,1], padding='SAME', name=scope.name)
		biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.1))
		conv = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(conv, name=scope.name)
		_activation_summary(conv2)

	with tf.variable_scope('lrn2') as scope:
		norm2 = tf.nn.local_response_normalization(conv2, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope.name)
		_activation_summary(norm2)

	with tf.variable_scope('pool2') as scope:
		pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=scope.name)
		_activation_summary(pool2)

	with tf.variable_scope('conv3') as scope:
		kernel = _variable_with_weight_decay('weights', [3,3,256,384], stddev=0.05, wd=0.0)
		conv = tf.nn.conv2d(pool2, kernel, strides=[1,1,1,1], padding='SAME', name=scope.name)
		biases = tf.get_variable('biases', shape=[384], initializer=tf.constant_initializer(0.1))
		conv = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(conv, name=scope.name)
		_activation_summary(conv3)

	with tf.variable_scope('conv4') as scope:
		kernel = _variable_with_weight_decay('weights', [3,3,384,384], stddev=0.05, wd=0.0)
		conv = tf.nn.conv2d(conv3, kernel, strides=[1,1,1,1], padding='SAME', name=scope.name)
		biases = tf.get_variable('biases', shape=[384], initializer=tf.constant_initializer(0.1))
		conv = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(conv, name=scope.name)
		_activation_summary(conv3)

	with tf.variable_scope('conv5') as scope:
		kernel = _variable_with_weight_decay('weights', [3,3,384,256], stddev=0.05, wd=0.0)
		conv = tf.nn.conv2d(conv4, kernel, strides=[1,1,1,1], padding='SAME', name=scope.name)
		biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.1))
		conv = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(conv, name=scope.name)
		_activation_summary(conv5)

	with tf.variable_scope('pool5') as scope:
		pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=scope.name)
		_activation_summary(pool5)

	with tf.variable_scope('local6') as scope:
		reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim,4096], stddev=0.04, wd=0.0005)
		bias = tf.get_variable('biases', shape=[4096], initializer=tf.constant_initializer(0.1))
		local6 = tf.nn.relu(tf.matmul(reshape, weights) + bias, name=scope.name)

		if train==True:
			local6 = tf.nn.dropout(local6, keep_prob=0.5)

		_activation_summary(local6)

	with tf.variable_scope('local7') as scope:
		weights = _variable_with_weight_decay('weights', shape=[4096,4096], stddev=0.04, wd=0.0005)
		bias = tf.get_variable('biases', shape=[4096], initializer=tf.constant_initializer(0.1))
		local7 = tf.nn.relu(tf.matmul(local6, weights) + bias, name=scope.name)

		if train==True:
			local7 = tf.nn.dropout(local6, keep_prob=0.5)

		_activation_summary(local7)

	with tf.variable_scope('local8') as scope:
		weights = _variable_with_weight_decay('weights', shape=[4096,1000], stddev=0.04, wd=0.0005)
		bias = tf.get_variable('biases', shape=[1000], initializer=tf.constant_initializer(0.1))
		local8 = tf.nn.relu(tf.matmul(local7, weights) + bias, name=scope.name)
		_activation_summary(local8)

	return local8

def loss(logits,labels):
	labels = tf.cast(labels, tf.int32)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses',cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	for l in losses+[total_loss]:
		tf.scalar_summary(l.op.name + ' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))

	return loss_averages_op

def train(total_loss, global_step):

	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
	tf.scalar_summary('learning_rate', lr)

	loss_averages_op = _add_loss_summaries(total_loss)

	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	for var in tf.trainable_variables():
		tf.histogram_summary(var.op.name, var)

	for grad, var in grads:
		if grad is not None:
			tf.histogram_summary(var.op.name + '/gradients', grad)

	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op
