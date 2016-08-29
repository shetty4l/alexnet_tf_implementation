from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import model as cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tmp/train', """Directory to write event logs and checkpoints""")
tf.app.flags.DEFINE_integer('max_steps', 20000, """Maximum number of iterations""")

def train():
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)

		images, labels = cifar10.distorted_inputs()

		logits = cifar10.inference(images,train=True)

		loss = cifar10.loss(logits, labels)

		train_op = cifar10.train(loss, global_step)

		saver = tf.train.Saver(tf.all_variables())

		summary_op = tf.merge_all_summaries()

		init = tf.initialize_all_variables()

		sess = tf.Session()

		sess.run(init)

		tf.train.start_queue_runners(sess=sess)

		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('Restoring from checkpoint')

		for step in xrange(FLAGS.max_steps):
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])
			duration = time.time() - start_time

			assert not np.isnan(loss_value), 'Model diverged with loss=NaN'

			if step % 100 == 0:
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				print('step %d loss = %f (%.1f examples/sec, %.3f sec/batch)' %
					(step,loss_value,examples_per_sec,sec_per_batch))

				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

			if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
	train()

if __name__=='__main__':
	tf.app.run()

