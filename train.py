import argparse
import tensorflow as tf
from tqdm import tqdm
import time
import os
from utils.utils import draw_box
from dataParser import Parser
from utils import checkmate
from model import model
from utils.model_loss import compute_loss
import numpy as np
import config


def get_classes(classes_path):
	""" Loads the classes 
		Input:
			classes_path: path to the file containing class names
		Output: list containing class names
	"""

	with open(classes_path) as f:
	    class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names


def read_anchors(file_path):
	""" Reads the anchors computer by k-means.py for from the provided path
		Input:
			file_path: path to anchors.txt contaning the anchors computer by k-means.py
		Output:
			A numpy array containing the anchors written into anchors.txt
	"""
	anchors = []
	with open(file_path, 'r') as file:
		for line in file.read().splitlines():
			w, h = line.split()
			anchor = [float(w), float(h)]
			anchors.append(anchor)

	return np.asarray(anchors)


def train(ckpt_path, log_path, class_path):
	""" Function to train the model.
		ckpt_path: string, path for saving/restoring the model
		log_path: string, path for saving the training/validation logs
		class_path: string, path for the classes of the dataset
		decay_steps: int, steps after which the learning rate is to be decayed
		decay_rate: float, rate to carrying out exponential decay
	"""


	# Getting the anchors
	anchors = read_anchors(config.anchors_path)
	if not os.path.exists(config.data_dir):
		os.mkdir(config.data_dir)

	classes = get_classes(class_path)

	# Building the training pipeline
	graph = tf.get_default_graph()

	with graph.as_default():

		# Getting the training data
		with tf.name_scope('data_parser/'):
			train_reader = Parser('train', config.data_dir, config.anchors_path, config.output_dir, 
				config.num_classes, input_shape=config.input_shape, max_boxes=config.max_boxes)
			train_data = train_reader.build_dataset(config.train_batch_size//config.subdivisions)
			train_iterator = train_data.make_one_shot_iterator()

			val_reader = Parser('val', config.data_dir, config.anchors_path, config.output_dir, 
				config.num_classes, input_shape=config.input_shape, max_boxes=config.max_boxes)
			val_data = val_reader.build_dataset(config.val_batch_size)
			val_iterator = val_data.make_one_shot_iterator()


			is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag') # Used for different behaviour of batch normalization
			mode = tf.placeholder(dtype=tf.int16, shape=[], name='mode_flag')


			def train():
				return train_iterator.get_next()
			def valid():
				return val_iterator.get_next()


			images, labels = tf.cond(pred=tf.equal(mode, 1), true_fn=train, false_fn=valid, name='train_val_data')
			grid_shapes = [config.input_shape // 32, config.input_shape // 16, config.input_shape // 8]

			images.set_shape([None, config.input_shape, config.input_shape, 3])
			labels.set_shape([None, required_shape, 5])

			# image_summary = draw_box(images, bbox, file_name)

		if not os.path.exists(ckpt_path):
			os.mkdir(ckpt_path)

		model = model(images, is_training, config.num_classes, config.num_anchors_per_scale, config.weight_decay, config.norm_decay)
		output, model_layers = model.forward()

		print('Summary of the created model.......\n')
		for layer in model_layers:
			print(layer)

		# Declaring the parameters for GT
		with tf.name_scope('Targets'):
			### GT PROCESSING ###

		# Compute Loss
		with tf.name_scope('Loss_and_Detect'):
			loss_scale,summaries = compute_loss(output, y_true, config.num_classes, ignore_threshold=config.ignore_thresh)
			exponential_moving_average_op = tf.train.ExponentialMovingAverage(config.weight_decay).apply(var_list=tf.trainable_variables())
			loss = model_loss
			model_loss_summary = tf.summary.scalar('model_loss', summaries, family='Losses')


		# Declaring the parameters for training the model
		with tf.name_scope('train_parameters'):
			global_step = tf.Variable(0, trainable=False, name='global_step')

		# Declaring the parameters for training the model
		with tf.name_scope('train_parameters'):
			global_step = tf.Variable(0, trainable=False, name='global_step')

			def learning_rate_scheduler(learning_rate, scheduler_name, global_step, decay_steps=100):
				if scheduler_name == 'exponential':
					lr =  tf.train.exponential_decay(learning_rate, global_step,
						decay_steps, decay_rate, staircase=True, name='exponential_learning_rate')
					return tf.maximum(lr, config.learning_rate_lower_bound)
				elif scheduler_name == 'polynomial':
					lr =  tf.train.polynomial_decay(learning_rate, global_step,
						decay_steps, config.learning_rate_lower_bound, power=0.8, cycle=True, name='polynomial_learning_rate')
					return tf.maximum(lr, config.learning_rate_lower_bound)
				elif scheduler_name == 'cosine':
					lr = tf.train.cosine_decay(learning_rate, global_step,
						decay_steps, alpha=0.5, name='cosine_learning_rate')
					return tf.maximum(lr, config.learning_rate_lower_bound)
				elif scheduler_name == 'linear':
					return tf.convert_to_tensor(learning_rate, name='linear_learning_rate')
				else:
					raise ValueError('Unsupported learning rate scheduler\n[supported types: exponential, polynomial, linear]')


			if config.use_warm_up:
				learning_rate = tf.cond(pred=tf.less(global_step, config.burn_in_epochs * (config.train_num // config.train_batch_size)),
					true_fn=lambda: learning_rate_scheduler(config.init_learning_rate, config.warm_up_lr_scheduler, global_step),
					false_fn=lambda: learning_rate_scheduler(config.learning_rate, config.lr_scheduler, global_step, decay_steps=2000))
			else:
				learning_rate = learning_rate_scheduler(config.learning_rate, config.lr_scheduler, global_step=global_step, decay_steps=2000)

			tf.summary.scalar('learning rate', learning_rate, family='Train_Parameters')


		# Define optimizer for minimizing the computed loss
		with tf.name_scope('Optimizer'):
			optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config.momentum)
			# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=config.momentum)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				# grads = optimizer.compute_gradients(loss=loss)
				# gradients = [(tf.placeholder(dtype=tf.float32, shape=grad[1].get_shape()), grad[1]) for grad in grads]
				# train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
				optimizing_op = optimizer.minimize(loss=loss, global_step=global_step)
			
			with tf.control_dependencies([optimizing_op]):
				with tf.control_dependencies([exponential_moving_average_op]):
					train_op_with_mve = tf.no_op()
			train_op = train_op_with_mve



#################################### Training loop ############################################################
		# A saver object for saving the model
		best_ckpt_saver_train = checkmate.BestCheckpointSaver(save_dir=ckpt_path+'train/', num_to_keep=5)
		best_ckpt_saver_valid = checkmate.BestCheckpointSaver(save_dir=ckpt_path+'valid/', num_to_keep=5)
		summary_op = tf.summary.merge_all()
		summary_op_valid = tf.summary.merge([model_loss_summary_without_learning_rate])
		init_op = tf.global_variables_initializer()


		
		# Defining some train loop dependencies
		gpu_config = tf.ConfigProto(log_device_placement=False)
		gpu_config.gpu_options.allow_growth = True
		sess = tf.Session(config=gpu_config)
		tf.logging.set_verbosity(tf.logging.ERROR)
		train_summary_writer = tf.summary.FileWriter(os.path.join(log_path, 'train'), sess.graph)
		val_summary_writer = tf.summary.FileWriter(os.path.join(log_path, 'val'), sess.graph)

		print(sess.run(receptive_field))
		
		# Restoring the model
		ckpt = tf.train.get_checkpoint_state(ckpt_path+'train/')
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print('Restoring model ', checkmate.get_best_checkpoint(ckpt_path+'train/'))
			tf.train.Saver().restore(sess, checkmate.get_best_checkpoint(ckpt_path+'train/'))
			print('Model Loaded!')
		else:
			sess.run(init_op)

		print('Uninitialized variables: ', sess.run(tf.report_uninitialized_variables()))


		epochbar = tqdm(range(config.Epoch))
		for epoch in epochbar:
			epochbar.set_description('Epoch %s of %s' % (epoch, config.Epoch))
			mean_loss_train = []
			mean_loss_valid = []

			trainbar = tqdm(range(config.train_num//config.train_batch_size))
			for k in trainbar:

				num_steps, train_summary, loss_train, _ = sess.run([global_step, summary_op, loss,
					train_op], feed_dict={is_training: True, mode: 1})

				train_summary_writer.add_summary(train_summary, epoch)
				train_summary_writer.flush()
				mean_loss_train.append(loss_train)
				trainbar.set_description('Train loss: %s' %str(loss_train))


			print('Validating.....')
			valbar = tqdm(range(config.val_num//config.val_batch_size))
			for k in valbar:
				val_summary, loss_valid = sess.run([summary_op_valid, loss], feed_dict={is_training: False, mode: 0})
				val_summary_writer.add_summary(val_summary, epoch)
				val_summary_writer.flush()
				mean_loss_valid.append(loss_valid)
				valbar.set_description('Validation loss: %s' %str(loss_valid))

			mean_loss_train = np.mean(mean_loss_train)
			mean_loss_valid = np.mean(mean_loss_valid)

			print('\n')
			print('Train loss after %d epochs is: %f' %(epoch+1, mean_loss_train))
			print('Validation loss after %d epochs is: %f' %(epoch+1, mean_loss_valid))
			print('\n\n')

			if (config.use_warm_up):
				if (num_steps > config.burn_in_epochs * (config.train_num // config.train_batch_size)):
					best_ckpt_saver_train.handle(mean_loss_train, sess, global_step)
					best_ckpt_saver_valid.handle(mean_loss_valid, sess, global_step)
				else:
					continue
			else:
				best_ckpt_saver_train.handle(mean_loss_train, sess, global_step)
				best_ckpt_saver_valid.handle(mean_loss_valid, sess, global_step)

		print('Tuning Completed!!')
		train_summary_writer.close()
		val_summary_writer.close()
		sess.close()





def main():
	""" main function which calls all the other required functions for training """
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
	train(config.model_dir, config.logs_dir, config.classes_path)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



if __name__ == '__main__':
	main()
