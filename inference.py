import warnings
warnings.filterwarnings("ignore")
from model import model
from utils import checkmate
import tensorflow as tf
import config
import os
import cv2
import numpy as np
import argparse
from utils.utils import *
import matplotlib.pyplot as plt
from time import time
from PIL import Image, ImageFont, ImageDraw
import colorsys
import random

from tensorflow.python.tools import inspect_checkpoint as chkp


# Some command line arguments for running the model
parser = argparse.ArgumentParser(description="Run inference using darknet converted model")
parser.add_argument('img_path', help="Path for running inference on a single image or \
	multiple images")
parser.add_argument("output_path", help="Output Path to save the results")



def read_image(img_path):
	""" A function which reads image(s) from the path provided
		Input:
			img_path: Path containing images
		Output:
			A batch containing all the images read using opencv
	"""
	assert img_path != None, 'Image path required for making inference'
	if os.path.exists(img_path):
		if os.path.isdir(img_path):
			img_dir = sorted(os.listdir(img_path))
			print('Reading {} images'.format(len(img_dir)))
			image = []
			for i in img_dir:
				img = cv2.imread(os.path.join(img_path, i))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				image.append(img)
			print('Read {} images'.format(len(img_dir)))

		else:
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return image
	else:
		print("Path does not exists!!")



def get_classes(labels_path):
	""" Loads the classes 
		Input:
			labels_path: path in which classes.txt is kept
		Output: list containing class names
	"""


	with open(labels_path) as f:
		class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names


def predict(output_nodes, num_classes, input_shape, image_shape):
	""" Predicts the output of an image
		Input:
			output_nodes: output_nodes of the graph
			num_classes: int, number of classes for making predictions
			input_shape: tuple, input image size to the model
			image_shape: tuple, original image shape
	"""

	#### DO ALL THE MODEL INFERENCING AND RETURN OUTPUT NODES, VALUES ####
	return output_nodes


def run_inference(img_path, output_dir,  args):
	""" A function making inference using the pre-trained darknet weights in the tensorflow 
		framework 
		Input:
			img_path: string, path to the image on which inference is to be run, path to the image directory containing images in the case of multiple images.
			output_dir: string, directory for saving the output
			args: argparse object
	"""

	# Reading the images
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	if not os.path.exists(os.path.join(output_dir, 'images')):
		os.mkdir(os.path.join(output_dir, 'images'))
	if not os.path.exists(os.path.join(output_dir, 'labels')):
		os.mkdir(os.path.join(output_dir, 'labels'))

	output_dir_images = os.path.join(output_dir, 'images')
	output_dir_labels = os.path.join(output_dir, 'labels')


	file_names = sorted(os.listdir(img_path))
	images_batch = read_image(img_path)


	# Getting anchors and labels for the prediction
	class_names = get_classes(config.classes_path)

	num_classes = config.num_classes
	num_anchors = config.num_anchors

	# Retriving the input shape of the model i.e. (608x608), (416x416), (320x320)
	input_shape = (config.input_shape, config.input_shape)

	# Defining placeholder for passing the image data onto the model
	image_tensor = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
	image_shape = tf.placeholder(dtype=tf.int32, shape=[2])


	model = model(image_tensor, is_training=False, num_classes=config.num_classes)
	output_nodes, model_layers = model.forward()

	print('Summary of the model created.......\n')
	for layer in model_layers:
		print(layer)


	# Creating a session for running the model
	gpu_config = tf.ConfigProto(log_device_placement=False)
	gpu_config.gpu_options.allow_growth = True
	sess = tf.Session(config=gpu_config)


	output_values = predict(output_nodes, num_classes, 
		input_shape, image_shape)

	ckpt_path = config.model_dir+'valid/'
	exponential_moving_average_obj = tf.train.ExponentialMovingAverage(config.weight_decay)
	saver = tf.train.Saver(exponential_moving_average_obj.variables_to_restore())
	ckpt = tf.train.get_checkpoint_state(ckpt_path)
	# chkp.print_tensors_in_checkpoint_file(checkmate.get_best_checkpoint(ckpt_path), tensor_name='', all_tensors=True)
	# exit()
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Restoring model ', checkmate.get_best_checkpoint(ckpt_path))
		saver.restore(sess, checkmate.get_best_checkpoint(ckpt_path))
		print('Model Loaded!')

	total_time_pred = []
	for x in range(len(images_batch)):
	
		image = images_batch[x]
		new_image_size = (config.input_shape, config.input_shape)
		image_data = np.array(resize_image(image, new_image_size))
		print('Image height: {}\tImage width: {}'.format(image.shape[0], image.shape[1]))


		img = image_data/255.
		img = np.expand_dims(img, 0) # Adding the batch dimension


		tick = time()
		# Actually run the graph in a tensorflow session to get the outputs
		out_values = sess.run([output_values], feed_dict={image_tensor: img, image_shape: [image.shape[0], image.shape[1]]})
		tock = time()
		total_time_pred.append(tock-tick)


		print('Found {} boxes for {} in {}sec'.format(len(out_boxes), 'img', tock-tick))

	 	######################## Visualization ######################
		font = ImageFont.truetype(font='./font/FiraMono-Medium.otf', 
			size=np.floor(1e-2 * image.shape[1] + 0.5).astype(np.int32))
		thickness = (image.shape[0] + image.shape[1]) // 500  # do day cua BB

		image = Image.fromarray((image).astype('uint8'), mode='RGB')
		output_labels = open(os.path.join(output_dir_labels, file_names[x].split('.')[0]+'.txt'), 'w')
		### DO ALL THE PLOTTING THING IF REQUIRED ###
		### SAVE THE IMAGE ###

		output_labels.close() # Saving labels

	sess.close()

	total_time_pred = sum(total_time_pred[1:])
	print('FPS of model with post processing over {} images is {}'.format(len(images_batch)-1, (len(images_batch)-1)/total_time_pred))


def main(args):
	""" A function fetching the image data from the provided patha nd calling function 
		run_inference for doing the inference
		Input:
			args : argument parser object containing the required command line arguments
	"""
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
	run_inference(args.img_path, args.output_path, args)


if __name__ == '__main__':
	main(parser.parse_args())
