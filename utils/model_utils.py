import tensorflow as tf
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def resize_image(image_data, size):
	""" Resizes the image without changing the aspect ratio with padding, so that
		the image size is as per model requirement.
		Input:
			image_data: array, original image data
			size: tuple, size the image is to e resized into
		Output:
			image: array, image data after resizing the image
	"""

	image_height, image_width, _ = image_data.shape
	input_height, input_width = size

	# Getting the scale that is to be used for resizing the image
	scale = min(input_width / image_width, input_height / image_height)
	new_width = int(image_width * scale) # new image width
	new_height = int(image_height * scale) # new image height

	# getting the number of pixels to be padded
	dx = (input_width - new_width)
	dy = (input_height - new_height)

	# resizing the image
	image = cv2.resize(image_data, (new_width, new_height), 
		interpolation=cv2.INTER_CUBIC)


	top, bottom = dy//2, dy-(dy//2)
	left, right = dx//2, dx-(dx//2)

	color = [128, 128, 128] # color pallete to be used for padding
	new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) # padding
	
	return new_image


def draw_box(image, bbox, filename):
	""" Draws boxes over the images provided for tensorboard.
		Input:
			image: tfrecord file holding the image information
			bbox: bounding box parameters
	"""

	with tf.name_scope('summary_image'):
		xmin, ymin, xmax, ymax, label = tf.split(value = bbox, num_or_size_splits = 5, axis=2)
		height = tf.cast(tf.shape(image)[1], tf.float32)
		weight = tf.cast(tf.shape(image)[2], tf.float32)
		new_bbox = tf.concat([tf.cast(ymin, tf.float32) / height, tf.cast(xmin, tf.float32) / weight, tf.cast(ymax, tf.float32) / height, tf.cast(xmax, tf.float32) / weight], 2)
		new_image = tf.image.draw_bounding_boxes(image, new_bbox)
		return tf.summary.image('image', tensor=new_image)