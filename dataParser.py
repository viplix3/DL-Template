# Importing some necessary libraries to run the program
import tensorflow as tf
import numpy as np
import os
import sys
import threading
import random
import config
import time
from datetime import datetime


# Defining some flags
tf.app.flags.DEFINE_integer('train_threads', 5, 
	'Number of threads to be used for processing training images')
tf.app.flags.DEFINE_integer('val_threads', 2, 
	'Number of threads to be used for processing validation images')
tf.app.flags.DEFINE_integer('train_shards', 10, 
	'Number of shards for training data')
tf.app.flags.DEFINE_integer('val_shards', 2, 
	'Number of shards for validation data')


FLAGS = tf.app.flags.FLAGS



class Parser:

	def __init__(self, mode, dataset_dir, anchors_path, output_dir, num_classes, 
		input_shape):
		""" Initializes the object of the parser class.
			Input:
				mode: string, sets the mode to 'train' or 'val'
				dataset_dir: string, path for the directory where the dataset has been stored
				anchors_path: string, path for the anchors
				output_dir: string, path for the directory where the tfrecords will be saved
				num_classes: int, number of classes in the dataset
				input_shape: int, shape of the input to the model
		"""

		self.data_dir = dataset_dir
		self.input_shape = input_shape
		self.mode = mode
		self.annotations_file = {'train': config.train_annotations_file, 'val': 
		config.val_annotations_file}
		# self.dataset_dir = {'train': config.train_data_file, 'val': config.val_data_file}
		self.anchors_path = anchors_path
		self.anchors = self.read_anchors()
		self.num_classes = num_classes
		self.output_dir = output_dir
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
		file_pattern = self.output_dir + self.mode + '*.tfrecord'
		self.TfrecordFile = tf.gfile.Glob(file_pattern)
		self.class_names = self.get_classes(config.classes_path)
		if len(self.TfrecordFile) == 0:
			self.make_tfrecord()


	def _int64_feature(self, value):
		""" Converts the given input into an int64 feature that can be used in tfrecords
			Input:
				value: value to be converte into int64 feature
			Output:
				tf.train.Int64List object encoding the int64 value that can be used in tfrecords
		"""
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



	def _bytes_feature(self, value):
		""" Converts the given input into a bytes feature that can be used in tfrecords
			Input:
				value: value to be converted into bytes feature
			Output:
				tf.train.BytesList object that can be used in tfrecords
		"""
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


	def _float_feature(self, value):
		""" Converts the given input into an float feature that can be used in tfrecords
			Input:
				value: value to be converted into float feature
			Output:
				tf.train.FloatList object that can be used in tfrecords
		"""

		return tf.train.Feature(float_list=tf.train.FloatList(value=value))



	def read_anchors(self):
		""" Reads the anchors computer by k-means.py for from the provided path
			Output:
				A numpy array containing the anchors written into anchors.txt
		"""
		anchors = []
		with open(self.anchors_path, 'r') as file:
			for line in file.read().splitlines():
				w, h = line.split()
				anchor = [float(w), float(h)]
				anchors.append(anchor)

		return np.asarray(anchors)



	def get_classes(self, classes_path):
	    """ Loads the classes 
	    	Input:
	    		classes_path: path to the file containing class names
	    	Output: list containing class names
	    """
	    
	    with open(classes_path) as f:
	        class_names = f.readlines()
	    class_names = [c.strip() for c in class_names]
	    return class_names



	def read_annotations(self, file_path):
		""" Reads the image_path and annotations from train.txt
			Input:
				file_path: path to file contatining annotations
			Output:
				file_name: array, containing relative path of dataset files
				other_information: required gt information for the image
		"""
		classes = self.class_names
		file_name = []
		class_id = []
		with open(file_path) as file:
			for lines in file.read().splitlines():
				line = lines.split()
				name = line[0]
				file_name.append(name)
				line = line[1::]

				# Parse other information from train.txt, val.txt if required

		return np.array(file_name), np.array(other_information)



	def process_tfrecord_batch(self, mode, thread_index, ranges, file_names, classes):
		""" Processes images and saves tfrecords 
			Input:
				mode: string, specify if the tfrecords are to be made for training, validation 
					or testing
				thread_index: specifies the thread which is executing the function
				ranges: list, specifies the range of images the thread calling this function 
					will process
				file_names: array, containing the relative filepaths of images
				classes: array, containing class_id associated to every image
		"""

		if mode == 'train':
			num_threads = FLAGS.train_threads
			num_shards = FLAGS.train_shards

		if mode == 'val' or mode == 'test':
			num_threads = FLAGS.val_threads
			num_shards = FLAGS.val_shards

		num_anchors = np.shape(self.anchors)[0]

		num_shards_per_batch = int(num_shards/num_threads)
		shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], 
			num_shards_per_batch+1).astype(int)
		num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

		counter = 0
		for s in range(num_shards_per_batch):
			shard = thread_index * num_shards_per_batch + s
			output_filename = '%s-%.5d-of-%.5d.tfrecord' % (mode, shard, num_shards)
			output_file = os.path.join(self.output_dir, output_filename)
			writer = tf.python_io.TFRecordWriter(output_file)

			shard_count = 0
			files_in_shard = np.arange(shard_ranges[s], shard_ranges[s+1], dtype=int)
			
			for i in files_in_shard:

				_filename = file_names[i]
				_classes = classes[i]

				image_data = self._process_image(_filename)

				example = self.convert_to_example(_filename, image_data, _classes)
				
				writer.write(example.SerializeToString())
				shard_count += 1
				counter += 1

			
			writer.close()
			print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, 
				shard_count, output_file))
			shard_count = 0
		print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, 
			counter, num_files_in_thread))



	def _process_image(self, filename):
		""" Read image files from disk 
			Input:
				file_name: str, relative path of the image
			Output:
				img_data: array, containing the image data
		"""
		with tf.gfile.GFile(filename, 'rb') as file:
			image_data = file.read()

		return image_data



	def preprocess_gt(self, thing_to_be_preproessed):
		### Do all the preprocessing required on the GT data ####
		return preprocessed_gt



	def convert_to_example(self, file_name, image_data, classes):
		""" Converts the values to Tensorflow TFRecord example for saving in the TFRecord file 
			Input:
				image_data: array, containing the image data read from the disk
				classes: array, containing the classes 
			Output:
				returns a Tensorflow tfrecord example
		"""
		classes = classes.T
		example = tf.train.Example(features=tf.train.Features(feature={
			'image/file_name': self._bytes_feature(tf.compat.as_bytes(file_name)),
			'image/encoded': self._bytes_feature(image_data),
			'image/object/labels':, self._bytes_feature(classes)
			}))
		return example



	def process_tfrecord(self, mode, file_names, classes):
		""" Makes required threds and calls further functions to execute the process of 
			making tfrecords in a multithreaded environment 
			Input:
				mode: string, specify if the tfrecords are to be made for training, validation 
					or testing
				file_names: array, containing the relative filepaths of images
		"""

		# Checking if the passed arguments are correct

		if mode == 'train':
			num_threads = FLAGS.train_threads
			num_shards = FLAGS.train_shards

		if mode == 'val' or mode == 'test':
			num_threads = FLAGS.val_threads
			num_shards = FLAGS.val_shards

		num_anchors = np.shape(self.anchors)[0]

		# Getting the number of images (spacing) to be used by each thread
		spacing = np.linspace(0, len(file_names), num_threads+1).astype(np.int)
		ranges = []
		for i in range(len(spacing)-1):
			ranges.append([spacing[i], spacing[i+1]])

		print("Launching %d threads for spacings: %s" % (num_threads, ranges))

		# For coordinating all the threads
		coord = tf.train.Coordinator()

		threads = []

		# Staring all the threads for making tfrecords
		for thread_idx in range(len(ranges)):
			args = (mode, thread_idx, ranges, file_names, classes)
			t = threading.Thread(target=self.process_tfrecord_batch, args=args)
			t.start()
			threads.append(t)


		# Wait for all threads to finish
		coord.join(threads)
		print("%s: Finished writing all %d images in dataset" %(datetime.now(), len(file_names)))



	def make_tfrecord(self):
		""" Does some assertions and calls other functions to create tfrecords """

		# Checking if flags and shards are in correct ratio
		assert not FLAGS.train_shards % FLAGS.train_threads, ('Please \
			make the FLAGS.num_threads commensurate with FLAGS.train_shards')
		assert not FLAGS.val_shards % FLAGS.val_threads, ('Please make \
			the FLAGS.num_threads commensurate with ''FLAGS.valtest_shards')


		num_anchors = self.anchors.shape[0]
		print('Number of anchors in {}: {}'.format(self.anchors_path, num_anchors))
		

		print('Reading {}'.format(self.annotations_file[self.mode]))
		file_path, other_information = self.read_annotations(self.annotations_file[self.mode])

		num_images = np.shape(file_path)[0]
		print('Number of images in dataset: %d' % (num_images))

		print('Preparing data....')
		self.process_tfrecord(self.mode, file_path, classes)
		

	def parser(self, serialized_example):
		""" Parsed the bianary serialized example
			Input:
				serialized_example, tensorflow tfrecords serialized example
			Output:
				image: tf tensor, conatines the image data
		"""
		features = tf.parse_single_example(
			serialized_example,
			features = {
				'image/file_name': tf.VarLenFeature(dtype=tf.string),
				'image/encoded' : tf.FixedLenFeature([], dtype=tf.string),
				'image/object/labels' : tf.VarLenFeature(dtype=tf.float32),
			}
		)
		file_name = features['image/file_name'].values
		# file_name = tf.Print(file_name, [file_name], message="file_name: ")
		image = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
		image = tf.image.convert_image_dtype(image, tf.uint8)
		labels = tf.expand_dims(features['image/object/labels'].values, axis=0)
		image, labels = self.Preprocess(image, labels)
		
		preprocessed_labels = tf.py_func(self.preprocess_gt, [labels], [tf.float32])
		return image, labels


	def Preprocess(self, image, labels):
		""" Resizes the image to required width and height without changing the aspect ratio,
			required prep-processing is done as well.
			Input:
				image: image for doing the pre-processing
				labels: labels for the image
			Output:
				returns the image after doing pre-processing
		"""

		image_width, image_high = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
		input_width = tf.cast(self.input_shape, tf.float32)
		input_high = tf.cast(self.input_shape, tf.float32)

		# Getting the new image width and height for resizing image by preserving the aspect ratio
		new_high = image_high * tf.minimum(input_width / image_width, input_high / image_high)
		new_width = image_width * tf.minimum(input_width / image_width, input_high / image_high)

		# Pixels to be added on the height and width respectively
		dx = (input_width - new_width) / 2
		dy = (input_high - new_high) / 2

		# Resizing the image
		image = tf.image.resize_images(image, [tf.cast(new_high, tf.int32), tf.cast(new_width, tf.int32)], method=tf.image.ResizeMethod.BICUBIC)
		# Padding done
		new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
		
		# Making the background for tapsting the image onto so that model gets the required image size
		image_ones = tf.ones_like(image)
		image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
		
		# Making space for adding the image pixels onto the background
		image_color_padded = (1 - image_ones_padded) * 128
		image = image_color_padded + new_image
		if self.mode == 'train':
			def _do_labels_preprocessing_if_required_during_flip(labels):
				return flipped_labels
			flipped_labels = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
			labels = tf.cond(flip_left_right, lambda: _do_labels_preprocessing_if_required_during_flip(labels), lambda: labels)

			random_saturation = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_saturation, lambda: tf.image.random_saturation(image=image, lower=0.4, upper=config.sat), lambda: image)

			random_hue = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_hue, lambda: tf.image.random_hue(image=image, max_delta=config.hue), lambda: image)

			random_contrast = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_contrast, lambda: tf.image.random_contrast(image=image, lower=0.4, upper=config.cont), lambda: image)

			random_brit = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.3)
			image = tf.cond(random_brit, lambda: tf.image.random_brightness(image=image, max_delta=config.bri), lambda: image)

		image = image / 255.
		image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
		return image, labels


	def build_dataset(self, batch_size):
		""" Builds the dataset according to the provided mode.
			Input:
				batch_size: int, batch_size to be fed into the model.
			Output:
				dataset: tf.data.Dataset object
		"""

		with tf.name_scope('data_parser/'):
			dataset = tf.data.TFRecordDataset(filenames=self.TfrecordFile)
			dataset = dataset.map(self.parser, num_parallel_calls=config.num_parallel_calls)
			if self.mode == 'train':
				dataset = dataset.repeat().shuffle(500).batch(batch_size).prefetch(batch_size)
			else:
				dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
			return dataset
