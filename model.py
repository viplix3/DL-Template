import warnings
warnings.filterwarnings("ignore")
import configparser
import tensorflow as tf
import io
import os
from collections import defaultdict
import numpy as np
import config

class model:

	def __init__(self, image_tensor, is_training, num_classes, weight_decay=0.0005, norm_decay=0.9):
		""" Initializes the object for the model.
			Input:
				image_tensor: tf tensor, which is to be passed onto the model
				is_training: boolean, for controlling the different batch_norm behaviour during training/testing phase.
				num_classes: int, number of classes in the model
				weight_decay: float, weight decay for carrying out regularization and prevent overt-fitting
				norm_decay: float, momentum for decaying the running mean and variance of batch_norm
		"""
		self.image_tensor = image_tensor
		self.is_training = is_training
		self.num_classes = num_classes
		self.norm_decay = norm_decay
		self.initializer = tf.glorot_uniform_initializer() # xavier initializer for initializing convolution filter weights
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay) # l2 regularizer for avoiding overfitting


	def module_1(self, previous_feature_map, filter_size, num_splits):
		""" Defines a block in the tensorflow graph.
			Input:
				previous_feature_map: tf tensor, the output feature map of the previous layer
				filter_size: int, filter size to be used in the convolution kernel
				num_splits: int, number of splits to be made to the provided feature map
			Output:
				module_1_out : tf tensor, the output feature map after applying module_1
		"""
		return modue_1_out


	def module_2(self, previous_feature_map, filter_size, num_splits):
		""" Defines a block in the tensorflow graph.
			Input:
				previous_feature_map: tf tensor, the output feature map of the previous layer
				filter_size: int, filter size to be used in the convolution kernel
				num_splits: int, number of splits to be made to the provided feature map
			Output:
				module_1_out : tf tensor, the output feature map after applying module_1
		"""
		return modue_2_out


	def module_4(self, previous_feature_map, filter_size, num_splits):
		""" Defines a block in the tensorflow graph.
			Input:
				previous_feature_map: tf tensor, the output feature map of the previous layer
				filter_size: int, filter size to be used in the convolution kernel
				num_splits: int, number of splits to be made to the provided feature map
			Output:
				module_1_out : tf tensor, the output feature map after applying module_1
		"""
		return modue_4_out


	def module_3(self, previous_feature_map, filter_size, num_splits):
		""" Defines a block in the tensorflow graph.
			Input:
				previous_feature_map: tf tensor, the output feature map of the previous layer
				filter_size: int, filter size to be used in the convolution kernel
				num_splits: int, number of splits to be made to the provided feature map
			Output:
				module_1_out : tf tensor, the output feature map after applying module_1
		"""
		return modue_3_out



	def forward(self):
		""" Executes a forward pass of the model on the provided input.
			Input:
				N/A
			Output:
				output: list of tensors, list of output nodes produced by the model.
		"""

		all_layers = [] # for model summary
		with tf.name_scope('model_name/'):
			model = self.image_tensor
			all_layers.append(model)

			build_further_model



		return [output_nodes], all_layers 
