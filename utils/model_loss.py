import tensorflow as tf
from utils import whatever_is_necessary
import config


def compute_loss(output, y_true, num_classes, print_loss=False):
	""" Computes the custom written loss for provided output.
		Input:
			output: array, output of model for provided input image
			y_true: array, y_true label corresponding to the output produced from GT
			num_classes: int, number of classes in the dataset
			print_loss, python bool, weather to print loss or not
		Output:
			loss: computed loss
			all_the_related summaries: summaries when and if required
	"""

	if print_loss:
		loss = tf.Print(loss, [loss, respective_sub_losses_if_required], message='loss: ')

	return loss, all_the_related_summaries