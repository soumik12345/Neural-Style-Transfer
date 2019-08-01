import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras import models
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image
from ImageHelper import ImageHelper


class StyleTransferer:


	def __init__(self, style_image_location, content_image_location):

		tf.enable_eager_execution()

		self.load_images()
		self.specify_layers()


	def load_images(self):
		self.content = ImageHelper.load_img(content_image_location)
		self.style = ImageHelper.load_img(style_image_location)


	def specify_layers(self):
		self.content_layers = ['block5_conv2']
		self.style_layers = [
			'block1_conv1',
			'block2_conv1',
			'block3_conv1',
			'block4_conv1',
			'block5_conv1'
		]
		self.num_content_layers = len(self.content_layers)
		self.num_style_layers = len(self.style_layers)


	def get_model(self):
		vgg = VGG19(include_top = False, weights = 'imagenet')
		vgg.trainable = False
		style_outputs = [vgg.get_layer(name).output for name in style_layers]
		content_outputs = [vgg.get_layer(name).output for name in content_layers]
		model_outputs = style_outputs + content_outputs
		return models.Model(vgg.input, model_outputs)


	def get_content_loss(self, base_content, target):
		return tf.reduce_mean(tf.square(base_content - target))


	def gram_matrix(self, input_tensor):
		channels = int(input_tensor.shape[-1])
		a = tf.reshape(input_tensor, [-1, channels])
		n = tf.shape(a)[0]
		gram = tf.matmul(a, a, transpose_a = True)
		return gram / tf.cast(n, tf.float32)


	def get_style_loss(self, base_style, gram_target):
		gram_style = gram_matrix(base_style)
		return tf.reduce_mean(tf.square(gram_style - gram_target))


	def get_feature_representations(self, model, content_path, style_path):
		content_image = load_and_process_img(content_path)
		style_image = load_and_process_img(style_path)
		style_outputs = model(style_image)
		content_outputs = model(content_image)
		style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
		content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
		return style_features, content_features


	def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
		style_weight, content_weight = loss_weights
		model_outputs = model(init_image)
		style_output_features = model_outputs[:num_style_layers]
		content_output_features = model_outputs[num_style_layers:]
		style_score, content_score = 0, 0
		weight_per_style_layer = 1.0 / float(num_style_layers)
		for target_style, comb_style in zip(gram_style_features, style_output_features):
			style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
		weight_per_content_layer = 1.0 / float(num_content_layers)
		for target_content, comb_content in zip(content_features, content_output_features):
			content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
		style_score *= style_weight
		content_score *= content_weight
		loss = style_score + content_score 
		return loss, style_score, content_score


	def compute_grads(cfg):
		with tf.GradientTape() as tape:
			all_loss = compute_loss(**cfg)
		total_loss = all_loss[0]
		return tape.gradient(total_loss, cfg['init_image']), all_loss