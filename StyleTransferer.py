import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras import models
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image
from ImageReader import ImageReader
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class StyleTransferer:


	def __init__(
		self,
		style_image_location,
		content_image_location,
		is_notebook_env = False,
		height = 480, width = 640,
		num_iterations = 2500):

		self.style_image_location = style_image_location
		self.content_image_location = content_image_location
		self.is_notebook_env = is_notebook_env
		self.height = height
		self.width = width

		tf.enable_eager_execution()

		self.load_images()
		self.specify_layers()

		(
			self.best,
			self.total_loss_hist,
			self.style_score_hist,
			self.content_score_hist
		) = self.run_style_transfer(
			content_image_location,
			style_image_location,
			num_iterations = num_iterations
		)

		self.show_results()



	def load_images(self):
		self.content = ImageReader.load_img(self.content_image_location, self.height, self.width)
		self.style = ImageReader.load_img(self.style_image_location, self.height, self.width)


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
		style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
		content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
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
		gram_style = self.gram_matrix(base_style)
		return tf.reduce_mean(tf.square(gram_style - gram_target))


	def get_feature_representations(self, model, content_path, style_path):
		content_image = ImageReader.load_and_process_img(content_path, self.height, self.width)
		style_image = ImageReader.load_and_process_img(style_path, self.height, self.width)
		style_outputs = model(style_image)
		content_outputs = model(content_image)
		style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
		content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
		return style_features, content_features


	def compute_loss(self, model, loss_weights, init_image, gram_style_features, content_features):
		style_weight, content_weight = loss_weights
		model_outputs = model(init_image)
		style_output_features = model_outputs[:self.num_style_layers]
		content_output_features = model_outputs[self.num_style_layers:]
		style_score, content_score = 0, 0
		weight_per_style_layer = 1.0 / float(self.num_style_layers)
		for target_style, comb_style in zip(gram_style_features, style_output_features):
			style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)
		weight_per_content_layer = 1.0 / float(self.num_content_layers)
		for target_content, comb_content in zip(content_features, content_output_features):
			content_score += weight_per_content_layer * self.get_content_loss(comb_content[0], target_content)
		style_score *= style_weight
		content_score *= content_weight
		loss = style_score + content_score 
		return loss, style_score, content_score


	def compute_grads(self, cfg):
		with tf.GradientTape() as tape:
			all_loss = self.compute_loss(**cfg)
		total_loss = all_loss[0]
		return tape.gradient(total_loss, cfg['init_image']), all_loss


	def run_style_transfer(self, content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2): 

		model = self.get_model() 

		style_features, content_features = self.get_feature_representations(model, content_path, style_path)
		gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]

		init_image = ImageReader.load_and_process_img(content_path, self.height, self.width)
		init_image = tfe.Variable(init_image, dtype = tf.float32)

		opt = tf.train.AdamOptimizer(learning_rate = 5, beta1 = 0.99, epsilon = 1e-1)
		loss_weights = (style_weight, content_weight)

		cfg = {
			'model': model,
			'loss_weights': loss_weights,
			'init_image': init_image,
			'gram_style_features': gram_style_features,
			'content_features': content_features
		}

		display_interval = 100

		norm_means = np.array([103.939, 116.779, 123.68])
		min_vals = - norm_means
		max_vals = 255 - norm_means   

		best_loss, best_img = float('inf'), None
		total_loss_hist, style_score_hist, content_score_hist = [], [], []


		print('\n\nStarted Transferring Style\n')
		print('-' * 70)

		iteration_range = tqdm_notebook(range(1, num_iterations + 1)) if self.is_notebook_env else tqdm(range(1, num_iterations + 1))
		for i in iteration_range:
			grads, all_loss = self.compute_grads(cfg)
			loss, style_score, content_score = all_loss
			opt.apply_gradients([(grads, init_image)])
			clipped = tf.clip_by_value(init_image, min_vals, max_vals)
			init_image.assign(clipped)
			if loss < best_loss:
				best_loss = loss
				best_img = ImageReader.deprocess_img(init_image.numpy())
			if i % display_interval == 0:
				total_loss_hist.append(loss)
				style_score_hist.append(style_score)
				content_score_hist.append(content_score)
				print('Iteration: {}'.format(i))        
				print('Total loss: {:.4e}, style loss: {:.4e}, content loss: {:.4e}'.format(loss, style_score, content_score))

		print('-' * 70)
		print('\nStyle Transferred\n')
		total_loss_hist = np.array(total_loss_hist)
		style_score_hist = np.array(style_score_hist)
		content_score_hist = np.array(content_score_hist)
		return best_img, total_loss_hist, style_score_hist, content_score_hist
	

	def show_results(self):
		content = ImageReader.load_img(self.content_image_location, self.height, self.width) 
		style = ImageReader.load_img(self.style_image_location, self.height, self.width)
		plt.figure(figsize = (12, 5))
		plt.subplot(1, 2, 1)
		ImageReader.imshow(content, 'Content Image')
		plt.subplot(1, 2, 2)
		ImageReader.imshow(style, 'Style Image')
		plt.figure(figsize = (12, 10))
		ImageReader.imshow(self.best, 'Output Image')
		plt.show()