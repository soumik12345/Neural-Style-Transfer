import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import preprocess_input

class ImageReader:

	@staticmethod
	def load_img(location, height, width):
	    img = cv2.resize(cv2.cvtColor(cv2.imread(location), cv2.COLOR_BGR2RGB), (width, height))
	    img = np.expand_dims(img, axis = 0).astype('uint8')
	    return img

	@staticmethod
	def load_and_process_img(location, height, width):
	    img = ImageReader.load_img(location, height, width)
	    img = preprocess_input(img)
	    return img

	@staticmethod
	def deprocess_img(processed_img):
	    x = processed_img.copy()
	    if len(x.shape) == 4:
	        x = np.squeeze(x, 0)
	    assert len(x.shape) == 3
	    if len(x.shape) != 3:
	        raise ValueError("Invalid input to deprocessing image")
	    x[:, :, 0] += 103.939
	    x[:, :, 1] += 116.779
	    x[:, :, 2] += 123.68
	    x = x[:, :, ::-1]
	    x = np.clip(x, 0, 255).astype('uint8')
	    return x

	@staticmethod
	def imshow(img, title = ""):
	    if img.ndim == 4:
	        img = np.squeeze(img, axis = 0).astype(np.uint8)
	    plt.imshow(img)
	    if title != "":
	        plt.title(title)
	    plt.xticks([])
	    plt.yticks([])