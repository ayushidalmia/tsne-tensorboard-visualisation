__author__ = "Ayushi Dalmia"
__email__ = "ayushidalmia2604@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import argparse
import cv2
from utils import *
import sys
import logging
import shutil

class tsne_visualisation():

	def __init__(self,basedir,embeddings,image_folder,mode="text",metadata="metadata.txt"):
	
		self.embeddings = embeddings
		self.basedir = basedir
		self.metadata = metadata
		self.mode = mode
		self.image_folder = image_folder

		if mode == "image":
			self._create_sprite_images()

	def _create_sprite_images(self):

		if os.path.exists(self.image_folder):
			images = getImages(self.image_folder,os.path.join(self.basedir,"tsne",self.metadata))
			self.images = np.array(images)
			sprite = images_to_sprite(self.images)
			cv2.imwrite(os.path.join(self.basedir,"tsne","sprite.jpg"), sprite)
		else:
			logging.warning('[images] folder not found')

	def visualize_embeddings(self):
		
		with tf.Session() as sess:

			embedding_var = tf.Variable(self.embeddings, name='embedding')
			sess.run(embedding_var.initializer)
			init = tf.global_variables_initializer()
			init.run()

			config = projector.ProjectorConfig()
			config.model_checkpoint_path = os.path.join(self.basedir,"tsne",'my-model.ckpt')
			embedding = config.embeddings.add()
			embedding.tensor_name = embedding_var.name


			embedding.metadata_path = os.path.join(os.path.join(self.basedir,"tsne", self.metadata))

			if self.mode=="image":
				embedding.sprite.image_path = os.path.join(self.basedir,"tsne","sprite.jpg")
				embedding.sprite.single_image_dim.extend([self.images.shape[1], self.images.shape[1]])

			summary_writer = tf.summary.FileWriter(self.basedir)
			projector.visualize_embeddings(summary_writer, config)

			# saves a configuration file that TensorBoard will read during startup.
			saver_embed = tf.train.Saver([embedding_var])
			saver_embed.save(sess, os.path.join(self.basedir,"tsne",'my-model.ckpt'))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Visualise Embeddings")
	parser.add_argument("-m","--mode", help="mode of data (text or image", type=str, dest="mode")
	parser.add_argument("-b", "--base_dir",help="base directory path",type=str, dest="baseDir")
	parser.add_argument("-f", "--embedding_filename",help="filename for embedding",type=str, dest="filename_embeddings")
	parser.add_argument("-l", "--metadata_filename",help="filename for metadata", type=str, dest="filename_label")
	options = parser.parse_args()
	

	embeddings = np.loadtxt(os.path.join(options.baseDir,"embeddings",options.filename_embeddings))

	tsv = tsne_visualisation(options.baseDir,embeddings, os.path.join(options.baseDir,"images"),mode=options.mode, metadata = options.filename_label)
	tsv.visualize_embeddings()