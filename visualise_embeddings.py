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

def visualize_embeddings(basedir,embeddings,mode="text",metadata="metadata.txt"):
	
	if mode == "image":
		if os.path.exists(os.path.join(basedir,"images")):
			images = getImages(os.path.join(basedir,"images","data"),os.path.join(basedir,"images",metadata))
			images = np.array(images)
			sprite = images_to_sprite(images)
			cv2.imwrite(os.path.join(basedir,"sprite.jpg"), sprite)
		else:
			logging.warning('[images] folder not found')
			 

	with tf.Session() as sess:

		embedding_var = tf.Variable(embeddings, name='embedding')
		sess.run(embedding_var.initializer)
		init = tf.global_variables_initializer()
		init.run()
		saver_embed = tf.train.Saver([embedding_var])
		saver_embed.save(sess, './my-model.ckpt')
		config = projector.ProjectorConfig()
		
		embedding = config.embeddings.add()
		embedding.tensor_name = embedding_var.name


		if mode=="text":
			embedding.metadata_path = os.path.join(os.path.join(basedir,"text",metadata))

		else:
			embedding.sprite.image_path = os.path.join(basedir,"sprite.jpg")
			embedding.sprite.single_image_dim.extend([images.shape[1], images.shape[1]])

		summary_writer = tf.summary.FileWriter(basedir)
		projector.visualize_embeddings(summary_writer, config)

		


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Visualise Embeddings")
	parser.add_argument("-m","--mode", help="mode of data (text or image", type=str, dest="mode")
	parser.add_argument("-b", "--base_dir",help="base directory path",type=str, dest="baseDir")
	parser.add_argument("-f", "--embedding_filename",help="filename for embedding",type=str, dest="filename_embeddings")
	parser.add_argument("-l", "--metadata_filename",help="filename for metadata", type=str, dest="filename_label")
	options = parser.parse_args()
	

	embeddings = np.loadtxt(os.path.join(options.baseDir,"embeddings",options.filename_embeddings))


	visualize_embeddings(options.baseDir, embeddings, mode=options.mode)