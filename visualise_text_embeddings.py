__author__ = "Ayushi Dalmia"
__email__ = "ayushidalmia2604@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import argparse

def main():
	parser = argparse.ArgumentParser(description="Visualise Embeddings")
	parser.add_argument("-b", "--base_dir",help="base directory path",type=str, dest="baseDir")
	parser.add_argument("-f", "--embedding_filename",help="filename for embedding",type=str, dest="filename_embeddings")
	parser.add_argument("-l", "--label_filenam",help="filename for labels",type=str, dest="filename_label")
	options = parser.parse_args()

	if options.filename_embeddings and options.filename_label and options.baseDir:
		embeddings = np.loadtxt(os.path.join(options.baseDir, options.filename_embeddings))
		with tf.Session() as sess:
			embedding_var = tf.Variable(embeddings, name='embedding')
			sess.run(embedding_var.initializer)
			init = tf.global_variables_initializer()
			init.run()

			config = projector.ProjectorConfig()
			summary_writer = tf.summary.FileWriter(options.baseDir)

			embedding = config.embeddings.add()
			embedding.tensor_name = embedding_var.name

			embedding.metadata_path = os.path.join(options.baseDir,options.filename_label)
			projector.visualize_embeddings(summary_writer, config)
			saver_embed = tf.train.Saver([embedding_var])
			saver_embed.save(sess, os.path.join(options.baseDir,'model3.ckpt'), 1)
	else:
		print parser.print_help()


if __name__ == '__main__':
	main()