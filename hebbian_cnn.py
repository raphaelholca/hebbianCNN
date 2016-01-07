""" 
Author: Raphael Holca-Lamarre
Date: 26/05/2015

Convolutional Hebbian neural network object.
"""

import numpy as np
import external as ex
import matplotlib.pyplot as plt
import progressbar
import pickle
import os

ex = reload(ex)

class Network:
	""" Hebbian convolutional neural network with reward-based learning """
	
	def __init__(self, dopa_conv, dopa_feedf, dopa_class, name='net', n_epi_crit=10, n_epi_dopa=10, A=900., lr=0.01, t=0.01, batch_size=196, conv_map_num=5, conv_filter_side=5, feedf_neuron_num=49, explore='feedf'):
		""" 
		Sets network parameters 

			Args:
				dopa_conv (dict): values of dopamine release in the convolutional layer
				dopa_feedf (dict): values of dopamine release in the feedforward layer
				dopa_class (dict): values of dopamine release in the classification layer
				name (str, optional): name of the network, used to save network to disk. Default: 'net'
				n_epi_crit (int, optional): number of statistical pre-training steps (pure Hebbian). Default: 10
				n_epi_dopa (int, optional): number of dopamine-mediated training steps. Default: 10
				A (float, optional): parameter for the normalization of the input images (pixel values sum to A). Default: 900
				lr (float, optional): learning rate of the network. Default: 0.01
				t (float, optional): temperature of the softmax function ('softness' of the winner-take-all). Default: 0.01
				batch_size (int, optional): size of training batch. Default: 196
				conv_map_num (int, optional): number of convolutional filter maps. Default: 5
				conv_filter_side (int, optional): size of each convolutional filter (side of filter in pixel; total number of pixel in filter is conv_filter_side^2). Default: 5
				feedf_neuron_num (int, optional): number of neurons in the feedforward layer. Default: 49
				explore (str, optional): determines in which layer to perform exploration by noise addition. Valid value: 'none', 'conv', 'feedf'. Default: 'feedf'
		"""
		self.dopa_conv 			= dopa_conv
		self.dopa_feedf 		= dopa_feedf
		self.dopa_class 		= dopa_class
		self.name 				= name
		self.n_epi_crit 		= n_epi_crit
		self.n_epi_dopa 		= n_epi_dopa
		self.n_epi_tot 			= n_epi_crit + n_epi_dopa
		self.A 					= A
		self.lr 				= lr
		self.t 					= t
		self.batch_size 		= batch_size
		self.conv_map_num 		= conv_map_num
		self.conv_filter_side 	= conv_filter_side
		self.feedf_neuron_num 	= feedf_neuron_num
		self.explore 			= explore

	def init_weights(self, images_side, n_classes, init_file=""):
		""" 
		Initializes weights of the network, either random or by loading weights from init_file 

			Args:
				images_side (int): side of the input images in pixels (total pixel number in image if images_side^2).
				n_classes (int): number of classes in the dataset. Used to set the number of neurons in the classificaion layer.
				init_file (str, optional): path to Network object to load saved weights from. Leave empty for random weigth initialization. Default: ''
		"""
		self.images_side 		= images_side
		self.class_neuron_num 	= n_classes
		self.init_file			= init_file
		self.conv_neuron_num 	= (images_side - self.conv_filter_side + 1)**2
		self.conv_map_side 		= int(np.sqrt(self.conv_neuron_num))
		self.subs_map_side 		= self.conv_map_side/2

		self.conv_W, self.feedf_W, self.class_W = ex.init_weights(self, init_file)

	def train(self, images, labels):
		""" 
		Train Hebbian convolutional neural network

			Args: 
				images (3D numpy array): images to train the Network on. images matrix must be 3D: [images_num, images_side, images_side] 
				labels (1D numpy array): labels of the training images.

			returns:
				(float): training performance of the network.
		"""

		print "\ntraining network..."
		classes = np.sort(np.unique(labels))
		n_images = images.shape[0]
		correct = 0.

		for e in range(self.n_epi_tot):
			print "\ntrain episope: %d/%d" % (e+1, self.n_epi_tot)
			
			rnd_images, rnd_labels = ex.shuffle_images(images, labels)
			last_neuron_class = np.zeros((self.feedf_neuron_num, self.class_neuron_num))
			dopa_save = np.array([])
			correct = 0.

			pbar_epi = progressbar.ProgressBar()
			for i in pbar_epi(range(rnd_images.shape[0])):
				explore_epi=np.copy(self.explore) if e>=self.n_epi_crit else 'none'
				
				#propagate image through the network
				classif, conv_input, conv_activ, subs_activ, feedf_activ, class_activ, class_activ_noise = ex.propagate(self, rnd_images[i,:,:], explore=explore_epi)

				#compute reward prediction, reward delivery and dopamine release
				reward_pred = ex.reward_prediction(np.argmax(class_activ), np.argmax(class_activ_noise))
				reward = ex.reward_delivery(rnd_labels[i], classes[np.argmax(class_activ_noise)])
				dopa_release = ex.dopa_release(reward_pred, reward)
					
				# learn weights...
				#...of the convolutional maps
				bs = self.batch_size
				for b in range(self.conv_neuron_num/bs):
					dopa_release_conv = ex.dopa_value(dopa_release, self.dopa_conv)*np.ones(bs) if explore_epi=='conv' else None
					self.conv_W = ex.learning_step(self, conv_input[b*bs:(b+1)*bs, :], conv_activ[b*bs:(b+1)*bs, :], self.conv_W, dopa=dopa_release_conv)

				#...of the feedforward layer
				dopa_release_feedf = ex.dopa_value(dopa_release, self.dopa_feedf) if explore_epi=='feedf' else None
				self.feedf_W = ex.learning_step(self, subs_activ, feedf_activ, self.feedf_W, dopa=dopa_release_feedf)

				#...of the classification layer	
				dopa_release_class = ex.dopa_value(dopa_release, self.dopa_class)
				self.class_W = ex.learning_step(self, feedf_activ, class_activ, self.class_W, dopa=dopa_release_class)

				dopa_save = np.append(dopa_save, dopa_release)
				correct += float(classes[np.argmax(class_activ)] == rnd_labels[i])
				last_neuron_class[np.argmax(feedf_activ), np.argwhere(rnd_labels[i]==classes)] += 1

			correct_class_W = np.sum(np.argmax(last_neuron_class,1)==np.argmax(self.class_W,1))
			print "train error: %.2F%%" % ((1. - correct/rnd_images.shape[0]) * 100)
			print "correct W_out assignment: %d/%d" % (correct_class_W, self.feedf_neuron_num)

		return (1. - correct/n_images)

	def test(self, images, labels):
		""" 
		Test Hebbian convolutional neural network

			Args: 
				images (3D numpy array): images to test the Network on. images matrix must be 3D: [images_num, images_side, images_side] 
				labels (1D numpy array): labels of the testing images.

			returns:
				(float): test performance of the network.
		"""

		print "\ntesting network..."
		classes = np.sort(np.unique(labels))
		n_images = images.shape[0]

		correct = 0.
		pbar_epi = progressbar.ProgressBar()
		for i in pbar_epi(range(images.shape[0])):
			classif = ex.propagate(self, images[i,:,:])[0]
			if classes[classif] == labels[i]: correct += 1.
		print "test error: %.2F%%" % ((1. - correct/images.shape[0]) * 100)

		return (1. - correct/n_images)

























