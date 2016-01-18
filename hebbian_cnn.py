""" 
Author: Raphael Holca-Lamarre
Date: 26/05/2015

Convolutional Hebbian neural network object.
"""

import numpy as np
import helper as hp
import matplotlib.pyplot as plt
import progressbar
import pickle
import os

hp = reload(hp)

class Network:
	""" Hebbian convolutional neural network with reward-based learning """
	
	def __init__(self, dopa_conv, dopa_feedf, dopa_class, name='net', n_epi_crit=10, n_epi_dopa=10, A=900., lr=0.01, t=0.01, batch_size=196, conv_map_num=5, conv_filter_side=5, feedf_neuron_num=49, explore='feedf', init_file=None):
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
				init_file (str, optional): initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization. Default: None
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
		self.init_file 			= init_file
		self.perf_train 		= np.zeros(self.n_epi_tot)
		self.perf_test 			= 0.

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
		self.classes = np.sort(np.unique(labels))
		self.images_side = np.size(images, 2)
		self._init_weights()
		n_images = images.shape[0]
		correct = 0.

		for e in range(self.n_epi_tot):
			print "\ntrain episope: %d/%d" % (e+1, self.n_epi_tot)
			
			rnd_images, rnd_labels = hp.shuffle_images(images, labels)
			last_neuron_class = np.zeros((self.feedf_neuron_num, self.class_neuron_num))
			dopa_save = np.array([])
			correct = 0.

			pbar_epi = progressbar.ProgressBar()
			for i in pbar_epi(range(rnd_images.shape[0])):
				explore_epi=np.copy(self.explore) if e>=self.n_epi_crit else 'none'
				
				#propagate image through the network
				classif, conv_input, conv_activ, subs_activ, feedf_activ, class_activ, class_activ_noise = self._propagate(rnd_images[i,:,:], explore=explore_epi)

				#compute reward prediction, reward delivery and dopamine release
				reward_pred = hp.reward_prediction(np.argmax(class_activ), np.argmax(class_activ_noise))
				reward = hp.reward_delivery(rnd_labels[i], self.classes[np.argmax(class_activ_noise)])
				dopa_release = hp.dopa_release(reward_pred, reward)
					
				# update weights...
				#...of the convolutional maps
				bs = self.batch_size
				for b in range(self.conv_neuron_num/bs):
					dopa_release_conv = hp.dopa_value(dopa_release, self.dopa_conv)*np.ones(bs) if explore_epi=='conv' else None
					self.conv_W = self._learning_step(conv_input[b*bs:(b+1)*bs, :], conv_activ[b*bs:(b+1)*bs, :], self.conv_W, dopa=dopa_release_conv)

				#...of the feedforward layer
				dopa_release_feedf = hp.dopa_value(dopa_release, self.dopa_feedf) if explore_epi=='feedf' else None
				self.feedf_W = self._learning_step(subs_activ, feedf_activ, self.feedf_W, dopa=dopa_release_feedf)

				#...of the classification layer	
				dopa_release_class = hp.dopa_value(dopa_release, self.dopa_class)
				self.class_W = self._learning_step(feedf_activ, class_activ, self.class_W, dopa=dopa_release_class)

				dopa_save = np.append(dopa_save, dopa_release)
				correct += float(self.classes[np.argmax(class_activ)] == rnd_labels[i])
				last_neuron_class[np.argmax(feedf_activ), np.argwhere(rnd_labels[i]==self.classes)] += 1

			self.perf_train[e] = correct
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
		n_images = images.shape[0]

		classResults = np.zeros(len(labels))
		self.perf_test = 0.
		pbar_epi = progressbar.ProgressBar()
		for i in pbar_epi(range(images.shape[0])):
			class_output = self._propagate(images[i,:,:])[0]
			if self.classes[class_output] == labels[i]: self.perf_test += 1.
			classResults[i] = self.classes[class_output]
		print "test error: %.2F%%" % ((1. - self.perf_test/images.shape[0]) * 100)

		for ilabel,label in enumerate(self.classes):
			for iclassif, classif in enumerate(self.classes):
				classifiedAs = np.sum(np.logical_and(labels==label, classResults==classif))
				overTot = np.sum(labels==label)
				self.CM[ilabel, iclassif] = float(classifiedAs)/overTot

		return (1. - self.perf_test/n_images)

	def _init_weights(self):
		"""	Initializes weights of the network, either randomly or by loading weights from init_file """
		
		self.class_neuron_num 	= len(self.classes)
		self.conv_neuron_num 	= (self.images_side - self.conv_filter_side + 1)**2
		self.conv_map_side 		= int(np.sqrt(self.conv_neuron_num))
		self.subs_map_side 		= self.conv_map_side/2
		self.CM 				= np.zeros((self.class_neuron_num, self.class_neuron_num))

		if self.init_file != '' and self.init_file != None:
			self._init_weights_file()
		else:
			self._init_weights_random()

	def _init_weights_random(self):
		""" initialize weights of the network randomly or by loading saved weights from file """
		
		conv_W_size = ( self.conv_filter_side**2 , self.conv_map_num )
		conv_W_norm = self.A/(self.conv_filter_side**2) + 2.5
		self.conv_W = np.random.random_sample(size=conv_W_size) + conv_W_norm
		
		feedf_W_size = ( (self.subs_map_side**2) * self.conv_map_num , self.feedf_neuron_num )
		# feedf_W_norm = float(self.subs_map_side**2) / ( (self.subs_map_side**2) * self.conv_map_num) + 0.6
		# self.feedf_W = np.random.random_sample(size=feedf_W_size)/1000 + feedf_W_norm
		self.feedf_W = np.random.random_sample(size=feedf_W_size) + 1.0
		
		class_W_size = ( self.feedf_neuron_num, self.class_neuron_num )
		self.class_W = (np.random.random_sample(size=class_W_size) /1000+1.0) / self.feedf_neuron_num

	def _init_weights_file(self):
		""" initialize weights of the network by loading saved weights from file """

		if not os.path.exists(self.init_file):
			raise IOError, "weight file \'%s\' not found" % self.init_file
		
		f = open(self.init_file, 'r')
		saved_net = pickle.load(f)
		f.close()

		self.conv_W = saved_net.conv_W
		self.feedf_W = saved_net.feedf_W
		self.class_W = saved_net.class_W

		del saved_net

	def _propagate(self, image, explore='none', noise_distrib=0.2):
		""" 
		Propagates a single image through the network and return its classification along with activation of neurons in the network. 

		Args:
			images (numpy array): 2D input image to propagate
			explore (str, optional): determines in which layer to add exploration noise; correct values are 'none', 'conv', 'feedf'
			noise_distrib (int, optional): extend of the uniform distribution from which noise is drawn for exploration

		returns:
			(int): classifcation of the network
			(numpy array): input to the convolutional filters
			(numpy array): activation of the convolutional filters
			(numpy array): activation of the subsampling layer
			(numpy array): activation of the feedforward layer
			(numpy array): activation of the classification layer *without* addition of noise for exploration
			(numpy array): activation of the classification layer *with* addition of noise for exploration

		"""

		#get input to the convolutional filter
		conv_input = np.zeros((self.conv_neuron_num, self.conv_filter_side**2))
		conv_input = hp.get_conv_input(image, conv_input, self.conv_filter_side)
		conv_input = hp.normalize_numba(conv_input, self.A)

		#activate convolutional feature maps
		conv_activ = hp.propagate_layerwise(conv_input, self.conv_W, SM=False)
		if explore=='conv':
			conv_activ_noise = conv_activ + np.random.normal(0, np.std(feedf_activ)*noise_distrib, np.shape(feedf_activ))
			conv_activ_noise = hp.softmax(conv_activ_noise, t=self.t)
			#subsample feature maps
			subs_activ_noise = hp.subsample(conv_activ_noise, self.conv_map_side, self.conv_map_num, self.subs_map_side)
		conv_activ = hp.softmax(conv_activ, t=self.t)

		#subsample feature maps
		subs_activ = hp.subsample(conv_activ, self.conv_map_side, self.conv_map_num, self.subs_map_side)

		#activate feedforward layer
		feedf_activ = hp.propagate_layerwise(subs_activ, self.feedf_W, SM=False)

		#add exploration
		if explore=='feedf':
			feedf_activ_noise = feedf_activ + np.random.normal(0, np.std(feedf_activ)*noise_distrib, np.shape(feedf_activ))
		elif explore=='conv':
			feedf_activ_noise = hp.propagate_layerwise(subs_activ_noise, self.feedf_W, SM=False)
		if explore=='feedf' or explore=='conv':
			feedf_activ_noise = hp.softmax(feedf_activ_noise, t=self.t)
			class_activ_noise = hp.propagate_layerwise(feedf_activ_noise, self.class_W, SM=True, t=0.001)
		
		feedf_activ = hp.softmax(feedf_activ, t=self.t)

		#activate classification layer
		class_activ = hp.propagate_layerwise(feedf_activ, self.class_W, SM=True, t=0.001)

		if explore=='none':
			return np.argmax(class_activ), conv_input, conv_activ, subs_activ, feedf_activ, class_activ, class_activ
		elif explore=='feedf':
			return np.argmax(class_activ), conv_input, conv_activ, subs_activ, feedf_activ_noise, class_activ, class_activ_noise
		elif explore=='conv':
			return np.argmax(class_activ), conv_input, conv_activ_noise, subs_activ, feedf_activ, class_activ, class_activ_noise

	def _learning_step(self, pre_neurons, post_neurons, W, dopa=None, numba=True):
		"""
		One learning step for the hebbian network; computes changes in weight, adds the change to the weights and impose bound on weights

		Args:
			pre_neurons (numpy array): activation of the pre-synaptic neurons
			post_neurons (numpy array): activation of the post-synaptic neurons
			W (numpy array): Weight matrix
			dopa (numpy array, optional): learning rate increase for the effect of dopamine

		returns:
			numpy array: change in weight; must be added to the weight matrix W
		"""
		if dopa is None: 
			dopa = np.ones(post_neurons.shape[0])
		elif isinstance(dopa, float): 
			dopa = np.array([dopa])

		if numba:
			post_neurons_lr = hp.disinhibition(post_neurons, self.lr, dopa, np.zeros_like(post_neurons)) #adds the effect of dopamine to the learning rate
			dot = np.dot(pre_neurons.T, post_neurons_lr)
			dW = hp.regularization(dot, post_neurons_lr, W, np.zeros(post_neurons_lr.shape[1]))
		else:
			post_neurons_lr = post_neurons * (self.lr * dopa[:,np.newaxis]) #adds the effect of dopamine to the learning rate  
			dW = (np.dot(pre_neurons.T, post_neurons_lr) - np.sum(post_neurons_lr, 0)*W)
		
		W += dW
		W = np.clip(W, 1e-10, np.inf)

		return W

















