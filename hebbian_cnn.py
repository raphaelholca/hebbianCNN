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
import time

hp = reload(hp)

class Network:
	""" Hebbian convolutional neural network with reward-based learning """
	
	def __init__(self, conv_dHigh, conv_dMid, conv_dNeut, conv_dLow, feedf_dHigh, feedf_dMid, feedf_dNeut, feedf_dLow, name='net', n_epi_crit=10, n_epi_dopa=10, A=900., lr_conv=0.01, lr_feedf=0.01, t=0.01, batch_size=196, conv_map_num=5, conv_filter_side=5, feedf_neuron_num=49, explore_layer='feedf', dopa_layer='feedf', noise_explore=0.2, classifier='neural_prob', init_file=None, seed=None, pypet=False, pypet_name=''):
		""" 
		Sets network parameters 

			Args:
				conv_dHigh (float): values of dopamine release in the convolutional layer for -reward expectation, +reward delivery
				conv_dMid (float): values of dopamine release in the convolutional layer for +reward expectation, +reward delivery
				conv_dNeut (float): values of dopamine release in the convolutional layer for -reward expectation, -reward delivery
				conv_dLow (float): values of dopamine release in the convolutional layer for +reward expectation, -reward delivery
				feedf_dHigh (float): values of dopamine release in the feedforward layer for -reward expectation, +reward delivery
				feedf_dMid (float): values of dopamine release in the feedforward layer for +reward expectation, +reward delivery
				feedf_dNeut (float): values of dopamine release in the feedforward layer for -reward expectation, -reward delivery
				feedf_dLow (float): values of dopamine release in the feedforward layer for +reward expectation, -reward delivery
				name (str, optional): name of the network, used to save network to disk. Default: 'net'
				n_epi_crit (int, optional): number of statistical pre-training steps (pure Hebbian). Default: 10
				n_epi_dopa (int, optional): number of dopamine-mediated training steps. Default: 10
				A (float, optional): parameter for the normalization of the input images (pixel values sum to A). Default: 900
				lr_conv (float, optional): learning rate of the network for the convolutional layer. Default: 0.01
				lr_feedf (float, optional): learning rate of the network for the feedforward layer. Default: 0.01
				t (float, optional): temperature of the softmax function ('softness' of the winner-take-all). Default: 0.01
				batch_size (int, optional): size of training batch. Default: 196
				conv_map_num (int, optional): number of convolutional filter maps. Default: 5
				conv_filter_side (int, optional): size of each convolutional filter (side of filter in pixel; total number of pixel in filter is conv_filter_side^2). Default: 5
				feedf_neuron_num (int, optional): number of neurons in the feedforward layer. Default: 49
				explore_layer (str, optional): in which layer to perform exploration by noise addition. Valid values: 'none', 'conv', 'feedf', 'both'. Default: 'feedf'
				dopa_layer (str, optional): in which layer to release dopamine. Valid values: 'none', 'conv', 'feedf', 'both'. Default: 'feedf'
				noise_explore (float, optional): parameter of the standard deviation of the normal distribution from which noise is drawn for exploration. Default: 0.2
				classifier (str, optional): which classifier to use as the output layer. Valid values: 'neural_dopa' (hebbian + dopa), 'neural_prob' (poisson mixture model). Default: 'neural_prob'
				init_file (str, optional): initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization. Default: None
				seed (int, optional): seed of the random number generator. Default: None
				pypet (bool, optional): whether the network simulation is part of pypet exploration. Default: False
				pypet_name (str, optional): name of the directory in which data is saved when doing pypet exploration. Default: ''
		"""
		self.dopa_conv 			= {'-e+r':conv_dHigh, '+e+r':conv_dMid, '-e-r':conv_dNeut, '+e-r':conv_dLow}
		self.dopa_feedf 		= {'-e+r':feedf_dHigh, '+e+r':feedf_dMid, '-e-r':feedf_dNeut, '+e-r':feedf_dLow}
		self.dopa_class 		= {'-e+r':0.3, '+e+r':0.3, '-e-r':-0.2, '+e-r':-0.2}
		self.name 				= name
		self.n_epi_crit 		= n_epi_crit
		self.n_epi_dopa 		= n_epi_dopa
		self.n_epi_tot 			= n_epi_crit + n_epi_dopa
		self.A 					= A
		self.lr_conv 			= lr_conv
		self.lr_feedf			= lr_feedf
		self.t 					= t
		self.batch_size 		= batch_size
		self.conv_map_num 		= conv_map_num
		self.conv_filter_side 	= conv_filter_side
		self.feedf_neuron_num 	= feedf_neuron_num
		self.explore_layer		= explore_layer
		self.dopa_layer 		= dopa_layer
		self.noise_explore 		= np.clip(noise_explore, 1e-20, np.inf)
		self.classifier 		= classifier
		self.init_file 			= init_file
		self.seed 				= seed
		self.perf_train 		= np.zeros(self.n_epi_tot)
		self.perf_test 			= 0.
		self.pypet 				= pypet
		self.pypet_name 		= pypet_name if pypet_name != '' else name
		
		np.random.seed(self.seed)

		hp.check_values(self)

	def train(self, images, labels):
		""" 
		Train Hebbian convolutional neural network

			Args: 
				images (3D numpy array): images to train the Network on. images matrix must be 3D: [images_num, images_side, images_side] 
				labels (1D numpy array): labels of the training images.

			returns:
				(float): training performance of the network.
		"""

		if not self.pypet: print "\ntraining network..."
		self._train_start = time.time()
		self.classes = np.sort(np.unique(labels))
		self.images_side = np.size(images, 2)
		self._init_weights()
		self.n_images = images.shape[0]
		self._feedf_activ_all = np.zeros((self.n_images, self.feedf_neuron_num))*np.nan
		self._labels_all = np.zeros((self.n_images))*np.nan
		correct = 0.

		for e in range(self.n_epi_tot):
			if not self.pypet: print "\ntrain episope: %d/%d" % (e+1, self.n_epi_tot)
			
			rnd_images, rnd_labels = hp.shuffle_images(images, labels)
			last_neuron_class = np.zeros((self.feedf_neuron_num, self.class_neuron_num))
			dopa_save = np.array([])
			correct = 0.

			loop_train = progressbar.ProgressBar()(range(rnd_images.shape[0])) if not self.pypet else range(rnd_images.shape[0])
			for i in loop_train:
				explore_epi=np.copy(self.explore_layer) if e>=self.n_epi_crit else 'none'
				dopa_layer_epi=np.copy(self.dopa_layer) if e>=self.n_epi_crit else 'none'
				
				#propagate image through the network
				classif, conv_input, conv_activ, subs_activ, feedf_activ, class_activ, class_activ_noise = self._propagate(rnd_images[i,:,:], explore=explore_epi, label=rnd_labels[i])

				#compute reward prediction, reward delivery and dopamine release
				reward_pred = hp.reward_prediction(np.argmax(class_activ), np.argmax(class_activ_noise))
				reward = hp.reward_delivery(rnd_labels[i], self.classes[np.argmax(class_activ_noise)])
				dopa_release = hp.dopa_release(reward_pred, reward)
					
				# update weights...
				#...of the convolutional maps
				bs = self.batch_size
				for b in range(self.conv_neuron_num/bs):
					dopa_release_conv = hp.dopa_value(dopa_release, self.dopa_conv)*np.ones(bs) if (dopa_layer_epi=='conv' or dopa_layer_epi=='both') else None
					self.conv_W = self._learning_step(conv_input[b*bs:(b+1)*bs, :], conv_activ[b*bs:(b+1)*bs, :], self.conv_W, lr=self.lr_conv, dopa=dopa_release_conv)

				#...of the feedforward layer
				dopa_release_feedf = hp.dopa_value(dopa_release, self.dopa_feedf) if (dopa_layer_epi=='feedf' or dopa_layer_epi=='both') else None
				self.feedf_W = self._learning_step(subs_activ, feedf_activ, self.feedf_W, lr=self.lr_feedf, dopa=dopa_release_feedf)

				#...of the classification layer
				if self.classifier == 'neural_dopa':
					dopa_release_class = hp.dopa_value(dopa_release, self.dopa_class)
					self.class_W = self._learning_step(feedf_activ, class_activ, self.class_W, lr=0.01, dopa=dopa_release_class)
				elif self.classifier == 'neural_prob':
					if i%100==0 and i!=0 : self._learn_out_proba()

				dopa_save = np.append(dopa_save, dopa_release)
				correct += float(self.classes[np.argmax(class_activ)] == rnd_labels[i])
				last_neuron_class[np.argmax(feedf_activ), np.argwhere(rnd_labels[i]==self.classes)] += 1

			self.perf_train[e] = correct/rnd_images.shape[0]
			correct_class_W = np.sum(np.argmax(last_neuron_class,1)==np.argmax(self.class_W,1))
			if not self.pypet: print "train performance: %.2F%%" % (self.perf_train[e] * 100)
			if not self.pypet: print "correct W_out assignment: %d/%d" % (correct_class_W, self.feedf_neuron_num)

		self._train_stop = time.time()
		self.runtime = self._train_stop - self._train_start

		return (correct/self.n_images)

	def test(self, images, labels):
		""" 
		Test Hebbian convolutional neural network

			Args: 
				images (3D numpy array): images to test the Network on. images matrix must be 3D: [images_num, images_side, images_side] 
				labels (1D numpy array): labels of the testing images.

			returns:
				(float): test performance of the network.
		"""

		if not self.pypet: print "\ntesting network..."

		class_results = np.zeros(len(labels))
		self.perf_test = 0.

		loop_test = progressbar.ProgressBar()(range(images.shape[0])) if not self.pypet else range(images.shape[0])
		for i in loop_test:
			class_results[i] = self.classes[self._propagate(images[i,:,:])[0]]
			
		self.perf_test = float(np.sum(class_results==labels))/len(labels)

		for ilabel,label in enumerate(self.classes):
			for iclassif, classif in enumerate(self.classes):
				classifiedAs = np.sum(np.logical_and(labels==label, class_results==classif))
				overTot = np.sum(labels==label)
				self.CM[ilabel, iclassif] = float(classifiedAs)/overTot

		if not self.pypet: hp.print_CM(self.perf_test, self.CM, self.classes)

		return self.perf_test

	def _init_weights(self):
		"""	Initializes weights of the network, either randomly or by loading weights from init_file """
		
		self.class_neuron_num 	= len(self.classes)
		self.conv_neuron_num 	= (self.images_side - self.conv_filter_side + 1)**2
		self.conv_map_side 		= int(np.sqrt(self.conv_neuron_num))
		self.subs_map_side 		= self.conv_map_side/2
		self.CM 				= np.zeros((self.class_neuron_num, self.class_neuron_num))

		if self.init_file == '' or self.init_file is None:
			self._init_weights_random()
		else:
			self._init_weights_file()

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

	def _propagate(self, image, explore='none', label=None):
		""" 
		Propagates a single image through the network and return its classification along with activation of neurons in the network. 

		Args:
			images (numpy array): 2D input image to propagate
			explore (str, optional): determines in which layer to add exploration noise; correct values are 'none', 'conv', 'feedf'
			label (int, optional): label of the current image

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
		if explore=='conv' or explore=='both':
			conv_activ_noise = conv_activ + np.random.normal(0, np.std(feedf_activ)*self.noise_explore, np.shape(feedf_activ))
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
			feedf_activ_noise = feedf_activ + np.random.normal(0, np.std(feedf_activ)*self.noise_explore, np.shape(feedf_activ))
		elif explore=='conv' or explore=='both':
			feedf_activ_noise = hp.propagate_layerwise(subs_activ_noise, self.feedf_W, SM=False)
		if explore=='both':
			feedf_activ_noise = feedf_activ_noise + np.random.normal(0, np.std(feedf_activ)*self.noise_explore, np.shape(feedf_activ))
		if explore=='feedf' or explore=='conv' or explore=='both':
			feedf_activ_noise = hp.softmax(feedf_activ_noise, t=self.t)
			if self.classifier == 'neural_dopa':
				class_activ_noise = hp.propagate_layerwise(feedf_activ_noise, self.class_W, SM=True, t=0.001)
			elif self.classifier == 'neural_prob':
				class_activ_noise = np.dot(feedf_activ_noise, self.class_W)
		
		feedf_activ = hp.softmax(feedf_activ, t=self.t)

		#activate classification layer
		if self.classifier == 'neural_dopa':
			class_activ = hp.propagate_layerwise(feedf_activ, self.class_W, SM=True, t=0.001)
		elif self.classifier == 'neural_prob':
			class_activ = np.dot(feedf_activ, self.class_W)

		#save activation of feedforward layer for computation of output weights
		if label is not None:
			self._feedf_activ_all = np.roll(self._feedf_activ_all, 1, axis=0)
			self._feedf_activ_all[0,:] = feedf_activ
			self._labels_all = np.roll(self._labels_all, 1)
			self._labels_all[0] = label

		if explore=='none':
			return np.argmax(class_activ), conv_input, conv_activ, subs_activ, feedf_activ, class_activ, class_activ
		elif explore=='feedf':
			return np.argmax(class_activ), conv_input, conv_activ, subs_activ, feedf_activ_noise, class_activ, class_activ_noise
		elif explore=='conv' or explore=='both':
			return np.argmax(class_activ), conv_input, conv_activ_noise, subs_activ_noise, feedf_activ_noise, class_activ, class_activ_noise

	def _learning_step(self, pre_neurons, post_neurons, W, lr, dopa=None, numba=True):
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
			post_neurons_lr = hp.disinhibition(post_neurons, lr, dopa, np.zeros_like(post_neurons)) #adds the effect of dopamine to the learning rate
			# dot = np.dot(pre_neurons.T, post_neurons_lr)
			dot = np.einsum('ij,jk', pre_neurons.T, post_neurons_lr)
			dW = hp.regularization(dot, post_neurons_lr, W, np.zeros(post_neurons_lr.shape[1]))
		else:
			post_neurons_lr = post_neurons * (lr * dopa[:,np.newaxis]) #adds the effect of dopamine to the learning rate  
			dW = (np.dot(pre_neurons.T, post_neurons_lr) - np.sum(post_neurons_lr, 0)*W)
		
		W += dW
		W = np.clip(W, 1e-10, np.inf)

		return W

	def _learn_out_proba(self):
		""" 
		learn output weights using using approximation of a Poisson Mixture Model 

		returns:
			nupmy array: updated weights classification weights
		"""

		for i_c, c in enumerate(self.classes):
			self.class_W[:,i_c] = np.nanmean(self._feedf_activ_all[self._labels_all==c,:],0)
		self.class_W = self.class_W/np.nansum(self.class_W, 1)[:,np.newaxis]















