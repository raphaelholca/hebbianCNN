"""
Author: Raphael Holca-Lamarre
Date: 26/05/2015

This function runs a hebbian convolutional neural network. The network consists of four layers: a convolutional layer, a subsampling layer, a feedforward layer, and a classification layer. Only the three of these layers have modifiable weights: the convolution, feedforward and classification layers. The learning rule is a hebbian learning rule with the addition of a learning mechanism inspired from dopamine signalling in animal cortex.
"""

import numpy as np
import external
import hebbian_cnn

reload(external)
reload(hebbian_cnn)

np.random.seed(951)

""" create hebbian convolution neural network """
net = hebbian_cnn.Network(	name 				= 'test',
							n_epi_crit 			= 5,
							n_epi_dopa 			= 5,
							A 					= 900.,
							lr 					= 1.5e-2,
							t 					= 0.01,
							batch_size 			= 196,
							conv_map_num 		= 20,
							conv_filter_side	= 5,
							feedf_neuron_num	= 49,
							explore				= 'feedf'
							)

""" load and pre-process training and testing images """
images_train, labels_train, images_test, labels_test = external.load_images(#classes 		= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int),
																			classes 		= np.array([4, 7, 9], dtype=int),
																			dataset_train	= 'train',
																			dataset_path 	= '/Users/raphaelholca/Documents/data-sets/MNIST',
																			pad_size 		= (net.conv_filter_side-1)/2,
																			load_test 		= True
																			)

""" initialize weights of network """
net.init_weights(	images_side 	= np.size(images_train, 2), 
					n_classes		= len(np.unique(labels_train)),
					init_file 		= ''
					)

""" train network """
net.train(images_train, labels_train)

""" test network """
net.test(images_test, labels_test)

""" plot weights of the network """
net.plot_weights()






