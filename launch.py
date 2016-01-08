"""
Author: Raphael Holca-Lamarre
Date: 26/05/2015

This code creates a hebbian convolutional neural network object and trains it on the MNIST dataset. The network consists of four layers: a convolutional layer, a subsampling layer, a feedforward layer and a classification layer. Only three of these layers have modifiable weights: the convolution, feedforward and classification layers. The learning rule is a hebbian learning rule augmented with a learning mechanism inspired from dopamine signalling in animal cortex.
"""

import numpy as np
import helper
import hebbian_cnn

reload(helper)
reload(hebbian_cnn)

np.random.seed(951)

""" create hebbian convolution neural network """
net = hebbian_cnn.Network(	dopa_conv			= {'-e+r':2.7, '+e+r':1.8, '-e-r':-0.07, '+e-r':-1.8},
							dopa_feedf			= {'-e+r':2.7, '+e+r':1.8, '-e-r':-0.07, '+e-r':-1.8},
							dopa_class			= {'-e+r':0.3, '+e+r':0.3, '-e-r':-0.2, '+e-r':-0.2},
							name 				= 'test',
							n_epi_crit 			= 0,
							n_epi_dopa 			= 1,
							A 					= 900.,
							lr 					= 0.01,
							t 					= 0.01,
							batch_size 			= 196,
							conv_map_num 		= 20,
							conv_filter_side	= 5,
							feedf_neuron_num	= 49,
							explore				= 'feedf'
							)

""" load and pre-process training and testing images """
images_train, labels_train, images_test, labels_test = helper.load_images(	classes 		= np.array([ 4, 7, 9 ], dtype=int),
																			dataset_train	= 'train',
																			dataset_path 	= '/Users/raphaelholca/Documents/data-sets/MNIST',
																			pad_size 		= (net.conv_filter_side-1)/2,
																			load_test 		= True
																			)

""" initialize weights of network """
net.init_weights(	images_side 	= np.size(images_train, 2), 
					n_classes		= len(np.unique(labels_train)),
					init_file 		= 'output/pre_trained/Network'
					)

""" train network """
perf_train = net.train(images_train, labels_train)

""" test network """
perf_test = net.test(images_test, labels_test)

""" plot weights of the network """
plots = helper.generate_plots(net)

""" save network to disk """
helper.save(net, overwrite=False, plots={})




