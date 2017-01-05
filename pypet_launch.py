"""
Author: Raphael Holca-Lamarre
Date: 04/01/2016

This code uses PyPet to explore the parameters of the convolutional neural network.


"""

import os
import matplotlib
if 'mnt' in os.getcwd(): matplotlib.use('Agg') #to avoid sending plots to screen when working on the servers
import numpy as np
import datetime
import time
import pypet
import pickle
import helper
import hebbian_cnn
import pypet_helper as pp
np.random.seed(0)

reload(helper)
pp = reload(pp)

""" static parameters """
parameter_dict = {	'conv_dHigh'			: 0.0,#2.7,
					'conv_dMid' 			: 0.0,#1.8,
					'conv_dNeut' 			: 0.0,#-0.07,
					'conv_dLow' 			: 0.0,#-1.8,
					'feedf_dHigh'			: 4.5,
					'feedf_dMid' 			: 0.02,
					'feedf_dNeut' 			: 0.01, 
					'feedf_dLow' 			: -2.0,
					'name' 					: 'pypet_feedf_greedy',
					'n_epi_crit' 			: 0,
					'n_epi_dopa' 			: 10,
					'A' 					: 900.,
					'lr_conv' 				: 0.01,
					'lr_feedf' 				: 0.01,
					't' 					: 0.01,
					'batch_size' 			: 196,
					'conv_map_num' 			: 20,
					'conv_filter_side'		: 5,
					'feedf_neuron_num'		: 49,
					'explore_layer'			: 'feedf',
					'dopa_layer'			: 'feedf',
					'noise_explore'			: 0.0,
					'classifier'			: 'neural_prob',
					'init_file' 			: 'output/pre_trained_all_classes/Network',
					'seed' 					: 952
					}

""" explored parameters """
explore_dict = {	
					# 'conv_dHigh'			: [+4.00, +8.00, +12.0, +2.00, +2.50],
					# 'conv_dNeut'			: [-0.10, -0.25, -0.75, +2.00, +2.50],
 
					# 'conv_dMid'			: [-0.01, +0.00, +0.01, +2.00, +2.50],
					# 'conv_dLow'			: [-1.00, -2.00, -3.00, +2.00, +2.50]

					# 'feedf_dHigh'			: [+0.50, +1.00, +1.50, +2.00, +2.50],
					# 'feedf_dNeut'			: [+0.50, +1.00, +1.50, +2.00, +2.50],

					'feedf_dMid'			: [+0.00, +0.10, +0.20, +0.50, +1.00],
					'feedf_dLow'			: [-4.00, -3.00, -2.00, -1.00, -0.00],
				}

""" load and pre-process images """
images_train, labels_train, images_test, labels_test = helper.load_images(	
																			classes 		= np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int),
																			# classes 		= np.array([4, 7, 9], dtype=int),
																			dataset_train	= 'train',
																			dataset_path 	= '/Users/raphaelholca/Documents/data-sets/MNIST',
																			pad_size 		= (parameter_dict['conv_filter_side']-1)/2,
																			load_test 		= True
																			)

""" create directory to save data """
parameter_dict['pypet'] = True
parameter_dict['pypet_name'] = parameter_dict['name']
save_path = os.path.join('output', parameter_dict['name'])
pp.check_dir(save_path, overwrite=False)
print_dict = parameter_dict.copy()
print_dict.update(explore_dict)

""" create pypet environment """
env = pypet.Environment(trajectory 		= 'explore_perf',
						log_stdout		= False,
						add_time 		= False,
						multiproc 		= True,
						ncores 			= 10,
						filename		=  os.path.join(save_path, 'explore_perf.hdf5'))


traj = env.v_trajectory
pp.add_parameters(traj, parameter_dict)

explore_dict = pypet.cartesian_product(explore_dict, tuple(explore_dict.keys())) #if not all entry of dict need be explored through cartesian product replace tuple(.) only with relevant dict keys in tuple

explore_dict['name'] = pp.set_run_names(explore_dict, parameter_dict['name'])
traj.f_explore(explore_dict)

""" launch simulation with pypet for parameter exploration """
tic = time.time()
env.f_run(pp.launch_exploration, images_train, labels_train, images_test, labels_test, save_path)
toc = time.time()

""" save parameters to file """
helper.print_params(print_dict, save_path, runtime=toc-tic)

""" plot results """
name_best = pp.plot_results(folder_path=save_path)
if len(explore_dict.keys())==5: pp.faceting(save_path)

print '\nrun name:\t' + parameter_dict['name']
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + str(datetime.timedelta(seconds=toc-tic))







































