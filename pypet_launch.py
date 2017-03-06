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
parameter_dict = {	'conv_dHigh'			: 0.5,
					'conv_dMid' 			: 0.1,
					'conv_dNeut' 			: -0.1,
					'conv_dLow' 			: -0.1,
					'feedf_dHigh'			: 4.5,
					'feedf_dMid' 			: 0.02,
					'feedf_dNeut' 			: 0.01, 
					'feedf_dLow' 			: -2.0,
					'name' 					: 'pypet_conv_fsize_9_explor_convnoise_4',
					'n_epi_crit' 			: 0,
					'n_epi_dopa' 			: 6,
					'A' 					: 900.,
					'lr_conv' 				: 2e-6, ##
					'lr_feedf' 				: 0.01,
					't_conv'				: 1.0,
					't_feedf'				: 1.0,
					'batch_size' 			: 196,
					'conv_map_num' 			: 20,
					'conv_filter_side'		: 9,
					'subs_stride' 			: 2,
					'feedf_neuron_num'		: 16,
					'explore_layer'			: 'conv',
					'dopa_layer'			: 'conv',
					'noise_explore'			: 0.2,
					'classifier'			: 'neural_prob',
					'init_file' 			: 'output/pretrain_fsize_9/pretrain_fsize_9', #'output/pretrain_lr_e-6_t_e-0/pretrain_lr_e-6_t_e-0_1',
					'seed' 					: 954
					}

""" explored parameters """
explore_dict = {	
					'conv_dHigh'			: [+2.0, +4.0, +6.0],
					'conv_dNeut'			: [-1.0, -0.5, -0.1],

					'conv_dMid'				: [+0.0, +0.1, +0.5],
					'conv_dLow'				: [-4.0, -2.0, -1.0]

					# 'feedf_dHigh'			: [+2.00, +6.00, +10.0],
					# 'feedf_dNeut'			: [-1.00, -0.50, -0.10],
					
					# 'feedf_dMid'			: [+0.00, +0.01, +0.10],
					# 'feedf_dLow'			: [-2.00, -1.00, -0.00]
					
					# 'conv_dMid'			: [-1.0, +0.0, +0.1],
					# 'conv_dLow'			: [-5.0, -2.0, -1.0, +0.0, +1.00],
					
				
				}

""" load and pre-process images """
images_train, labels_train, images_test, labels_test = helper.load_images(	
																			# classes 		= np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int),
																			classes 		= np.array([4, 7, 9], dtype=int),
																			dataset_train	= 'train',
																			dataset_path 	= '/Users/raphaelholca/Documents/data-sets/MNIST',
																			pad_size 		= (parameter_dict['conv_filter_side']-1)/2,
																			load_test 		= True
																			)

""" create directory to save data """
parameter_dict['pypet'] = True
parameter_dict['verbose'] = 0
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
						ncores 			= 6,
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
pp.faceting(save_path)

print '\nrun name:\t' + parameter_dict['name']
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + str(datetime.timedelta(seconds=toc-tic))







































