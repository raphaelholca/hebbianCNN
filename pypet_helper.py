import os
import numpy as np
import matplotlib.pyplot as plt
import hebbian_cnn
import helper
import pypet
import pickle
import shutil
import time
from scipy import stats

reload(helper)
reload(hebbian_cnn)

def launch_exploration(traj, images_train, labels_train, images_test, labels_test, save_path):
	""" launch all the exploration of the parameters """
	parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)
	try:
		test_perf = launch_one_exploration(parameter_dict, images_train, labels_train, images_test, labels_test, save_path)
	except ValueError:
		test_perf = [-1.]

	traj.f_add_result('test_perf', test_perf=test_perf)

def launch_one_exploration(parameter_dict, images_train, labels_train, images_test, labels_test, save_path):
	""" launch one instance of the network """

	net = hebbian_cnn.Network(**parameter_dict)

	net.train(images_train, labels_train)

	perf_test = net.test(images_test, labels_test)

	p_file = open(os.path.join(save_path, 'networks', net.name), 'w')
	pickle.dump(net, p_file)
	p_file.close()

	return perf_test

def add_parameters(traj, parameter_dict):
	for k in parameter_dict.keys():
		traj.f_add_parameter(k, parameter_dict[k])

def set_run_names(explore_dict, name):
	nXplr = len(explore_dict[explore_dict.keys()[0]])
	runName_list = [name for _ in range(nXplr)]
	for n in range(nXplr):
		for k in sorted(explore_dict.keys()):
			runName_list[n] += '_'
			runName_list[n] += k
			runName_list[n] += str(explore_dict[k][n]).replace('.', ',')
	return runName_list

def check_dir(save_path, overwrite=False):
	if os.path.isdir(save_path) and overwrite==True:
		shutil.rmtree(save_path)
	elif os.path.isdir(save_path) and overwrite==False:
		raise RuntimeError("trajectory already exists and will not be overwritten")
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	if not os.path.isdir(os.path.join(save_path, 'networks')):
		os.mkdir(os.path.join(save_path, 'networks'))

def plot_results(folder_path=''):
	if folder_path=='':
		folder_path = '/Users/raphaelholca/Documents/hebbian_cnn/output/test_pypet_0/'

	traj_name = 'explore_perf'
	traj = pypet.load_trajectory(traj_name, filename=os.path.join(folder_path, 'explore_perf.hdf5'), force=True)
	traj.v_auto_load = True

	perf_all = []
	for run in traj.f_iter_runs():
		perf_all.append(traj.results[run].test_perf)
	perf = np.array(perf_all)

	param_traj = traj.f_get_explored_parameters()
	param = {}
	for k in param_traj:
		if k[11:] != 'name':
			xplr_values = np.array(param_traj[k].f_get_range())
			param[k[11:]] = xplr_values

	arg_best = np.argmax(perf)

	best_param = {}

	print 'best parameters:'
	print '================'
	for k in param.keys():
		best_param[k] = param[k][arg_best]
		print k + ' : ' + str(param[k][arg_best]) + '\t\t' + str(np.round(np.unique(param[k]),3))
	print "\nbest performance: " + str(np.round(np.max(perf)*100,2)) + "\n"

	keys = param.keys()
	for ik in range(len(keys)):
		if len(keys)==1: ik=-1
		for k in keys[ik+1:]:
			others = keys[:]
			if len(keys)>1: 
				others.remove(keys[ik])
				others.remove(k)
			
			mask = np.ones_like(param[k], dtype=bool)
			if len(param)>2:
				for o in others:
					mask = np.logical_and(mask, param[o]==best_param[o])
			pX = param[keys[ik]][mask]
			pY = param[k][mask]
			rC = np.hstack(perf)[mask]

			if True: #True: non-linear representation of results; False: linear representation 
				ipX = np.zeros(len(pX))
				ipY = np.zeros(len(pY))
				for i in range(len(pX)):
					ipX[i] = np.argwhere(pX[i]==np.sort(np.unique(pX)))
					ipY[i] = np.argwhere(pY[i]==np.sort(np.unique(pY)))
			else:
				ipX = np.copy(pX)
				ipY = np.copy(pY)

			fig = plt.figure()
			fig.patch.set_facecolor('white')
			
			plt.scatter(ipX, ipY, c=rC, cmap='CMRmap', vmin=np.min(perf)-0.1, vmax=np.max(perf), s=1000, marker='s')
			# plt.scatter(param[keys[ik]][arg_best], param[k][arg_best], c='r', s=50, marker='x')
			for i in range(len(pX)):
				if pX[i]==param[keys[ik]][arg_best] and pY[i]==param[k][arg_best]:
					plt.text(ipX[i], ipY[i], str(np.round(rC[i]*100,1)), horizontalalignment='center', verticalalignment='center', weight='bold', bbox=dict(facecolor='red', alpha=0.5))
				else:
					plt.text(ipX[i], ipY[i], str(np.round(rC[i]*100,1)), horizontalalignment='center', verticalalignment='center')
			plt.xticks(ipX, pX)
			plt.yticks(ipY, pY)
			plt.xlabel(keys[ik], fontsize=25)
			plt.ylabel(k, fontsize=25)
			plt.tick_params(axis='both', which='major', labelsize=18)
			plt.tight_layout()
			plt.savefig(os.path.join(folder_path, keys[ik] + '_' + k + '.pdf'))

			plt.close(fig)

	name_best = ''
	for k in sorted(param.keys()):
		name_best += '_'
		name_best += k
		name_best += str(best_param[k]).replace('.', ',')

	""" plot best net """
	best_net = pickle.load(open(os.path.join(folder_path, 'networks', param_traj['parameters.name']._data + name_best), 'r'))
	plots = helper.generate_plots(best_net)
	os.mkdir(os.path.join(folder_path, 'best_net'))
	for plot in plots.keys():
		plots[plot].savefig(os.path.join(folder_path, 'best_net', plot))
		plt.close(plots[plot])

	return name_best

def bar_plot(best_param_all, best_perf_all=None):
	""" best release properties bar plot """
	order_bar 	= [best_param_all.keys()[0],	best_param_all.keys()[1],	best_param_all.keys()[2],	best_param_all.keys()[3]]
	n_measures = len(best_param_all[best_param_all.keys()[0]])
	normalize = False #normalize bar height so dHigh=1
	order_perf = False #order bar by best performance
	order_value = False #order bar by VTA value

	num_colors = 20
	color_list = np.array([plt.get_cmap('Paired')(1.*i/num_colors) for i in range(num_colors)])
	color_cycle_idx = np.arange(n_measures)%5 

	if normalize:
		for param in order_bar:
			best_param_all[param] = best_param_all[param]/best_param_all['dHigh']
	if order_perf and best_perf_all is not None:
		sorter = best_perf_all.argsort()[::-1]
		best_perf_all = best_perf_all[sorter]
		for param in order_bar:
			best_param_all[param] = best_param_all[param][sorter]

	fig, ax = plt.subplots(figsize=(7.5,4))
	fig.patch.set_facecolor('white')
	width = 0.8/n_measures

	for idx, param in enumerate(order_bar):
		if order_value:
			ax.bar(np.arange(n_measures)*width+idx, np.sort(best_param_all[param]+1e-10), width=width, edgecolor='r', color='r')
		else:
			ax.bar(np.arange(n_measures)*width+idx, best_param_all[param]+1e-10, width=width*0.8, edgecolor='r', color='r')
			# ax.bar(np.arange(n_measures)*width+idx, best_param_all[param]+1e-10, width=width, edgecolor=color_list[color_cycle_idx], color=color_list[color_cycle_idx])

	#x-axis
	ax.set_xticks(np.arange(len(order_bar))+0.4)
	ax.set_xticklabels([order_bar[i] for i in range(len(order_bar))], fontsize=18)
	ax.set_xlabel('reward prediction error', fontsize=18)
	ax.xaxis.set_ticks_position('bottom')
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	# ax.hlines(0, 0, 4)

	#y-axis
	ax.set_ylabel('DA value', fontsize=18)
	ax.yaxis.set_ticks_position('left')
	ax.spines['right'].set_visible(False)

	#both axes
	ax.tick_params(axis='both', which='major', direction='out', width=2, labelsize=18)
	fig.set_tight_layout(True)

	return fig, ax

def import_traj(folder_path, file_name, traj_name='explore_perf'):
	print "importing data..."
	traj = pypet.load_trajectory(traj_name, filename=os.path.join(folder_path, file_name+'.hdf5'), force=True)
	traj.v_auto_load = True

	perc_correct = np.array([])
	perc_correct_all = []
	ok_runs = []
	for run in traj.f_iter_runs():
		perc_correct = np.append(perc_correct, np.mean(traj.results[run].test_perf))
		perc_correct_all.append(traj.results[run].test_perf)
		ok_runs.append(int(run[4:]))

	param_traj = traj.f_get_explored_parameters()
	param = {}
	for k in param_traj:
		if k[11:] != 'name':
			xplr_values = np.array(param_traj[k].f_get_range())[ok_runs]
			if len(np.unique(xplr_values)) >= 1:
				param[k[11:]] = xplr_values

	return perc_correct, np.array(perc_correct_all), param

def faceting(folder_path, savefig=True, stat_test=False, threshold=0.002, t_threshold=0.10, vmin=0.8, vmax=0.97, fontsize_label=20, fontsize_tick=18, print_best_perf=True):
	"""
	Function to plot the results of gridsearch parameter exploration

	Args:
		folder_path (str): path to data file.
		savefig (bool, optional): whether to save the figure to file (True) or to return the figure (False). Default: True
		stat_test (bool, optional): whether to use statistical testing (True) or relative difference (False) to check if performance is equivalent. Default: False
		threshold (float, optional): percent of difference in between max perf and other perf to be considered the same. Default: 0.002
		t_threshold (float, optional): threshold p-value for the t-test. Default: 0.10
		vmin (float, optional): min value for plots. Default: 0.80
		vmax (float, optional): max value for plots. Default: 0.97
		fontsize_label (int, optional): font size of label. Default: 20
		fontsize_tick (int, optional): font size of tick. Default: 18
		best_perf (bool, optional): whether to write value of best perfomance on plot. Default: True
		name_pred (str, optional): signifies which labels to use: '+pred', 'delta', or 'dMid'. Default: 'dMid'
	"""
	file_name = 'explore_perf'

	perc_correct, perc_correct_all, param = import_traj(folder_path, file_name)

	order_face = param.keys() #['dNeut', 'dHigh', 'dMid', 'dLow'] #order is x1, y1, x2, y2
	n_measures = len(np.unique(param[param.keys()[0]]))

	#find best parameters combination
	arg_best = np.argmax(perc_correct)
	best_param = {}
	print 'best parameters:'
	print '================'
	for k in param.keys():
		best_param[k] = param[k][arg_best]
		print k + ' : ' + str(param[k][arg_best]) + '\t\t' + str(np.round(np.unique(param[k]),2))
	best_perf_str = str(int(np.max(perc_correct)*1000)/10.) + '%'
	print '\nbest performance: ' + best_perf_str

	invert=True if np.argwhere(param[order_face[0]]!= param[order_face[0]][0])[0] > np.argwhere(param[order_face[1]]!= param[order_face[1]][0])[0] else False

	#find similarly good parameters...
	if stat_test and np.size(perc_correct_all,1)>1:#...using statistical significance testing
		arg_best_all = np.array([], dtype=int)
		best_perf = perc_correct_all[arg_best, :]
		for arg in range(np.size(perc_correct_all,0)):
			t, prob = stats.ttest_ind(best_perf, perc_correct_all[arg, :], equal_var=True) #two-sided t-test with independent samples
			if prob > t_threshold: #not statistically significantly different
				arg_best_all = np.append(arg_best_all, arg)
	else: #...within [threshold]% of best performance
		arg_best_all = np.argwhere(perc_correct >= np.max(perc_correct)-threshold*np.max(perc_correct))
	best_param_all = {}
	for k in param.keys():
		best_param_all[k] = np.hstack(param[k][arg_best_all])
	best_perf_all = np.hstack(perc_correct[arg_best_all])

	#creates a structured array of performance
	perc_correct_struct = np.zeros(len(perc_correct), dtype=[('perf', float), (param.keys()[0], float), (param.keys()[1], float), (param.keys()[2], float), (param.keys()[3], float)])
	perc_correct_struct['perf'] = perc_correct
	for k in param.keys():
		perc_correct_struct[k] = param[k]

	""" faceting plot """
	fig_faceting, ax_faceting = plt.subplots(n_measures,n_measures, figsize=(8,7))#, sharex=True, sharey=True)
	fig_faceting.patch.set_facecolor('white')
	for x2_i, x2 in enumerate(np.sort(np.unique(param[order_face[2]]))):
		for y2_i, y2 in enumerate(np.sort(np.unique(param[order_face[3]]))[::-1]):
			tmp_perc = np.copy(perc_correct_struct[np.logical_and(param[order_face[2]]==x2, param[order_face[3]]==y2)])
			tmp_perc[order_face[2]] *= -1
			tmp_perc = np.sort(tmp_perc, order=[order_face[1], order_face[0]])
			tmp_perc_sqrd = np.reshape(tmp_perc, (n_measures, n_measures))
			
			ax_faceting[y2_i, x2_i].imshow(tmp_perc_sqrd['perf'], origin='lower', interpolation='nearest', cmap='CMRmap', vmin=vmin, vmax=vmax)

			x1 = np.unique(param[order_face[0]][np.logical_and(x2==param[order_face[2]] , y2==param[order_face[3]])])
			y1 = np.unique(param[order_face[1]][np.logical_and(x2==param[order_face[2]] , y2==param[order_face[3]])])

			#indicate all params that give statistically greater slopes after than before training
			# if x2 in param_greater_slope[order_face[2]][y2==param_greater_slope[order_face[3]]]:
			# 	mask_best = np.logical_and(x2==param_greater_slope[order_face[2]], y2==param_greater_slope[order_face[3]])
			# 	x1_dots = []
			# 	y1_dots = []
			# 	for x in param_greater_slope[order_face[0]][mask_best]: x1_dots.append(np.argwhere(x==x1))
			# 	for y in param_greater_slope[order_face[1]][mask_best]: y1_dots.append(np.argwhere(y==y1))
			# 	ax_faceting[y2_i, x2_i].scatter(x1_dots, y1_dots, marker ='s', s=60, c='w', edgecolor='k', linewidths=0.5)

			# indicate all params that give performance within [threshold]% of best performance
			if x2 in best_param_all[order_face[2]][y2==best_param_all[order_face[3]]]:
				mask_best = np.logical_and(x2==best_param_all[order_face[2]], y2==best_param_all[order_face[3]])
				x1_dots = []
				y1_dots = []
				for x in best_param_all[order_face[0]][mask_best]: x1_dots.append(np.argwhere(x==x1))
				for y in best_param_all[order_face[1]][mask_best]: y1_dots.append(np.argwhere(y==y1))
				ax_faceting[y2_i, x2_i].scatter(x1_dots, y1_dots, s=30, c='r', edgecolor='k')

			# indicate params that give best performance
			if x2==best_param[order_face[2]] and y2==best_param[order_face[3]]:
				x1_dot = np.argwhere(best_param[order_face[0]] == x1)
				y1_dot = np.argwhere(best_param[order_face[1]] == y1)
				ax_faceting[y2_i, x2_i].scatter(x1_dot, y1_dot, s=100, c='r', marker='*')

			if y2_i==0: #ticks for top row
				ax_faceting[y2_i,x2_i].set_yticks([])
				ax_faceting[y2_i,x2_i].set_xticks([n_measures/2])
				ax_faceting[y2_i,x2_i].xaxis.set_ticks_position('top')
				ax_faceting[y2_i,x2_i].set_xticklabels([str(x2)], fontsize=fontsize_tick)
				ax_faceting[y2_i,x2_i].tick_params(axis='both', which='major', direction='out')
			if x2_i==n_measures-1: #ticks for right row
				if y2_i!=0:
					ax_faceting[y2_i,x2_i].set_xticks([])
				ax_faceting[y2_i,x2_i].set_yticks([n_measures/2])
				ax_faceting[y2_i,x2_i].yaxis.set_ticks_position('right')
				ax_faceting[y2_i,x2_i].set_yticklabels([str(y2)], fontsize=fontsize_tick)
				ax_faceting[y2_i,x2_i].tick_params(axis='both', which='major', direction='out')
			elif y2_i!=0:
				ax_faceting[y2_i,x2_i].set_xticks([])
				ax_faceting[y2_i,x2_i].set_yticks([])

			ax_faceting[y2_i,x2_i].set_xlim(-.5, n_measures-.5)
			ax_faceting[y2_i,x2_i].set_ylim(-.5, n_measures-.5)

	ax_faceting[n_measures-1,0].set_xticks(range(n_measures)[::2])
	ax_faceting[n_measures-1,0].xaxis.set_ticks_position('bottom')
	# ax_faceting[n_measures-1,0].set_xticklabels(x1[::2], fontsize=fontsize_tick)
	# ax_faceting[n_measures-1,0].set_xticklabels([i.replace('0.', '.') for i in map(str,x1)], fontsize=fontsize_tick)
	ax_faceting[n_measures-1,0].set_xticklabels([i.replace('0.', '.') for i in map(str,x1)[::2]], fontsize=fontsize_tick)

	ax_faceting[n_measures-1,0].set_yticks(range(n_measures)[::2])
	ax_faceting[n_measures-1,0].yaxis.set_ticks_position('left')
	ax_faceting[n_measures-1,0].set_yticklabels(y1[::2], fontsize=fontsize_tick)
	ax_faceting[n_measures-1,0].tick_params(axis='both', which='major', direction='out')

	ax_faceting[n_measures-1,0].set_xlabel(order_face[0], fontsize=fontsize_label)
	ax_faceting[n_measures-1,0].set_ylabel(order_face[1], fontsize=fontsize_label)
		
	fig_faceting.text(0.5, 0.96, order_face[2], ha='center', fontsize=fontsize_label)
	fig_faceting.text(0.965, 0.5, order_face[3], va='center', rotation=90, fontsize=fontsize_label)
	if print_best_perf:
		fig_faceting.text(0.025, 0.975, 'best perf: '+best_perf_str, va='center', fontsize=fontsize_tick, weight='semibold')

	fig_bar, ax_bar = bar_plot(best_param_all, best_perf_all)

	if savefig:
		fig_faceting.savefig(os.path.join(folder_path, 'faceting.pdf'), format='pdf')
		fig_bar.savefig(os.path.join(folder_path, 'release_values.pdf'), format='pdf')
	else:
		return fig_faceting, ax_faceting, fig_bar, ax_bar














