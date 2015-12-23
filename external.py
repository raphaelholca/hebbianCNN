""" support functions for the convolutional hebbian network """

import numpy as np
import pickle
import os
import struct
import numba
import matplotlib.pyplot as plt
from array import array

def init_weights(net, init_file=''):
	""" initialize weights of the network, either randomly or by loading saved weights from file """
	
	#load pre-trained weights from file
	if init_file != '':
		if not os.path.exists(init_file):
			raise IOError, "weight file \'%s\' not found" % init_file
		
		f = open(init_file, 'r')
		weights = pickle.load(f)
		f.close()

		conv_W = weights['conv_W']
		feedf_W = weights['feedf_W']
		class_W = weights['class_W']
	
	#random initialization
	else:
		conv_W_size = ( net.conv_filter_side**2 , net.conv_map_num )
		conv_W_norm = net.A/(net.conv_filter_side**2) + 2.5
		conv_W = np.random.random_sample(size=conv_W_size) + conv_W_norm
		
		feedf_W_size = ( (net.subs_map_side**2) * net.conv_map_num , net.feedf_neuron_num )
		feedf_W_norm = float(net.subs_map_side**2) / ( (net.subs_map_side**2) * net.conv_map_num) + 0.6
		feedf_W = np.random.random_sample(size=feedf_W_size)/1000 + feedf_W_norm
		
		class_W_size = ( net.feedf_neuron_num, net.class_neuron_num )
		class_W = (np.random.random_sample(size=class_W_size) /1000+1.0) / net.feedf_neuron_num

	return conv_W, feedf_W, class_W

def load_images(classes, dataset_train, dataset_path, pad_size=2, load_test=True):
	""" load images training and testing images """
	
	if not os.path.exists(dataset_path):
		raise IOError, "MNIST dataset not found; dataset path does not exists: %s" % dataset_path

	images_train, labels_train 	= load_preprocess_mnist(classes, dataset_train, dataset_path, pad_size)

	if load_test:
		dataset_test = 'test' if dataset_train=='train' else 'train'
		images_test, labels_test = load_preprocess_mnist(classes, dataset_test, dataset_path, pad_size)
	else:
		images_test, labels_test = np.copy(images_train), np.copy(labels_train)

	return images_train, labels_train, images_test, labels_test

def load_preprocess_mnist(classes, dataset, dataset_path, pad_size):
	""" Loads, evens out and pads MNIST images """

	print "importing and pre-processing " + dataset + " images..."
	images, labels = load_mnist(classes = classes, dataset = dataset, path = dataset_path)
	images, labels = even_labels(images, labels, classes)
	images = add_padding(images, pad_size=pad_size, pad_value=0)
	images += 1e-5 #to avoid division by zero error when the convolving filter takes as input a patch of images that is filled with 0s

	return images, labels

def load_mnist(classes, dataset = 'train', path = '/Users/raphaelholca/Documents/data-sets/MNIST'):
    """
    Imports the MNIST dataset

    	Args:
    		classes (numpy array): classes of the MNIST dataset to be imported
    		classes (str): dataset to import; can be 'train' or 'test'

    	returns:
    		images and labels of the MNIST dataset
    """

    if dataset is "train":
    	if not os.path.exists(os.path.join(path, 'train-images.idx3-ubyte')):
			raise IOError, "MNIST dataset not found; please make sure file \'%s\' exists" % 'train-images.idx3-ubyte'
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "test":
    	if not os.path.exists(os.path.join(path, 't10k-images.idx3-ubyte')):
			raise IOError, "MNIST dataset not found; please make sure file \'%s\' exists" % 't10k-images.idx3-ubyte'
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'test' or 'train'"

    flbl = open(fname_lbl, 'rb')
    struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    size, rows, cols = struct.unpack(">IIII", fimg.read(16))[1:4]
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in classes ]
    images = np.zeros(shape=(len(ind), rows*cols))
    labels = np.zeros(shape=(len(ind)), dtype=int)
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i] = lbl[ind[i]]

    return images, labels

def add_padding(images, pad_size, pad_value=0.):
	""" 
	Adds padding around images and reshapes images vector to 3D (1D all images + 2D spatial dimensions).

	Args:
		images (2D numpy array): images to add the padding to; size = (images_num x nPixels_1D)
		pad_size (int): amount of padding to add on each 4 sides of the 2D image
		pad_value (float, optional): value of the padding to add

	returns:
		3D numpy array; size = (images_num x nPixels_2D x nPixels_2D)
	"""

	images_num = np.size(images, 0)
	sqrt_pixels = int(np.sqrt(np.size(images,1)))
	images = np.reshape(images, (images_num, sqrt_pixels, sqrt_pixels))
	images_padded = np.ones((images_num, sqrt_pixels+pad_size*2, sqrt_pixels+pad_size*2))*pad_value

	images_padded[:, pad_size:sqrt_pixels+pad_size, pad_size:sqrt_pixels+pad_size] = images
	return images_padded

def shuffle_images(images, labels):
	""" Shuffles images and labels """

	rnd_index = range(images.shape[0])
	np.random.shuffle(rnd_index)
	rnd_images = images[rnd_index,:,:]
	rnd_labels = labels[rnd_index]

	return rnd_images, rnd_labels

def even_labels(images, labels, classes):
	"""
	Evens out images and labels distribution so that they are evenly distributed over the labels.

	Args:
		images (numpy array): images
		labels (numpy array): labels constant
		classes (numpy array): classes of the MNIST dataset to be imported

	returns:
		numpy array: evened-out images
		numpy array: evened-out labels
	"""

	n_classes = len(classes)
	n_digits, bins = np.histogram(labels, bins=10, range=(0,9))
	m = np.min(n_digits[n_digits!=0])
	images_even = np.zeros((m*n_classes, np.size(images,1)))
	labels_even = np.zeros(m*n_classes, dtype=int)
	for i, c in enumerate(classes):
		images_even[i*m:(i+1)*m,:] = images[labels==c,:][0:m,:]
		labels_even[i*m:(i+1)*m] = labels[labels==c][0:m]
	images, labels = np.copy(images_even), np.copy(labels_even)
	return images, labels

def propagate(net, image, explore='none', noise_distrib=50):
	""" 
	Propagates a single image through the network and return its classification along with activation of neurons in the network. 

	Args:
		net (Network object): network on which to perform the propagation
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
	conv_input = np.zeros((net.conv_neuron_num, net.conv_filter_side**2))
	conv_input = get_conv_input(image, conv_input, net.conv_filter_side)
	conv_input = normalize_numba(conv_input, net.A)

	#activate convolutional feature maps
	conv_activ = propagate_layerwise(conv_input, net.conv_W, SM=False)
	if explore=='conv':
		conv_activ_noise = conv_activ + np.random.uniform(0, noise_distrib, np.shape(conv_activ))
		conv_activ_noise = softmax(conv_activ_noise, t=net.t)
		#subsample feature maps
		subs_activ_noise = subsample(conv_activ_noise, net.conv_map_side, net.conv_map_num, net.subs_map_side)
	conv_activ = softmax(conv_activ, t=net.t)

	#subsample feature maps
	subs_activ = subsample(conv_activ, net.conv_map_side, net.conv_map_num, net.subs_map_side)

	#activate feedforward layer
	feedf_activ = propagate_layerwise(subs_activ, net.feedf_W, SM=False)

	#add exploration
	if explore=='feedf':
		feedf_activ_noise = feedf_activ + np.random.uniform(0, noise_distrib, np.shape(feedf_activ))
	elif explore=='conv':
		feedf_activ_noise = propagate_layerwise(subs_activ_noise, net.feedf_W, SM=False)
	if explore=='feedf' or explore=='conv':
		feedf_activ_noise = softmax(feedf_activ_noise, t=net.t)
		class_activ_noise = propagate_layerwise(feedf_activ_noise, net.class_W, SM=True, t=0.001)
	
	feedf_activ = softmax(feedf_activ, t=net.t)

	#activate classification layer
	class_activ = propagate_layerwise(feedf_activ, net.class_W, SM=True, t=0.001)

	if explore=='none':
		return np.argmax(class_activ), conv_input, conv_activ, subs_activ, feedf_activ, class_activ, class_activ
	elif explore=='feedf':
		return np.argmax(class_activ), conv_input, conv_activ, subs_activ, feedf_activ_noise, class_activ, class_activ_noise
	elif explore=='conv':
		return np.argmax(class_activ), conv_input, conv_activ_noise, subs_activ, feedf_activ, class_activ, class_activ_noise

@numba.njit
def get_conv_input(image, conv_input, conv_side):
	"""
	Gets the input to the convolution weight matrix; must be pure python for numba implementation.

	Args:
		image (2D numpy array): image to get the input from; size = (images_side x images_side)
		conv_input (2D numpy array): empty input array to be filled; size = (conv_neuronNum x conv_side**2)
		conv_side (int): size of the convolutional filter

	returns:
		2D numpy array; size = (conv_neuronNum x conv_side**2)
	"""
	images_side = image.shape[0]

	im=0
	for i in range(images_side-conv_side+1):
		for j in range(images_side-conv_side+1):
			select = image[i:i+conv_side, j:j+conv_side]
			km=0
			for k in range(conv_side):
				for l in range(conv_side):
					conv_input[im,km] = select[k,l]
					km+=1
			im+=1
	return conv_input	

def subsample(conv_activ, conv_map_side, conv_map_num, subs_map_side):
	"""
	Subsamples the convolutional feature maps

	Args:
		conv_activ (numpy array): activation of the convolutional layer
		conv_map_side (int): size of the convolutional maps
		conv_map_num (int): number of maps in the conv layer
		subs_map_side (int): size of the subsampled conv maps

	returns:
		subsampled feature maps
	"""
	FM = np.reshape(conv_activ, (conv_map_side, conv_map_side, conv_map_num))
	SSM = np.zeros((subs_map_side, subs_map_side, conv_map_num))
	ite = np.arange(0, conv_map_side, 2)
	SSM = subsampling_numba(FM, SSM, ite)
	SSM = softmax( np.reshape(SSM, (subs_map_side**2, conv_map_num) ), t=1. )
	subs_activ = np.reshape(SSM, (-1))[np.newaxis,:]

	return subs_activ

@numba.njit
def subsampling_numba(FM, SSM, ite):
	"""
	Numba implementation of the subsample routine; must be pure python.

	Args:
		FM (3D numpy array): feature maps; size = (cMaps_side x cMaps_side x nFeatureMaps)
		SSM (3D numpy array): empty subsampled feature maps to be filled; size = (cMaps_side/2 x cMaps_side/2 x nFeatureMaps)
		ite (1D numpy array): iterator used over FM (contains np.arange(0, cMaps_side, 2))

	returns:
		3D numpy array; subsampled feature maps; size = (cMaps_side/2 x cMaps_side/2 nFeatureMaps)
	"""
	for f in range(FM.shape[2]):
		for im in range(ite.shape[0]):
			i=ite[im]
			for jm in range(ite.shape[0]):
				j=ite[jm]
				select = FM[i:i+2,j:j+2,f]
				tmp_sum=0
				for k in range(2):
					for l in range(2):
						tmp_sum = tmp_sum + select[k,l]
				SSM[im,jm,f] = tmp_sum
	return SSM

@numba.njit
def normalize_numba(images, A):
	"""
	numba-optimized version of the normalize function; Normalizes each image to the sum of its pixel value (equivalent to feedforward inhibition)

	Args:

		images (numpy array): image to normalize
		A (int): normalization constant

	returns:
		numpy array: normalized images
	"""
	A_i = (A-images.shape[1])
	for im in range(images.shape[0]):
		sum_px = 0
		for px in range(images.shape[1]):
			sum_px += images[im,px]
		for px in range(images.shape[1]):
			images[im,px] = A_i*images[im,px]/sum_px + 1.

	return images

def softmax(activ, implementation='numba', t=1.):
	"""
	Softmax function (equivalent to lateral inhibition, or soft winner-take-all)

	Args:
		activ (numpy array): activation of neurons to be fed to the function; should be (training examples x neurons)
		vectorial (str, optional): which implementation to use ('vectorial', 'iterative', 'numba')
		t (float): temperature parameter; determines the sharpness of the softmax, or the strength of the competition

	returns:
		numpy array: the activation fed through the softmax function
	"""

	#vectorial
	if implementation=='vectorial':
		activ_norm = np.copy(activ - np.max(activ,1)[:,np.newaxis])
		activ_SM = np.exp((activ_norm)/t) / np.sum(np.exp((activ_norm)/t), 1)[:,np.newaxis]

	#iterative
	elif implementation=='iterative':
		activ_SM = np.zeros_like(activ)
		for i in range(np.size(activ,0)):
			scale = 0
			I = np.copy(activ[i,:])
			if (I[np.argmax(I)] > 700):
			    scale  = I[np.argmax(I)] - 700
			if (I[np.argmin(I)] < -740 + scale):
			    I[np.argmin(I)] = -740 + scale
			activ_SM[i,:] = np.exp((I-scale)/t) / np.sum(np.exp((I-scale)/t))

	#iterative with numba
	elif implementation=='numba':
		activ_SM = np.zeros_like(activ)
		activ_SM = softmax_numba(activ, activ_SM, t=t)
	
	return activ_SM

@numba.njit
def softmax_numba(activ, activ_SM, t=1.):
	""" Numba implementation of the softmax function """

	for ex in range(activ.shape[0]):
		sum_tot = 0.
		ex_max=np.max(activ[ex,:])
		for i in range(activ.shape[1]): #compute exponential
			activ_SM[ex,i] = np.exp((activ[ex,i]-ex_max)/t)
			sum_tot += activ_SM[ex,i]
		for i in range(activ.shape[1]): #divide by sum of exponential
			activ_SM[ex,i] /= sum_tot

	return activ_SM

def propagate_layerwise(input, W, SM=True, t=1.):
	"""
	One propagation step

	Args:
		input (numpy array): input vector to the neurons of layer 1
		W (numpy matrix): weight matrix; shape: (input neurons x hidden neurons)
		SM (bool, optional): whether to pass the activation throught the Softmax function
		t (float): temperature parameter for the softmax function (only passed to the function, not used here)

	returns:
		numpy array: the activation of the output neurons
	"""

	output = np.dot(input, np.log(W))
	if SM: output = softmax(output, t=t)
	return output

def reward_prediction(best_action, action_taken):
	"""
	Computes reward prediction based on the best (greedy) action and the action taken (differ when noise is added for exploration)

	Args:
		best_action (numpy array): best (greedy) action for each trial of a batch
		action_taken (numpy array): action taken for each trial of a batch

	returns:
		numpy array: reward prediction (either 0 or 1)
	"""

	reward_prediction = int(best_action==action_taken)

	return reward_prediction

def reward_delivery(labels, actions):
	"""
	Computes the reward based on the action taken and the label of the current input

	Args:
		labels (numpy array): image labels
		actions (numpy array): action taken

	returns:
		numpy array: 1 (reward) for correct label-action pair, otherwise 0
	"""

	reward = int(labels==actions)

	return reward

def dopa_release(predicted_reward, delivered_reward):
	"""
	Computes the dopamine signal based on the actual and predicted rewards

	Args:
		predicted_reward (numpy array, bool): predicted reward (True, False)
		delivered_reward (numpy array, int): reward received (0, 1)

	returns:
		numpy array: array of dopamine release value
	"""

	if 		predicted_reward==0 and delivered_reward==1: dopa = 'dHigh'				#unpredicted reward
	elif 	predicted_reward==1 and delivered_reward==1: dopa = 'dMid'				#correct reward prediction
	elif	predicted_reward==0 and delivered_reward==0: dopa = 'dNeut'				#correct no reward prediction
	elif 	predicted_reward==1 and delivered_reward==0: dopa = 'dLow'				#incorrect reward prediction

	return dopa

def dopa_value(dopa_rel, dHigh, dMid, dNeut, dLow):
	""" Returns the value associated with each dopa release """

	if 		dopa_rel == 'dHigh'	: dopa_val = dHigh				#unpredicted reward
	elif 	dopa_rel == 'dMid'	: dopa_val = dMid				#correct reward prediction
	elif	dopa_rel == 'dNeut'	: dopa_val = dNeut				#correct no reward prediction
	elif 	dopa_rel == 'dLow' 	: dopa_val = dLow				#incorrect reward prediction

	return dopa_val

def learning_step(net, pre_neurons, post_neurons, W, dopa=None, numba=True):
	"""
	One learning step for the hebbian network; computes changes in weight, adds the change to the weights and impose bound on weights

	Args:
		net (Network object): object on which to learn
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
		post_neurons_lr = disinhibition(post_neurons, net.lr, dopa, np.zeros_like(post_neurons)) #adds the effect of dopamine to the learning rate
		dot = np.dot(pre_neurons.T, post_neurons_lr)
		dW = regularization(dot, post_neurons_lr, W, np.zeros(post_neurons_lr.shape[1]))
	else:
		post_neurons_lr = post_neurons * (net.lr * dopa[:,np.newaxis]) #adds the effect of dopamine to the learning rate  
		dW = (np.dot(pre_neurons.T, post_neurons_lr) - np.sum(post_neurons_lr, 0)*W)
	
	W += dW
	W = np.clip(W, 1e-10, np.inf)

	return W

@numba.njit
def disinhibition(post_neurons, lr, dopa, post_neurons_lr):
	""" Support function for numba implementation of learning_step(). Performs the disinhibition/increase in learning rate due to dopamine. Must be pure python. """
	for b in range(post_neurons.shape[0]):
		for pn in range(post_neurons.shape[1]):
			post_neurons_lr[b, pn] = post_neurons[b, pn] * lr * dopa[b]

	return post_neurons_lr

@numba.njit
def regularization(dot, post_neurons, W, sum_ar):
	""" Support function for numba implementation of learning_step(). Regularization of Hebbian learning, prevents weight explosion. Must be pure python. """
	for j in range(post_neurons.shape[1]):
		for i in range(post_neurons.shape[0]):
			sum_ar[j] += post_neurons[i,j]
	
	for i in range(dot.shape[0]):
		for j in range(dot.shape[1]):
			dot[i,j] -= W[i,j] * sum_ar[j]

	return dot

def reconstruct(net, W, display_all=False):
	"""
	Reconstructs weights in the layer following the convolutional layer OR reconstructs an input image based on the activation in the sub-sampling layer

	Args:
		conv_W (2D numpy array): convolutional filter; size=(filter_side**2, L1_map_num)
		W (1D or 3D numpy array): weights following the conv+subs layer OR activation in the sub-sampling layer; size=(L1_subs_mapSide, L1_subs_mapSide, L1_map_num)
	"""
	filter_side = int(np.sqrt(np.size(net.conv_W,0)))
	L1_map_num = np.size(net.conv_W,1)
	n_pixels = net.images_side
	step=1 if n_pixels==18 else 2

	conv_W_plot = np.reshape(net.conv_W, (filter_side, filter_side, L1_map_num))
	W = np.reshape(W, (net.subs_map_side, net.subs_map_side, L1_map_num))

	recon = np.zeros((n_pixels, n_pixels, L1_map_num))
	for f in range(L1_map_num):
		im=0
		for i in range(n_pixels-filter_side+1)[::step]:
			jm=0
			for j in range(n_pixels-filter_side+1)[::step]:
				recon[i:i+filter_side,j:j+filter_side,f] += conv_W_plot[:,:,f] * W[im,jm,f]
				jm+=1
			im+=1

		if display_all:
			plt.figure()
			plt.imshow(recon[:,:,f], interpolation='nearest', cmap='Greys', vmin=np.min(recon), vmax=np.max(recon))

	recon_sum = np.zeros((n_pixels,n_pixels))
	for f in range(L1_map_num):
		recon_sum[:,:]+=recon[:,:,f]/np.max(recon[:,:,f])
	
	recon_sum[::2,::2]/=1.7
	recon_sum[1::2,1::2]*=1.7

	return recon_sum











