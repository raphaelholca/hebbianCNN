""" convolutional hebbian neural network """
import numpy as np
import external as ex
import matplotlib.pyplot as plt
import progressbar

ex = reload(ex)

class Network:
	""" This is a Hebbian convolutional neural network """
	
	def __init__(self, name, n_epi_crit, n_epi_dopa, A, lr, t, batch_size, conv_map_num, conv_filter_side, feedf_neuron_num, explore):
		""" 
		Sets network parameters 

			Args:
				name (str): name of the network, used to save network to disk
				n_epi_crit (int): number of statistical pre-training steps (pure Hebbian)
				n_epi_dopa (int): number of dopamine-mediated training steps
				A (float): parameter for the normalization of the input images (pixel values sum to A)
				lr (float): learning rate of the network
				t (float): temperature of the softmax function ('softness' of the winner-take-all)
				batch_size (int): size of training batch
				conv_map_num (int): number of convolutional filter maps
				conv_filter_side (int): size of each convolutional filter (side of filter in pixel; total number of pixel in filter is conv_filter_side^2)
				feedf_neuron_num (int): number of neurons in the feedforward layer
				explore (str): determines in which layer to perform exploration by noise addition. Valid value: 'none', 'conv', 'feedf'
		"""
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
		self.explore 				= explore

	def init_weights(self, images_side, n_classes, init_file=''):
		""" initialize weights of the network, either random or by loading weights from init_file """
		self.images_side 		= images_side
		self.class_neuron_num 	= n_classes
		self.init_file			= init_file
		self.conv_neuron_num 	= (images_side - self.conv_filter_side + 1)**2
		self.conv_map_side 		= int(np.sqrt(self.conv_neuron_num))
		self.subs_map_side 		= self.conv_map_side/2

		self.conv_W, self.feedf_W, self.class_W = ex.init_weights(self, init_file)

	def train(self, images, labels):
		""" Train hebbian convolutional neural network. """
		print "training network..."
		classes = np.sort(np.unique(labels))

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
					dopa_conv = ex.dopa_value(dopa_release, dHigh=3.375, dMid=2.25, dNeut=-0.09, dLow=-2.25)*np.ones(bs) if explore_epi=='conv' else None
					self.conv_W = ex.learning_step(self, conv_input[b*bs:(b+1)*bs, :], conv_activ[b*bs:(b+1)*bs, :], self.conv_W, dopa=dopa_conv)

				#...of the feedforward layer
				dopa_feedf = ex.dopa_value(dopa_release, dHigh=3.375, dMid=2.25, dNeut=-0.09, dLow=-2.25) if explore_epi=='feedf' else None
				self.feedf_W = ex.learning_step(self, subs_activ, feedf_activ, self.feedf_W, dopa=dopa_feedf)

				#...of the classification layer	
				dopa_class = ex.dopa_value(dopa_release, dHigh=0.375, dMid=0.375, dNeut=-0.25, dLow=-0.25)
				self.class_W = ex.learning_step(self, feedf_activ, class_activ, self.class_W, dopa=dopa_class)

				dopa_save = np.append(dopa_save, dopa_release)
				correct += float(classes[np.argmax(class_activ)] == rnd_labels[i])
				last_neuron_class[np.argmax(feedf_activ), np.argwhere(rnd_labels[i]==classes)] += 1

			correct_class_W = np.sum(np.argmax(last_neuron_class,1)==np.argmax(self.class_W,1))
			print "train error: %.2F%%" % ((1. - correct/rnd_images.shape[0]) * 100)
			print "correct W_out assignment: %d/%d" % (correct_class_W, self.feedf_neuron_num)

	def test(self, images, labels):
		""" test network """
		print "\ntesting network..."
		classes = np.sort(np.unique(labels))

		step = images.shape[0]/(8000/len(classes))
		if step==0: step=1
		images=images[::step,:,:]
		labels=labels[::step]

		correct = 0.
		pbar_epi = progressbar.ProgressBar()
		for i in pbar_epi(range(images.shape[0])):
			classif = ex.propagate(self, images[i,:,:])[0]
			if classes[classif] == labels[i]: correct += 1.
		print "test error: %.2F%%" % ((1. - correct/images.shape[0]) * 100)

	def plot_weights(self):
		""" Plots convolutional and feedforward weights """

		#convolutional filter
		n_rows = int(np.sqrt(self.conv_map_num))
		n_cols = int(np.ceil(self.conv_map_num/float(n_rows)))
		fig = plt.figure(figsize=(n_cols,n_rows))
		
		for f in range(self.conv_map_num):
			plt.subplot(n_rows, n_cols, f+1)
			conv_W_square = np.reshape(self.conv_W[:,f], (self.conv_filter_side, self.conv_filter_side))
			# plt.imshow(conv_W_square, interpolation='nearest', cmap='Greys', vmin=np.min(self.conv_W), vmax=np.max(self.conv_W))
			plt.imshow(conv_W_square, interpolation='nearest', cmap='Greys', vmin=np.min(self.conv_W[:,f]), vmax=np.max(self.conv_W[:,f]))
			plt.xticks([])
			plt.yticks([])
		fig.patch.set_facecolor('white')
		plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)
		plt.show(block=False)

		#reconstruction of preojective fields of output neurons
		n_rows = int(np.sqrt(self.feedf_neuron_num))
		n_cols = self.feedf_neuron_num/n_rows
		fig = plt.figure(figsize=(n_cols,n_rows))
		
		for n in range(self.feedf_neuron_num):
			plt.subplot(n_rows, n_cols, n)
			W = np.reshape(self.feedf_W[:,n], (self.subs_map_side, self.subs_map_side, self.conv_map_num))
			recon_sum = ex.reconstruct(self, W, display_all=False)
			plt.imshow(recon_sum, interpolation='nearest', cmap='Greys')
			plt.xticks([])
			plt.yticks([])
		fig.patch.set_facecolor('white')
		plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)
		plt.show(block=False)



























