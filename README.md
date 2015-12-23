# Biology-inspired Convolutional Neural Network

This neural network is an extension of a previous single-layer, reward-based and Hebbian learning network. Learning does not require gradient information or error back-propagation. Instead, weight updates are carried through Hebbian learning augmented with reward-based learning. The network also doesn't have access direct access label information but only to rewards for correct classifications.

In it's shallow form, the network marginally outperfoms gradient-based network of similar architecture on the MNIST dataset (3.3% versus 4.7% error rate; single hidden layer with 300 hidden neurons). The present network extends the shallow architecure to include a convolutional layer, a subsampling layer, an all-to-all feedforward layer and a classification layer. At the moment, the additional layers do not improve the performance of the network (but this is work in progress...).

#### Hebbian Learning

Hebbian learning is a biology-inspired learning paradigm stating that "neurons that fire together wire together." Hebbian learning requires only synaptically-local information: pre-synaptic activation, post-synaptic activation and local synaptic strength. In this form, Hebbian learning is a statistical learning method, akin to clustering algorithms. 

#### Reward-based learning

Reward-based learning is a learning theory related to reinforcement learning. It explains how animals learn from rewards they receive from their environment. Reward-based learning is not limited to action selection but also affects perceptual systems, for instance during discrimation learning. Reward-based learning is thought to be mediated by the neurotrasmitter dopamine. Dopamine is released following reward prediction errors (RPEs), the difference between a predicted and received reward. Dopaminergic neurons in the midbrain send projections throughout the neocortex, broadcasting RPE signals to most cortical areas. In the cortex, dopamine has a disinhibitory effect known to promote synpatic plasticity.

The learning mechanism in this neural network is inpisred from dopamine signalling. At presentation of a stimulus, the network outputs a classification for the stimulus. The network also makes a reward prediction (if the network follows a greedy 'policy', it will always predict a reward). If the network correctly classified its input, it then receives a reward; if it incorrectly classified its input, it receives no reward. Reward predictions and deliveries are binary (0,1). The difference between the predicted and delivered rewards makes a RPE; this RPE is broadcasted to all synapses of the network, mimmicking dopamine signalling in animals. The RPE signal affects the learning rate of the network for the current input. Negative RPEs have the learning rate multiplied by a negative constant; positive RPEs have the learning rate multiplied by a positive constant; null RPEs have the learning rate multiplied by zero.

This reward-based learning mechanism augments the statistical learning of Hebbian learning to include error information. It leads to a progressive refinements of the weights and effectively minimizes the error signal (although no explicit error gradient is used).

## Examples

Weights and performance of the network when trained on the MNIST dataset.

###### Weights of the convolutional maps (20 maps, 5x5 pixels)
<p align="center">
<img src=https://github.com/raphaelholca/hebbianCNN/blob/master/docs/conv_W.png width=300 />
</p>

###### Reconstructed projective fields of neurons in the feedforward layer (49 hidden neurons)
<p align="center">
<img src=https://github.com/raphaelholca/hebbianCNN/blob/master/docs/feedf_W.png width=500 />
</p>

###### Comparison of weights and classification performance in a network with only Hebbian learning and in a network with both Hebbian and reward-based learning
<p align="center">
<img src=https://github.com/raphaelholca/hebbianCNN/blob/master/docs/post-pre_simple-1.png width=400 />
</p>

## Code use

The convolutional neural network is object-oriented. The Network class initializes parameters of the neural network and contains functions to initialize the weights, train and test the network and plot the weights of the network. 

##### hebbian_cnn.Network(name='net', n_epi_crit=10, n_epi_dopa=10, A=900., lr=0.01, t=0.01, batch_size=196, conv_map_num=5, conv_filter_side=5, feedf_neuron_num=49, explore='feedf')

Hebbian convolutional neural network with reward-based learning

###### Parameters:
- name (str, optional): name of the network, used to save network to disk. Default: 'net'
- n_epi_crit (int, optional): number of statistical pre-training steps (pure Hebbian). Default: 10
- n_epi_dopa (int, optional): number of dopamine-mediated training steps. Default: 10
- A (float, optional): parameter for the normalization of the input images (pixel values sum to A). Default: 900
- lr (float, optional): learning rate of the network. Default: 0.01
- t (float, optional): temperature of the softmax function ('softness' of the winner-take-all). Default: 0.01
- batch_size (int, optional): size of training batch. Default: 196
- conv_map_num (int, optional): number of convolutional filter maps. Default: 5
- conv_filter_side (int, optional): size of each convolutional filter (side of filter in pixel; total number of pixel in filter is conv_filter_side^2). Default: 5
- feedf_neuron_num (int, optional): number of neurons in the feedforward layer. Default: 49
- explore (str, optional): determines in which layer to perform exploration by noise addition. Valid value: 'none', 'conv', 'feedf'. Default: 'feedf'

###### Methods:
- init_weights(images_side, n_classes, init_file="")
- train(images, labels)
- test(images, labels)
- plot_weights()

##### init_weights(images_side, n_classes, init_file="")

Initializes weights of the network, either random or by loading weights from init_file 

###### Args:
- images_side (int): side of the input images in pixels (total pixel number in image if images_side^2).
- n_classes (int): number of classes in the dataset. Used to set the number of neurons in the classificaion layer.
- init_file (str, optional): path to file where weights are saved. Give path to load pretrained weights from file or leave empty for random weigth initialization. Default: "" (random initialization)

##### train(images, labels)

Trains Hebbian convolutional neural network

###### Args: 
- images (3D numpy array): images to train the Network on. Images matrix must be 3D: [images_num, images_side, images_side] 
- labels (1D numpy array): labels of the training images.

###### returns:
- (float): training performance of the network.

##### test(images, labels)

Tests Hebbian convolutional neural network

###### Args: 
- images (3D numpy array): images to test the Network on. Images matrix must be 3D: [images_num, images_side, images_side] 
- labels (1D numpy array): labels of the testing images.

###### returns:
- (float): testing performance of the network.

##### plot_weights()

Plots convolutional and feedforward weights
