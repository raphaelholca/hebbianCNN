# Biology-inspired Convolutional Neural Network

This neural network is an extension of a previous single-layer, reward-based and Hebbian learning network. Learning does not require gradient information or error back-propagation. Instead, weight updates are carried through Hebbian learning augmented with reward-based learning. The network also doesn't have access direct access label information but only to rewards for correct classifications.

In it's shallow form, the network marginally outperfoms gradient-based network of similar architecture on the MNIST dataset (3.3% versus 4.7% error rate; single hidden layer with 300 hidden neurons). The present network extends the shallow architecure to include a convolutional layer, a subsampling layer, an all-to-all feedforward layer and a classification layer. At the moment, the additional layers do not improve the performance of the network (but this is work in progress...).

#### Hebbian Learning

Hebbian learning is a biology-inspired learning paradigm stating that "neurons that fire together wire together." Hebbian learning requires only synaptically-local information: pre-synaptic activation, post-synaptic activation and local synaptic strength. In this form, Hebbian learning is a statistical learning method, akin to clustering algorithms. 

#### Reward-based learning

Reward-based learning is a learning theory related to reinforcement learning. It explains how animals learn from rewards they receive from their environment. Reward-based learning is not limited to action selection but also affects perceptual systems, for instance during discrimation learning. Reward-based learning is thought to be mediated by the neurotrasmitter dopamine. Dopamine is released following reward prediction errors (RPEs), the difference between a predicted and received reward. Dopaminergic neurons in the midbrain send projections throughout the neocortex, broadcasting RPE signals to most cortical areas. In the cortex, dopamine has a disinhibitory effect known to promote synpatic plasticity.

The learning mechanism in this neural network is inpisred from dopamine signalling. At presentation of a stimulus, the network outputs a classification for the stimulus. The network also makes a reward prediction (if the network follows a greedy 'policy', it will always predict a reward). If the network correctly classified its input, it then receives a reward; if it incorrectly classified its input, it receives no reward. Reward predictions and deliveries are binary (0,1). The difference between the predicted and delivered rewards makes a RPE; this RPE is broadcasted to all synapses of the network, mimmicking dopamine signalling in animals. The RPE signal affects the learning rate of the network for the current input. Negative RPEs have the learning rate multiplied by a negative constant; positive RPEs have the learning rate multiplied by a positive constant; null RPEs have the learning rate multiplied by zero.

This reward-based learning mechanism augments the statistical learning of Hebbian learning to include error information. It leads to a progressive refinements of the weights and effectively minimizes the error signal (although no explicit error gradient is used).

#### Examples

###### Weights of the convolutional maps (20 maps, 5x5 pixels)
<p align="center">
<img src=https://github.com/raphaelholca/hebbianCNN/blob/master/docs/conv_W.png width=400 />
</p>

###### Reconstructed projective fields of neurons in the feedforward layer (49 hidden neurons)
<p align="center">
<img src=https://github.com/raphaelholca/hebbianCNN/blob/master/docs/feedf_W.png width=600 />
</p>

###### Comparison of weights and classification performance in a network with only Hebbian learning and in a network with both Hebbian and reward-based learning
<p align="center">
<img src=https://github.com/raphaelholca/hebbianCNN/blob/master/docs/post-pre_simple.pdf width=600 />
</p>

#### Code use

The convolutional neural network is object-oriented. The Network class initialize various parameters of the network. These are:

- name (str): name of the network, used to save network to disk
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

The Network object is trained using the train function of the Network class. The train function takes images and labels and modifies the weights of the Network object. The train function returns the training performance of the network.

The performance of the Network object can be tested using the test function of the Network class. The test function takes images and labels and returns the performance of the network of this test dataset.

The weights of the Network object can be vizualize using the plot_weights function of the Network class.
