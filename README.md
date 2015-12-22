# Reward-based Hebbian Convolutional Neural Network

This neural network is an extension of a previous single-layer, reward-based learning network. Learning does not require gradient information or error back-propagation. Weight updates are carried through Hebbian learning augmented with reward-based learning.

In it's shallow form, the network marginally outperfoms gradient-based network of similar architecture on the MNIST dataset (3.3% versus 4.7% error rate; single hidden layer with 300 hidden neurons). The present network extends the shallow architecure to include a convolutional layer, a subsampling layer, an all-to-all feedforward layer and a classification layer. At the moment, the additional layers do not improve the performance of the network (but this is work in progress...).

#### Hebbian Learning

Hebbian learning is a biology-inspired learning paradigm stating that "neurons that fire together wire together." Hebbian learning requires only synaptically-local information: pre-synaptic activation, post-synaptic activation and local synaptic strength (for normalization). In this form, Hebbian learning is a statistical learning method, akin to clustering algorithms. 

#### Reward-based learning

Reward-based learning is a learning theory related to reinforcement learning. It explains how animals learn from rewards they receive from their environment. Reward-based learning is not limited to action selection but also affects perceptual systems, for instance during discrimation learning. Reward-based learning is thought to be carried by dopamine. Dopamine is released following reward prediction errors (RPEs), the difference between a predicted and received reward. Dopaminergic fibers send projections throughout the cortex, broadcasting the RPEs signal. In the cortex, dopamine has a disinhibitory effect known to promote synpatic plasticity.

The learning mechanism in the present network is inpisred by dopamine signalling. At presentation of each stimulus, the network makes a reward prediction (if the network follows a greedy 'policy', it will always predict a reward). If the network makes a correct classification, it receives a reward. Reward predictions and deliveries are binary (0,1). The difference between the predicted and delivered rewards makes a RPE; this RPE is broadcasted to all synapses of the network, mimmicking dopamine signalling in animals. The RPE signal affects the learning rate of the network for the current input. Negative RPEs have the learning rate multiplied by a negative constant; positive RPEs have the learning rate multiplied by a positive constant; null RPEs have the learning rate multiplied by zero.

This learning mechanism augments the statistical learning of Hebbian learning to include error information. It leads to a progressive refinements of the weights and effectively minimizes the reward signal (although no explicit error gradient is used).

#### Examples

###### Weights of the convolutional maps (20 maps, 5x5 pixels)
![conv_W](https://github.com/raphaelholca/hebbianCNN/blob/master/docs/conv_W.png)

###### Projective fields of the weights of the feedforward layer (49 hidden neurons)
