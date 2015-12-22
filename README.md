# Hebbian Convolutional Neural Network
Convolutional deep neural network with biology-inspired learning (Hebbian and reward-based learning)

This neural network consists of a convolutional layer, a subsampling layer, an all-to-all feedforward layer and a classification layer. Learning does not require any gradient information or error back-propagation. Weight updates are carried through Hebbian learning augmented with reward-based learning.

Hebbian learning is a biology-inspired learning rule stating that "neurons that fire together wire together." Hebbian learning requires only synaptically-local information: pre-synaptic activation, post-synaptic activation and local synaptic strength (for normalization). In this form, Hebbian learning is a statistical learning method, akin to clustering algorithms. 

Reward-based learning is inspired from dopamine signalling in animals. Dopamine release in animals follows reward prediction errors (RPE).
