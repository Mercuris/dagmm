import os
import torch


def send_to_gpu(x):
    """ Send a variable to the GPU if it is available """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def relative_euclidean_distance(a, b):
    """ Compute the relative euclidean distance between a and b """
    return (a - b).norm(2, dim=1) / a.norm(2, dim=1)


def make_directory(directory):
    """ Creates the given directory if it doesn't already exists """
    if not os.path.exists(directory):
        os.makedirs(directory)


def initialize_weights(module):
    """ Initialize the weights and the biases of the linear layers of the DAGMM """
    if type(module) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)  # Xavier Uniform (0)
        module.bias.data.fill_(0.01)  # Almost 0
