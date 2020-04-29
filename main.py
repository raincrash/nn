
"""

Example code to show CIFAR-10 classification.
Install Pytorch and run `main.py` to execute.

"""

from boilerplate.metrics import get_accuracy, get_accuracy_per_class
from boilerplate.solver import train_network
from boilerplate.losses import cross_entropy
from boilerplate.models import cnn_3_32_32_2layers
from boilerplate.utils import imshow_with_unnormalize, save_net_dict, load_net_from_path, imshow_predictions
from boilerplate.dataloader import get_CIFAR10_datasets
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = './saved_models'


trainloader, testloader, classes = get_CIFAR10_datasets()


"""
    Architecture
"""

net = cnn_3_32_32_2layers.Net().to(device)

"""
    Training
"""
# imshow_with_unnormalize(4, trainloader, classes)

epochs = 2
lr = 1e-2
momentum = 0.9

criterion = cross_entropy()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

train_network(epochs, trainloader, net, criterion, optimizer)
save_net_dict(net, save_path + '/cifar_net.pth')

"""
    Inference
"""
# imshow_with_unnormalize(4, testloader, classes, train = False)

# sets params on 'net'. Use same arch as training
load_net_from_path(net, save_path + '/cifar_net.pth')

# imshow_predictions(4, trainloader, classes, net)

accuracy = get_accuracy(testloader, net)
print(accuracy)

accuracy_per_class = get_accuracy_per_class(testloader, classes, net)
print(accuracy_per_class)
# [('plane', 53.9), ('car', 66.0), ('bird', 39.4), ('cat', 54.7), ('deer', 50.9),
# ('dog', 27.2), ('frog', 56.0), ('horse', 60.1), ('ship', 80.6), ('truck', 66.2)]
