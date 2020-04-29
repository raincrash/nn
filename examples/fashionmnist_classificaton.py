
"""

Example code to show FashionMNIST classification.
Install Pytorch and run `main.py` to execute.

"""

from boilerplate.metrics import get_accuracy, get_accuracy_per_class, pr_curve_tensorboard
from boilerplate.solver import train_network
from boilerplate.losses import cross_entropy
from boilerplate.models import cnn_28_28_2layers
from boilerplate.utils import matplotlib_imshow, imshow_with_unnormalize, save_net_dict, load_net_from_path, imshow_predictions, TBoard
from boilerplate.dataloader import FashionMNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = './saved_models'

dataset = FashionMNIST()
trainset, testset, trainloader, testloader, classes = dataset.get_FashionMNIST_datasets()
# imshow_with_unnormalize(4, trainloader, classes)


"""
    Architecture
"""

net = cnn_28_28_2layers.Net().to(device)

"""
    Tensorboard setup - start by `tensorboard --logdir=runs`

"""

tboard = TBoard('runs/fashion_mnist_experiment_1')
tboard.imshow_tensorboard(
    trainloader, name="four_fashion_mnist_images")
tboard.graph(net)
# tboard.embeddings(trainset, testset, classes, 28 * 28)

"""
    Training
"""

# epochs = 2
# lr = 1e-2
# momentum = 0.9

# criterion = cross_entropy()
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# train_network(epochs, trainloader, net, criterion,
#               optimizer, classes, tboard)
# save_net_dict(net, save_path + '/fashion_mnist_net.pth')

"""
    Inference
"""
# imshow_with_unnormalize(4, testloader, classes, train = False)

# # sets params on 'net'. Use same arch as training
load_net_from_path(net, save_path + '/fashion_mnist_net.pth')

# imshow_predictions(4, trainloader, classes, net)

# accuracy = get_accuracy(testloader, net)
# print(accuracy)


# accuracy_per_class = get_accuracy_per_class(testloader, classes, net)
# print(accuracy_per_class)
# [('T-shirt/top', 79.2), ('Trouser', 91.2), ('Pullover', 52.9), ('Dress', 59.1), ('Coat', 70.9),
# ('Sandal', 72.3), ('Shirt', 16.2), ('Sneaker', 89.0), ('Bag', 90.2), ('Ankle Boot', 93.5)]

# pr_curve_tensorboard(testloader, classes, net, tboard)
