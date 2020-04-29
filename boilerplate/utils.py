import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F


"""
    model utils
"""


def save_net_dict(net, path):
    torch.save(net.state_dict(), path)


def load_net_from_path(net, path):
    net.load_state_dict(torch.load(path))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


"""
    vis utils
"""


def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


class TBoard:
    def __init__(self, name):
        self.writer = SummaryWriter(name)

    def get_writer(self):
        return self.writer

    def scalar(self, name, scalar_value, global_step):
        self.writer.add_scalar(name, scalar_value, global_step)

    def figure(self, name, figure, global_step):
        self.writer.add_figure(name, figure, global_step)

    def select_n_random(self, data, labels, n=100):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]

    def embeddings(self, trainset, testset, classes, size):
        # select random images and their target indices
        images, labels = self.select_n_random(trainset.data, trainset.targets)

        # get the class labels for each image
        class_labels = [classes[lab] for lab in labels]

        # log embeddings
        features = images.view(-1, size)
        self.writer.add_embedding(
            features, metadata=class_labels, label_img=images.unsqueeze(1), global_step=0)
        self.writer.close()

    def imshow_tensorboard(self, loader, name="New Image", one_channel=True):
        self.dataiter = iter(loader)
        self.images, self.labels = self.dataiter.next()
        img_grid = torchvision.utils.make_grid(self.images)
        matplotlib_imshow(img_grid, one_channel=True)

        self.writer.add_image(name, img_grid)

    def graph(self, net):
        self.writer.add_graph(net, self.images)

    def add_pr_curve_tensorboard(self, class_index, test_probs, test_preds, classes, global_step=0):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        self.writer.add_pr_curve(classes[class_index],
                                 tensorboard_preds,
                                 tensorboard_probs,
                                 global_step=global_step)
        self.writer.close()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def imshow_with_unnormalize(number, loader, classes, train=True, one_channel=False):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)

    # TODO: Call matpoltlib_imshow here

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),)
    if train:
        plt.title(" ".join("%5s" % classes[labels[j]] for j in range(number)))
    else:
        plt.title(
            "GroundTruth: "
            + " ".join("%5s" % classes[labels[j]] for j in range(number))
        )
    plt.show()


def imshow_predictions(number, loader, classes, net):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),)
    plt.title(
        "Predicted: " + " ".join("%5s" %
                                 classes[predicted[j]] for j in range(number))
    )
    plt.show()
