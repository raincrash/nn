import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

"""
    model utils
"""


def save_net_dict(net, path):
    torch.save(net.state_dict(), path)


def load_net_from_path(net, path):
    net.load_state_dict(torch.load(path))


"""
    vis utils
"""


def imshow_with_unnormalize(number, loader, classes, train=True):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
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
