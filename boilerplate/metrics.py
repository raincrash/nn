import torch
import torch.nn as nn


def get_accuracy(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total)


def get_accuracy_per_class(testloader, classes, net, number=4):
    no_of_classes = len(classes)
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    class_accuracies = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(number):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(len(classes)):
        class_predictions = (
            classes[i], 100 * class_correct[i] / class_total[i])
        class_accuracies.append(class_predictions)
    return class_accuracies
