from boilerplate.utils import plot_classes_preds, TBoard


def train_network(epochs, trainloader, net, criterion, optimizer, classes, writer=None):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 2000 mini-batches

                print("[%d, %5d] loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / 2000))

                writer.scalar('training_loss: ', running_loss /
                              1000, epoch * len(trainloader) + i)
                writer.figure('predictions vs. actuals: ', plot_classes_preds(
                    net, inputs, labels, classes), global_step=epoch * len(trainloader) + i)
                #
                running_loss = 0.0
    print("Finished Training")


def infer_network():
    pass
