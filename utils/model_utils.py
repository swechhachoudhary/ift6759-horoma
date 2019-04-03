import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import torch


def get_ae_dataloaders(traindata, batch_size, split):
    """get dataloaders for train and valid sets"""
    indices = list(range(len(traindata)))
    np.random.shuffle(indices)
    n_train = int(split * len(indices))
    train_loader = DataLoader(traindata, batch_size=batch_size, sampler=SubsetRandomSampler(indices[:n_train]))
    valid_loader = DataLoader(traindata, batch_size=batch_size, sampler=SubsetRandomSampler(indices[n_train:]))

    return train_loader, valid_loader


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x.view(-1, 32 * 32 * 3), x.view(-1, 32 * 32 * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def encode_dataset(model, data, batch_size, device):
    full_loader = DataLoader(data, batch_size=batch_size)
    model.eval()
    tensors = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(full_loader):
            inputs = inputs.to(device)
            tensors.append(model.encode(inputs))
    return torch.cat(tensors, dim=0)

def encode_dataset_unlabeled(model, data, batch_size, device):
    full_loader = DataLoader(data, batch_size=batch_size)
    model.eval()
    tensors = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(full_loader):
            inputs = inputs[0].to(device)
            tensors.append(model.encode(inputs))
    return torch.cat(tensors, dim=0)


def _train_one_epoch(model, train_loader, optimizer, epoch, device, experiment):
    """Train one epoch for model."""
    model.train()

    running_loss = 0.0

    for batch_idx, inputs in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        if model.is_variational:
            pred, mu, logvar = model(inputs)
            if model.calculate_own_loss:
                loss = mu
            else:
                loss = loss_function(pred, inputs, mu, logvar)
        else:
            outputs = model(inputs)
            criterion = nn.MSELoss(reduction='sum')
            loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(inputs),
                                                                           len(train_loader) * len(inputs),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item() / len(inputs)))

    train_loss = running_loss / len(train_loader.dataset)
    experiment.log_metric("Train loss", train_loss, step=epoch)
    return train_loss

def _train_one_epoch_classifier(model, train_loader, optimizer, epoch, device, experiment):
    """Train one epoch for model."""
    model.clustering_network.train()
    model.output_layer_conv_net.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.long()
        if labels[0] > -1 and labels[0] < 17:
            #print("Labels labels")
            #print(labels)
            #print("Labels labels shape")
            #print(labels.shape)
            #label_vector = torch.LongTensor(17).zero_()
            #label_vector[labels[0].item()] = 1
            #label_vector = label_vector.to(device)
            #label_vector = label_vector.unsqueeze(0)
            #print("Vector labels")
            #print(label_vector)
            #print("Vector labels shape")
            #print(label_vector.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #print("inputs shape before")
            #print(inputs.shape)
            inputs = inputs.unsqueeze(0)
            #print("inputs shape after")
            #print(inputs.shape)
            outputs = model.train_clustering_network(inputs)
            #print("outputs")
            #print(outputs)
            #print("SHAPE")
            #print(outputs.shape)
            criterion = nn.CrossEntropyLoss()
            #print("First row labels")
            #print(labels[0])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            """
            if batch_idx % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 10))
            """

    train_loss = running_loss / len(train_loader)
    experiment.log_metric("Conv classifier train loss", train_loss, step=epoch)
    print('Finished Training')
    return train_loss

def _train_one_epoch_damic(model, train_loader, optimizer, epoch, device, experiment):
    """Train one epoch for model."""
    model.clustering_network.train()
    model.output_layer_conv_net.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.long()
        if labels[0] > -1 and labels[0] < 17:
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(0)

            outputs, ae_reconstruction = model.train_damic(inputs)
            #criterion_conv_net = nn.CrossEntropyLoss()
            criterion_ae = nn.MSELoss()

            #loss_conv_net = criterion_conv_net(outputs, labels)
            #print("LOSS CONV NET")
            #print(loss_conv_net)
            loss_autoencoders = torch.FloatTensor(17).zero_().to(device)
            for i in range(17):
                loss_autoencoder = criterion_ae(inputs, ae_reconstruction[i].to(device))
                #print("LOSS AUTO ENCODERS")
                #print(loss_autoencoder)
                loss_autoencoders[i] = -(loss_autoencoder/2)
                #loss_autoencoders = np.append(loss_autoencoders, -(loss_autoencoder/2))
            #print("FINAL LOSS AE")
            #print(loss_autoencoders)
            exp_loss_autoencoders = loss_autoencoders.exp()
            #print("EXP AE")
            #print(exp_loss_autoencoders)
            total_loss_per_class = outputs * loss_autoencoders
            #total_loss_per_class = loss_conv_net * loss_autoencoders
            #print("Total loss per class")
            #print(total_loss_per_class)
            total_loss = total_loss_per_class.sum()
            #print("Total loss sum")
            #print(total_loss)
            total_loss_log = total_loss.log()
            #print("Total loss log result")
            #print(total_loss_log)
            # Simultaneously train all the autoencoders and the convolutional network
            total_loss_log.backward()
            optimizer.step()

            # print statistics
            running_loss += total_loss_log.item()
            """
            if batch_idx % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 1))
            """

    train_loss = running_loss / len(train_loader)
    experiment.log_metric("DAMIC train loss", train_loss, step=epoch)
    return train_loss

def _test(model, test_loader, epoch, device, experiment):
    """ Compute reconstruction loss of model over given dataset. """
    model.eval()

    test_loss = 0
    test_size = 0

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            if model.is_variational:
                output, mu, logvar = model(inputs)
                if model.calculate_own_loss:
                    test_loss += mu
                else:
                    test_loss += loss_function(output, inputs, mu, logvar).item()
                test_size += len(inputs)
            else:
                output = model(inputs)
                criterion = nn.MSELoss(reduction='sum')
                test_loss += criterion(output, inputs).item()
                test_size += len(inputs)

    test_loss /= test_size
    experiment.log_metric("Validation loss", test_loss, step=epoch)
    return test_loss

def train_network(model, train_loader, test_loader, optimizer, n_epochs, device, experiment, train_classifier=False, train_damic=False):
    best_loss = np.inf
    key = experiment.get_key()
    best_model = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    for epoch in range(n_epochs):
        scheduler.step()
        if train_damic:
            train_loss = _train_one_epoch_damic(model, train_loader, optimizer, epoch, device, experiment)
            print("train loss is")
            print(train_loss)
        elif train_classifier:
            train_loss = _train_one_epoch_classifier(model, train_loader, optimizer, epoch, device, experiment)
        else:
            train_loss = _train_one_epoch(model, train_loader, optimizer, epoch, device, experiment)
        if test_loader != None:
            valid_loss = _test(model, test_loader, epoch, device, experiment)
        else:
            valid_loss = -1
        try:
            if valid_loss < best_loss:
                torch.save({
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "model": model.state_dict(),
                    "loss": valid_loss
                }, "experiment_models/" + str(key) + '.pth')
                best_loss = valid_loss
                best_model = deepcopy(model)  # Keep best model thus far
        except FileNotFoundError as e:
            print("Directory for logging experiments does not exist. Launch script from repository root.")
            raise e

        print("Training loss after {} epochs: {:.6f}".format(epoch, train_loss))
        print("Validation loss after {} epochs: {:.6f}".format(epoch, valid_loss))

    # Return best model
    return best_model
