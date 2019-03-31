import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import torch


def get_ae_dataloaders(traindata, valid_data, batch_size=32):
    """get dataloaders for train and valid sets"""
    # indices = list(range(len(traindata)))
    # np.random.shuffle(indices)
    # n_train = int(split * len(indices))
    # train_loader = DataLoader(traindata, batch_size=batch_size, sampler=SubsetRandomSampler(indices[:n_train]))
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    return train_loader, valid_loader


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x.view(-1, 32 * 32 * 3),
                     x.view(-1, 32 * 32 * 3), reduction='sum')

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


def _train_one_epoch(model, train_loader, optimizer, epoch, device, experiment):
    """Train one epoch for model."""
    model.train()

    running_loss = 0.0

    for batch_idx, inputs in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        if model.is_variational:
            pred, mu, logvar = model(inputs)
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
                                                                           len(train_loader) *
                                                                           len(inputs),
                                                                           100. * batch_idx /
                                                                           len(train_loader),
                                                                           loss.item() / len(inputs)))

    train_loss = running_loss / len(train_loader.dataset)
    experiment.log_metric("Train loss", train_loss, step=epoch)
    return train_loss


def _test(model, test_loader, epoch, device, experiment):
    """ Compute reconstruction loss of model over given dataset. """
    model.eval()

    test_loss = 0
    test_size = 0

    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            if model.is_variational:
                output, mu, logvar = model(inputs)
                test_loss += loss_function(output, inputs, mu, logvar).item()
                test_size += len(inputs)
            else:
                output = model(inputs)
                criterion = nn.MSELoss(reduction='sum')
                test_loss += criterion(output, inputs).item()
                # test_size += len(inputs)

    test_loss /= len(test_loader.dataset)
    experiment.log_metric("Validation loss", test_loss, step=epoch)
    return test_loss


def train_network(model, train_loader, test_loader, optimizer, n_epochs, device, experiment):
    best_loss = np.inf
    key = experiment.get_key()
    best_model = None
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.5)
    for epoch in range(n_epochs):
        scheduler.step()
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, epoch, device, experiment)
        valid_loss = _test(model, test_loader, epoch, device, experiment)

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
            print(
                "Directory for logging experiments does not exist. Launch script from repository root.")
            raise e

        print("Training loss after {} epochs: {:.6f}".format(epoch, train_loss))
        print("Validation loss after {} epochs: {:.6f}".format(epoch, valid_loss))

    # Return best model
    return best_model
