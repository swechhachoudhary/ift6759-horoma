from comet_ml import OfflineExperiment
import os
import sys
import json
import torch
import argparse
import numpy as np
from time import time
from torch.utils import data


sys.path.append("../")
from models.encoders import *
from models.svm_classifier import SVMClassifier
from utils.model_utils import encode_dataset
from utils.utils import compute_metrics, __compute_metrics
from utils.constants import Constants
from utils.dataset import HoromaDataset


def main(datapath, encoding_model, batch_size, n_epochs, lr, device, train_split, valid_split, train_labeled_split,
         experiment, path_to_model=None):

    full_dataset = HoromaDataset(datapath, split=train_split, flattened=flattened)
    train_labeled = HoromaDataset(
        datapath, split=train_labeled_split, flattened=flattened)
    # Validation data(labeled) for the supervised task(Classification)
    valid_data = HoromaDataset(
        datapath, split=valid_split, flattened=flattened)

    # split the full_dataset(labeled and unlabeled train data) into train and valid for autoencoder pre-training
    n_train = int(0.90 * len(full_dataset))
    n_valid = len(full_dataset) - n_train
    train_dataset, valid_dataset = data.random_split(full_dataset, [n_train, n_valid])

    # Train and apply encoding model
    train_enc, encoding_model = encoding_model.fit(train_dataset, valid_dataset, batch_size=batch_size, n_epochs=n_epochs,
                                                   lr=lr, device=device, experiment=experiment)

    # extract latent representation of train_labeled data
    train_labeled_enc = encode_dataset(
        encoding_model, train_labeled, batch_size, device, is_unlabeled=False)
    print("Train labeled data encoding complete.\n")

    # extract latent representation of validation data
    valid_enc = encode_dataset(
        encoding_model, valid_data, batch_size, device, is_unlabeled=False)
    print("validation data encoding complete.\n")

    start_time = time()

    # Train SVM classifier
    svm_classifier = SVMClassifier()
    print("Traing SVM classifier...\n")
    pred_train_y = svm_classifier.train_classifier(
        train_labeled_enc, train_labeled.targets)

    print("Computing metrics for train data\n")
    train_accuracy, train_f1, __train_f1 = __compute_metrics(
        train_labeled.targets, pred_train_y)

    print("Prediction for validation data. \n")
    pred_valid_y = svm_classifier.validate_classifier(
        valid_enc)

    print("Computing metrics for validation data\n")
    valid_accuracy, valid_f1, __valid_f1 = __compute_metrics(
        valid_data.targets, pred_valid_y)

    print("Done in {:.2f} sec.".format(time() - start_time))
    print(
        "Train : Accuracy: {:.2f} | F1: {:.2f}".format(train_accuracy * 100, train_f1 * 100))
    print(
        "Train : F1 score for each class: {}".format(__train_f1 * 100))
    print(
        "Validation : Accuracy: {:.2f} | F1: {:.2f}".format(valid_accuracy * 100, valid_f1 * 100))
    print(
        "Validation : F1 score for each class: {}".format(__valid_f1 * 100))

    experiment.log_metric('Train accuracy', train_accuracy)
    experiment.log_metric('Train f1-score', train_f1)
    experiment.log_metric('Validation accuracy', valid_accuracy)
    experiment.log_metric('Validation f1-score', valid_f1)

    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(train_labeled.targets, pred_train_y, classes=np.arange(17),
    #                       title='Confusion matrix for Train, without normalization')
    # validation data
    # plot_confusion_matrix(valid_data.targets, pred_valid_y, classes=np.arange(17),
    #                       title='Confusion matrix for Validation, without normalization')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=Constants.DATAPATH,
                        help="Path to dataset folder")
    parser.add_argument("--encoder_path", type=str, default=None)

    parser.add_argument("--config", type=str, default="CAE_BASE",
                        help="To select configuration from config.json")
    args = parser.parse_args()
    config_key = args.config
    datapath = args.datapath
    path_to_model = args.encoder_path

    with open(Constants.CONFIG_PATH, 'r') as f:
        configuration = json.load(f)[config_key]

    # Parse configuration file
    encoding_model = configuration['enc_model']
    batch_size = configuration['batch_size']
    seed = configuration['seed']
    n_epochs = configuration['n_epochs']
    lr = configuration['lr']
    train_split = configuration['train_split']
    valid_split = configuration['valid_split']
    train_labeled_split = configuration['train_labeled_split']
    latent_dim = configuration['latent_dim']
    flattened = False  # Default
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set all seeds for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set up Comet Experiment tracking  # Replace this with appropriate comet
    # workspaces
    experiment = OfflineExperiment(
        "z15Um8oxWZwiXQXZxZKGh48cl", workspace='swechhachoudhary', offline_directory="../swechhas_experiments")

    # Set up Comet Experiment tracking
    # experiment = OfflineExperiment(project_name='general',
    #                                workspace='benjaminbenoit',  # Replace this with appropriate comet workspace
    #                                offline_directory="experiments")
    experiment.set_name(
        name=args.config + "_dim={}_overlapped={}".format(latent_dim, train_split))
    experiment.log_parameters(configuration)

    if encoding_model == 'pca':
        encoding_model = PCAEncoder(seed)
        flattened = True
    elif encoding_model == 'vae':
        encoding_model = VAE(latent_dim=latent_dim).to(device)
        flattened = True
    elif encoding_model == "ae":
        encoding_model = AE(latent_dim=latent_dim).to(device)
        flattened = True
    elif encoding_model == "cae":
        encoding_model = CAE(latent_dim=latent_dim).to(device)
        flattened = False
    elif encoding_model == "cvae":
        encoding_model = CVAE(latent_dim=latent_dim).to(device)
        flattened = False
    elif encoding_model == "convae":
        encoding_model = ConvAE(latent_dim=latent_dim).to(device)
        flattened = False
    else:
        print('No encoding model specified. Using PCA.')
        encoding_model = PCAEncoder(seed)

    # Initiate experiment
    main(datapath, encoding_model, batch_size, n_epochs, lr, device, train_split, valid_split, train_labeled_split,
         experiment, path_to_model=path_to_model)
