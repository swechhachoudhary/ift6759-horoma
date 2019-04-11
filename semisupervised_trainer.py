from comet_ml import OfflineExperiment
import json
import argparse
import torch
from torch.utils.data import DataLoader
from models.encoders import *
from models.mlp_classifier import MLPClassifier
from utils.constants import Constants
from data.dataset import HoromaDataset
from utils.model_utils import train_semi_supervised_network


def main(datapath, encoding_model, classifier_model, batch_size, n_epochs, lr_unsup, lr_sup, device,
         train_unlabeled_split, valid_split, train_labeled_split, patience,
         experiment, path_to_model=None):
    """
    :param datapath: path to the directory containing the samples
    :param classifier_model: which classifier model to use
    :param encoding_model: which encoding model to use, convolutional, variational or simple autoencoders.
    :param batch_size: batch size
    :param n_epochs: number of epochs
    :param lr_unsup: learning rate for unsupervised part
    :param lr_sup: learning rate for supervised part
    :param device: use CUDA device if available else CPU .
    :param experiment: track experiment
    :param path_to_model: path to the directory containing saved models.
    """
    train_unlabeled = HoromaDataset(datapath, split=train_unlabeled_split)
    train_labeled = HoromaDataset(datapath, split=train_labeled_split)
    valid_data = HoromaDataset(datapath, split=valid_split)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    n_labeled_batch = len(train_labeled) // batch_size
    n_unlabeled_batch = n_labeled_batch

    train_semi_supervised_network(encoding_model, classifier_model, train_unlabeled, train_labeled, valid_loader,
                                  n_epochs, batch_size, lr_unsup, lr_sup, device, n_labeled_batch, n_unlabeled_batch,
                                  patience, experiment)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=Constants.DATAPATH,
                        help="Path to dataset folder")
    parser.add_argument("--encoder_path", type=str, default='results/best_model1.pth')

    parser.add_argument("--config", type=str, default="CAE_MLP",
                        help="To select configuration from config.json")
    args = parser.parse_args()
    config_key = args.config
    datapath = args.datapath
    path_to_model = args.encoder_path

    with open(Constants.CONFIG_PATH, 'r') as f:
        configuration = json.load(f)[config_key]

    encoding_model = configuration['enc_model']
    classifier_model = configuration["classifier_model"]
    batch_size = configuration['batch_size']
    seed = configuration['seed']
    n_epochs = configuration['n_epochs']
    lr_unsup = configuration['lr_unsup']
    lr_sup = configuration['lr_sup']
    patience = configuration['patience']
    train_unlabeled_split = configuration['train_unlabeled_split']
    valid_split = configuration['valid_split']
    train_labeled_split = configuration['train_labeled_split']
    latent_dim = configuration['latent_dim']
    hidden_size = configuration['hidden_size']
    n_layers = configuration['n_layers']
    flattened = False
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
        "z15Um8oxWZwiXQXZxZKGh48cl", workspace='swechhachoudhary', offline_directory="swechhas_experiments")

    experiment.set_name(
        name=args.config + "_dim={}_split={}".format(latent_dim, train_unlabeled_split))
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

    if classifier_model == "MLPClassifier":
        classifier_model = MLPClassifier(latent_dim=latent_dim, hidden_size=hidden_size, n_layers=n_layers).to(device)

    # print("Loading model....\n")
    # # # load the best model
    # x = torch.load(path_to_model, map_location=device)
    # print(x["best_acc"], x["best_f1"])
    # print(x["train_acc"], x["train_f1"])

    # Initiate experiment
    main(datapath, encoding_model, classifier_model, batch_size, n_epochs, lr_unsup, lr_sup, device,
         train_unlabeled_split, valid_split, train_labeled_split, patience,
         experiment, path_to_model=path_to_model)
