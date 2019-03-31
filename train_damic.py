from comet_ml import OfflineExperiment
import json
import argparse
from models.encoders import *
from models.clustering import *
from utils.utils import *
from utils.constants import Constants
import torch


def main(datapath, encoding_model, batch_size, n_epochs, lr, flattened, device, overlapped,
         experiment, train_encoder, train_subset=None, path_to_model=None):
    """
    :param datapath: path to the directory containing the samples
    :param clustering_model: which clustering model to use [kmeans, gmm].
    :param encoding_model: which encoding model to use, convolutional, variational or simple autoencoders.
    :param batch_size: batch size
    :param n_epochs: number of epochs
    :param lr: learning rate
    :param flattened: If True return the images in a flatten format.
    :param device: use CUDA device if available else CPU .
    :param overlapped: boolean, if True use the overlapped pixel patches.
    :param experiment: track experiment
    :param encode: boolean, if True, train and apply encoding model.
    :param cluster: boolean, if True, train and apply clustering model.
    :param train_subset: How many elements will be used. Default: all.
    :param path_to_model: path to the directory containing saved models.

    """
    unlabeled_train, labeled_train, labeled_valid = load_datasets2(datapath, train_subset=train_subset,
                                   flattened=flattened, overlapped=overlapped)
    train_label_indices, valid_indices = get_split_indices(labeled.targets, overlapped=overlapped)

    print("Shape of unlabeled training set: ", unlabeled_train.data.shape)
    print("Shape of labeled training set: ", labeled_train.data.shape)
    print("Shape of labeled valid set: ", labeled_valid.data.shape)

    # ===== START PRE-TRAINING AS SHOWN IN DAMIC PAPER
    # Train a single autoencoder for the entire dataset
    if train_encoder:
        # Train and apply encoding model
        encoded_unlabeled_train, encoding_model = encoding_model.fit(data=unlabeled_train, batch_size=batch_size, n_epochs=n_epochs,
                                                       lr=lr, device=device, experiment=experiment)
    else:
        # Load encoding model and apply encoding
        encoding_model.load_state_dict(torch.load(path_to_model)["model"])
        encoded_unlabeled_train = encode_dataset(encoding_model, unlabeled_train, batch_size, device)
    
    # Apply a k-means algorithm in the embedded space
    encoded_labeled_train = encoding_model.encode(labeled_train[0].to(device))
    encoded_labeled_valid = encoding_model.encode(labeled_valid[0].to(device))
    clustering_model.train(encoded_unlabeled_train)
    cluster_labels = assign_labels_to_clusters(clustering_model, encoded_labeled_train, labeled_train.targets)
    _, accuracy, f1 = eval_model_predictions(clustering_model, encoded_labeled_valid, labeled_valid.targets, cluster_labels)
    experiment.log_metric('accuracy', accuracy)
    experiment.log_metric('f1-score', f1)

    # Use the k-means clustering to initialize the network parameters
    
    
    # ===== END OF PRE-TRAINING
    
    # ===== BEGIN REAL TRAINING
    # TODO


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--", type=str, default=Constants.DATAPATH,
                        help="Path to dataset folder")
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--config", type=str, default="CONV_AE", help="To select configuration from config.json")
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
    n_clusters = configuration['n_clusters']
    train_subset = configuration['train_subset']
    overlapped = configuration['overlapped']
    # If true, we train an autoencoder and do a kmean clustering to pretrain the clustering network
    train_encoder = configuration['train_encoder']
    latent_dim = configuration['latent_dim']
    flattened = False  # Default
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set all seeds for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set up Comet Experiment tracking
    experiment = OfflineExperiment(project_name='general',
                                   workspace='benjaminbenoit',
                                   offline_directory="experiments")
    experiment.set_name(
        name=args.config + "_dim={}_cluster={}_overlapped={}".format(latent_dim, n_clusters, overlapped))
    experiment.log_parameters(configuration)

    # Initialize necessary objects
    clustering_model = KMeansClustering(n_clusters, seed)

    if encoding_model == "cvae":
        encoding_model = CVAE(latent_dim=latent_dim).to(device)
        flattened = False
    elif encoding_model == "convae":
        encoding_model = ConvAE(latent_dim=latent_dim).to(device)
        flattened = False
    else:
        print('No encoding model specified. Using CVAE.')
        encoding_model = CVAE(seed)

    # Initiate experiment
    main(datapath, encoding_model, batch_size, n_epochs, lr, flattened, device, overlapped,
         experiment, train_encoder, train_subset, path_to_model)
