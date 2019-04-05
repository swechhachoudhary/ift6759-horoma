from comet_ml import OfflineExperiment
import json
import argparse
from models.encoders import *
from models.clustering import *
from utils.utils import *
from utils.utils import load_datasets
from utils.constants import Constants
from data.dataset import HoromaDataset
import torch


def main(datapath, clustering_model, encoding_model, batch_size, n_epochs, lr, flattened, device, train_split, valid_split, train_labeled_split,
         experiment, encode, cluster, train_subset=None, path_to_model=None):
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
    train = HoromaDataset(datapath, split=train_split, subset=train_subset,
                          flattened=flattened)
    labeled = HoromaDataset(datapath, split=train_labeled_split, subset=train_subset,
                            flattened=flattened)
    valid_data = HoromaDataset(
        datapath, split=valid_split, subset=train_subset, flattened=flattened)

    train_label_indices = labeled.targets
    valid_indices = valid_data.targets
    # train_label_indices, valid_indices = get_split_indices(
    #     labeled.targets, overlapped=overlapped)

    print("Shape of training set: ", train.data.shape)
    print("Shape of validation set: ", valid_data.data.shape)
    # print("Shape of labeled training set: ",
    #       labeled.data[train_label_indices].shape)
    # print("Shape of validation dataset: ", labeled.data[valid_indices].shape)

    if encode:
        # Train and apply encoding model
        train_enc, encoding_model = encoding_model.fit(train, valid_data, batch_size=batch_size, n_epochs=n_epochs,
                                                       lr=lr, device=device, experiment=experiment)
    else:
        encoding_model.load_state_dict(torch.load(path_to_model)["model"])
        train_enc = encode_dataset(encoding_model, train, batch_size, device)
    if cluster:
        train_labeled_enc = encoding_model.encode(
            labeled[train_label_indices][0].to(device))
        valid_enc = encoding_model.encode(labeled[valid_indices][0].to(device))

        # Train and apply clustering model
        clustering_model.train(train_enc)
        cluster_labels = assign_labels_to_clusters(clustering_model, train_labeled_enc,
                                                   labeled.targets[train_label_indices])
        _, accuracy, f1 = eval_model_predictions(clustering_model, valid_enc, labeled.targets[valid_indices],
                                                 cluster_labels)
        experiment.log_metric('accuracy', accuracy)
        experiment.log_metric('f1-score', f1)

        # Save models
        model = {'cluster': clustering_model,
                 'embedding': encoding_model, 'cluster_labels': cluster_labels}
        torch.save(model, Constants.PATH_TO_MODEL +
                   str(experiment.get_key()) + '.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=Constants.DATAPATH,
                        help="Path to dataset folder")
    parser.add_argument("--encoder_path", type=str, default=None)

    parser.add_argument("--config", type=str, default="CVAE_BASE",
                        help="To select configuration from config.json")
    args = parser.parse_args()
    config_key = args.config
    datapath = args.datapath
    path_to_model = args.encoder_path
    print(Constants.CONFIG_PATH)
    print(config_key)
    print(Constants.CONFIG_PATH)
    with open(Constants.CONFIG_PATH, "r") as f:
        configuration = json.load(f)[config_key]

    # Parse configuration file
    clustering_model = configuration['cluster_model']
    encoding_model = configuration['enc_model']
    batch_size = configuration['batch_size']
    seed = configuration['seed']
    n_epochs = configuration['n_epochs']
    lr = configuration['lr']
    n_clusters = configuration['n_clusters']
    # train_subset = configuration['train_subset']
    train_split = configuration['train_split']
    valid_split = configuration['valid_split']
    train_labeled_split = configuration['train_labeled_split']
    encode = configuration['encode']
    cluster = configuration['cluster']
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
#     experiment = OfflineExperiment(
#         "z15Um8oxWZwiXQXZxZKGh48cl", workspace='swechhachoudhary', offline_directory="swechhas_experiments")

    # Set up Comet Experiment tracking
    experiment = OfflineExperiment(project_name='general',
                                   workspace='timothynest',  # Replace this with appropriate comet workspace
                                   offline_directory="experiments")
    experiment.set_name(
        name=args.config + "_dim={}_overlapped={}".format(latent_dim, train_split))
    experiment.log_parameters(configuration)

    # Initialize necessary objects
    if clustering_model == 'kmeans':
        clustering_model = KMeansClustering(n_clusters, seed)
    elif clustering_model == 'gmm':
        clustering_model = GMMClustering(n_clusters, seed)
    elif clustering_model =='svm':
        clustering_model = SVMClustering(seed)
    else:
        print('No clustering model specified. Using Kmeans.')
        clustering_model = KMeansClustering(n_clusters, seed)

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
    main(datapath, clustering_model, encoding_model, batch_size, n_epochs, lr, flattened, device, train_split, valid_split, train_labeled_split,
         experiment, encode, cluster, path_to_model=path_to_model)
