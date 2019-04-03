from comet_ml import OfflineExperiment
import json
import argparse
from models.encoders import *
from models.clustering import *
from utils.utils import *
from utils.constants import Constants
import torch
from data.dataset import LocalHoromaDataset
from torch.utils.data import DataLoader

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

    print("Shape of unlabeled training set: ", unlabeled_train.data.shape)
    print("Shape of labeled training set: ", labeled_train.data.shape)
    print("Shape of labeled valid set: ", labeled_valid.data.shape)
    
    damic_model = DAMICClustering(17).to(device)

    # ===== START PRE-TRAINING AS SHOWN IN DAMIC PAPER
    print("== Start DAMIC pre-training ...")
    # Train a single autoencoder for the entire dataset
    if train_encoder:
        # Train and apply encoding model
        print("Start training of pre-train auto-encoder...")
        encoded_unlabeled_train, encoding_model = encoding_model.fit(data=unlabeled_train, batch_size=batch_size, n_epochs=n_epochs,
                                                       lr=lr, device=device, experiment=experiment)
        print("Done")
    else:
        # Load encoding model and apply encoding
        print("Load pre-train auto-encoder...")
        encoding_model.load_state_dict(torch.load(path_to_model)["model"])
        print("Done")
        print("Start encoding of unlabeled dataset...")
        encoded_unlabeled_train = encode_dataset_unlabeled(encoding_model, unlabeled_train, batch_size, device)
        print("Done")
    
    # Apply a k-means algorithm in the embedded space
    print("Start encoding of labeled dataset...")
    encoded_labeled_train = encode_dataset_unlabeled(encoding_model, labeled_train, batch_size, device)
    print("Done")
    print("Start encoding of labeled dataset...")
    encoded_labeled_valid = encode_dataset_unlabeled(encoding_model, labeled_valid, batch_size, device)
    print("Done")
    print("Start kmean training on unlabeled...")
    clustering_model.train(encoded_unlabeled_train)
    print("Done")
    cluster_labels = assign_labels_to_clusters(clustering_model, encoded_labeled_train, labeled_train.targets)
    _, accuracy, f1 = eval_model_predictions(clustering_model, encoded_labeled_valid, labeled_valid.targets, cluster_labels)
    experiment.log_metric('accuracy', accuracy)
    experiment.log_metric('f1-score', f1)

    # For each unlabeled data, we get the class predicted by the cluster
    unlabeled_target_pred_by_cluster = clustering_model.predict_cluster(encoded_unlabeled_train)
    
    # Use the k-means clustering to initialize the network parameters
    tensor_unla_train = torch.Tensor(unlabeled_train.data)
    tensor_unla_target_pred_by_cluster = torch.Tensor(unlabeled_target_pred_by_cluster)
    numpy_unla_train = tensor_unla_train.cpu().numpy()
    numpy_unla_target_pred_by_cluster = tensor_unla_target_pred_by_cluster.cpu().numpy()

    print("Start pre-training of clustering convolutional model...")
    pretrain_dataset_with_label = LocalHoromaDataset(numpy_unla_train, numpy_unla_target_pred_by_cluster)
    clust_network_params = list(damic_model.clustering_network.parameters()) + list(damic_model.output_layer_conv_net.parameters())
    optimizer = torch.optim.Adam(clust_network_params, lr=lr)
    damic_model = train_network(damic_model,
                                pretrain_dataset_with_label,
                                None,
                               optimizer,
                               10,
                               device,
                               experiment,
                               train_classifier=True)
    print("Done")

    # Train each auto encoder of each cluster on his own data class
    for i in range(17):
        print("Start pre-training of auto encoder for cluster ...")
        indexes_of_class_i = np.where(np.isin(numpy_unla_target_pred_by_cluster,[i]))
        data_of_class_i = numpy_unla_train[indexes_of_class_i]
        _, damic_model.autoencoders[i] = damic_model.autoencoders[i].fit(data=data_of_class_i, batch_size=batch_size, n_epochs=n_epochs,
                                                       lr=lr, device=device, experiment=experiment)
        print("Done")
  
    print("== DAMIC Pre-training done!")
    # ===== END OF PRE-TRAINING
    
    # ===== BEGIN REAL TRAINING
    print("== Start DAMIC training ...!")
    # TODO : get all the new labels predicted after the training of the clustering conv. network and use it to initialize LocalHoromaData
    # Put all the data through the conv network
    # TODO TODO : in numpy_unla_target_pred_by_cluster replace all value inf to 0 by 0 and sup to 16 by 16
    pretrain_dataset_with_label = LocalHoromaDataset(numpy_unla_train, numpy_unla_target_pred_by_cluster)
    clust_network_params = list(damic_model.clustering_network.parameters()) + list(damic_model.output_layer_conv_net.parameters())
    autoencoders_params = list()
    for i in range(17):
        autoencoders_params = autoencoders_params + list(damic_model.autoencoders[i].parameters())
    damic_parameters = clust_network_params + autoencoders_params
    optimizer = torch.optim.Adam(damic_parameters, lr=lr)
    damic_model = train_network(damic_model,
                               pretrain_dataset_with_label,
                               None,
                               optimizer,
                               1,
                               device,
                               experiment,
                               train_classifier=False,
                               train_damic=True)


    print("== DAMIC training done!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=Constants.DATAPATH,
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
