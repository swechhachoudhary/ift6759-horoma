from comet_ml import OfflineExperiment
import json
import argparse
from models.encoders import *
from models.clustering import *
from utils.utils import *
from utils.damic_utils import *
from utils.constants import Constants
import torch
from data.dataset import LocalHoromaDataset

def main(datapath, encoding_model, clustering_model, batch_size, n_epochs, lr, flattened, device, overlapped,
         experiment, train_encoder, train_subset=None, path_to_model=None):
    """
    :param datapath: path to the directory containing the samples
    :param encoding_model: which encoding model to use for the pre-training of DAMIC
    :param clustering_model: clustering model used for the pre-training of DAMIC
    :param batch_size: batch size
    :param n_epochs: number of epochs
    :param lr: learning rate
    :param flattened: If True return the images in a flatten format.
    :param device: use CUDA device if available else CPU .
    :param overlapped: boolean, if True use the overlapped pixel patches.
    :param experiment: track experiment
    :param train_encoder: if True, an auto-encoder is trained on the unlabeled data for DAMIC pre-training
    :param train_subset: whether of not we use a subset for the dataset specified in datapath
    :param path_to_model: if train_encoder is False, then we load an autoencoder from there
    """   
    damic_model = DAMICClustering(17).to(device)
    damic_model, numpy_unla_train, numpy_unla_target_pred_by_cluster = execute_damic_pre_training(datapath, damic_model, encoding_model, 
                                                                                                  train_encoder, clustering_model,
                                                                                                  batch_size, n_epochs, lr, flattened,
                                                                                                  device, overlapped, experiment,
                                                                                                  train_subset, path_to_model)

    print("== Start DAMIC training ...!")
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
    torch.save(damic_model, Constants.PATH_TO_MODEL + "DAMIC_MODEL" + str(experiment.get_key()) + '.pth')

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
    main(datapath, encoding_model, clustering_model, batch_size, n_epochs, lr, flattened, device, overlapped,
         experiment, train_encoder, train_subset, path_to_model)
