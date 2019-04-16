from comet_ml import OfflineExperiment
import json
import argparse
import torch
import os
import sys

sys.path.append("../")
from models import *
from models.clustering import *
from utils.ali_utils import *
from utils.utils import *
from utils.utils import load_datasets
from utils.constants import Constants
from utils.dataset import HoromaDataset


def main(datapath, clustering_model, encoding_model, configs, train_split, valid_split, train_labeled_split,
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
        if encoding_model == "hali":
            Gx1, Gx2, Gz1, Gz2, Disc, z_pred1, z_pred2, optim_g, optim_d, train_loader, cuda, configs = initialize_hali(
                configs, train)
            training_loop_hali(Gz1, Gz2, Gx1, Gx2, Disc, optim_d, optim_g, train_loader,
                               configs, experiment, cuda, z_pred1, z_pred2)
        else:  # default to ALI
            Gx, Gz, Disc, z_pred, optim_g, optim_d, train_loader, cuda, configs = initialize_ali(configs, train)
            training_loop_ali(Gz, Gx, Disc, optim_d, optim_g, train_loader, configs, experiment, cuda, z_pred)

    if cluster:

        if encoding_model == "hali":

            best_f1, best_acc, best_model = get_results_hali(configs, experiment, train, labeled, valid_data)

        else:

            best_f1, best_acc, best_model = get_results_ali(configs, experiment, train, labeled, valid_data)

        experiment.log_metric('best_accuracy', best_acc)
        experiment.log_metric('best_f1-score', best_f1)
        experiment.log_metric('best_model_epoch', best_model)

    return best_acc, best_f1, best_model


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
    clustering_model = configuration['cluster_model']
    encoding_model = configuration['enc_model']
    batch_size = configuration['batch_size']
    seed = configuration['seed']
    n_epochs = configuration['n_epochs']
    # train_subset = configuration['train_subset']
    train_split = configuration['train_split']
    valid_split = configuration['valid_split']
    train_labeled_split = configuration['train_labeled_split']
    encode = False
    cluster = True
    flattened = False  # Default
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set all seeds for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    latent_dim = configuration['Zdim']
    if not os.path.exists('experiments'):
        print('mkdir ', 'experiments')
        os.mkdir('experiments')

    exp_best_acc = 0
    exp_best_f1 = 0
    exp_best_model = 0
    best_experiment = ''

    exp_list = get_experiments()
    for exp in exp_list:

        configuration['experiment'] = exp

        experiment = OfflineExperiment(project_name="ali", workspace='timothynest',  # Replace this with appropriate comet workspace
                                       offline_directory=str('experiments/' + configuration['experiment'] + '/cluster'))
        experiment.set_name(
            name=args.config + "_dim={}_overlapped={}".format(latent_dim, train_split))
        experiment.log_parameters(configuration)
        experiment.add_tag(configuration['experiment'])

        # Initiate experiment
        best_acc, best_f1, best_model = main(datapath, clustering_model, encoding_model, configuration, train_split, valid_split, train_labeled_split,
                                             experiment, encode, cluster, path_to_model=path_to_model)

        if best_f1 > exp_best_f1:
            exp_best_f1 = best_f1
            exp_best_acc = best_acc
            exp_best_model = best_model
            best_experiment = exp

        print(('("Best_acc: %f. Best F1: %f, Best_model: %f, Exp Best f1: %f, Exp Best Model: %d' %
               (best_acc, best_f1, best_model, exp_best_f1, exp_best_model)))
        print('best_experiment:', best_experiment)
        configuration['continue_from'] = 1
