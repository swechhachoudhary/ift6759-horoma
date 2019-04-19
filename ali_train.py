from comet_ml import OfflineExperiment
import json
import argparse
from models import *
from models.clustering import *
from utils.ali_utils import *
from utils.utils import *
from utils.utils import load_datasets
from utils.constants import Constants
from data.dataset import HoromaDataset
import torch


def main(datapath, configs, experiment):
    """
    :param datapath: path to the directory containing the samples
    :param configs: dictionary containing hyperparameters for training.
    :param experiment: comet ml experiment object for logging results
    """

    train_split = configs['train_split']
    valid_split = configs['valid_split']
    train_labeled_split = configs['train_labeled_split']

    train = HoromaDataset(datapath, split=train_split, subset=None,
                          flattened=False)
    labeled = HoromaDataset(datapath, split=train_labeled_split, subset=None,
                            flattened=False)
    valid_data = HoromaDataset(
        datapath, split=valid_split, subset=None, flattened=False)

    print("Shape of training set: ", train.data.shape)
    print("Shape of validation set: ", valid_data.data.shape)

    if configs['encode']:
        if configs['enc_model'] == "hali":
            Gx1, Gx2, Gz1, Gz2, Disc, optim_g, optim_d, train_loader, cuda, configs = initialize_hali(
                configs, train)
            training_loop_hali(Gz1, Gz2, Gx1, Gx2, Disc, optim_d,
                               optim_g, train_loader, configs, experiment, cuda)
        else:
            Gx, Gz, Disc, optim_g, optim_d, train_loader, cuda, configs = initialize_ali(
                configs, train)
            training_loop_ali(Gz, Gx, Disc, optim_d, optim_g,
                              train_loader, configs, experiment, cuda)

    if configs['cluster']:

        if configs['enc_model'] == "hali":

            best_f1, best_acc, best_model = get_results_hali(
                configs, experiment, train, labeled, valid_data)

        else:

            best_f1, best_acc, best_model = get_results_ali(
                configs, experiment, train, labeled, valid_data)

        experiment.log_metric('best_accuracy', best_acc)
        experiment.log_metric('best_f1-score', best_f1)
        experiment.log_metric('best_model_epoch', best_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=Constants.DATAPATH,
                        help="Path to dataset folder")
    parser.add_argument("--encoder_path", type=str, default=None)

    parser.add_argument("--config", type=str, default="HALI",
                        help="To select configuration from config.json")
    args = parser.parse_args()
    config_key = args.config
    datapath = args.datapath
    path_to_model = args.encoder_path

    with open(Constants.CONFIG_PATH, 'r') as f:
        configuration = json.load(f)[config_key]

    # Parse configuration file
    batch_size = configuration['batch_size']
    seed = configuration['seed']
    n_epochs = configuration['n_epochs']

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

    if configuration['encode']:
        experiment = OfflineExperiment(project_name="ali", workspace='timothynest',  # Replace this with appropriate comet workspace
                                       offline_directory=str('experiments/' + configuration['experiment']))
    elif configuration['cluster']:
        experiment = OfflineExperiment(project_name="ali", workspace='timothynest',  # Replace this with appropriate comet workspace
                                       offline_directory=str('experiments/' + configuration['experiment'] + '/cluster'))
    experiment.set_name(name=configuration['experiment'])
    experiment.log_parameters(configuration)
    experiment.add_tag(configuration['experiment'])

    # Initiate experiment
    main(datapath, configuration, experiment)
