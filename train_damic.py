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

def main(datapath, configuration, config_key):
    """
    :param datapath: path to the directory containing the samples
    :param configuration: dictionnary containing all the keys/values part of the config_key json file
    :param config_key: key of the configuration we have to load from the configuration file (Ex: DAMIC)
    """ 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed = configuration['seed']
    _set_torch_seed(seed)

    experiment = _set_comet_experiment(configuration, config_key)
    
    damic_model = DAMICClustering(17).to(device)

    ae_pretrain_config = configuration['autoencoder_pretrain']
    conv_net_pretrain_config = configuration['damic_conv_net_pretrain']
    damic_autoencoders_pretrain_config = configuration['damic_autoencoders_pretrain']
    train_subset = configuration['train_subset']
    overlapped = configuration['overlapped']
    damic_model, numpy_unla_train, numpy_unla_target_pred_by_cluster = execute_damic_pre_training(datapath, damic_model, train_subset,
                                                                                                  overlapped, ae_pretrain_config,
                                                                                                  conv_net_pretrain_config,
                                                                                                  damic_autoencoders_pretrain_config,
                                                                                                  experiment, seed)

    execute_damic_training(damic_model, configuration, numpy_unla_train, numpy_unla_target_pred_by_cluster, device, experiment)
    
def _set_torch_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def _set_comet_experiment(configuration, config_key):
    experiment = OfflineExperiment(project_name='general',
                                   workspace='benjaminbenoit',
                                   offline_directory="experiments")
    experiment.set_name(config_key)
    experiment.log_parameters(configuration)
    return experiment

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=Constants.DATAPATH,
                        help="Path to dataset folder")
    parser.add_argument("--config", type=str, default="DAMIC", help="To select configuration from config.json")
    args = parser.parse_args()
    config_key = args.config
    datapath = args.datapath

    with open(Constants.CONFIG_PATH, 'r') as f:
        configuration = json.load(f)[config_key]

    # Initiate experiment
    main(datapath, configuration, config_key)
