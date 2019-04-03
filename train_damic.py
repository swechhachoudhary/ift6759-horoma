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

    print("== Start DAMIC training ...!")
    damic_train_config = configuration['damic_train']
    pretrain_dataset_with_label = LocalHoromaDataset(numpy_unla_train, numpy_unla_target_pred_by_cluster)
    clust_network_params = list(damic_model.clustering_network.parameters()) + list(damic_model.output_layer_conv_net.parameters())
    autoencoders_params = list()
    for i in range(17):
        autoencoders_params = autoencoders_params + list(damic_model.autoencoders[i].parameters())
    damic_parameters = clust_network_params + autoencoders_params
    lr = damic_train_config["lr"]
    n_epochs = damic_train_config["n_epochs"]
    optimizer = torch.optim.Adam(damic_parameters, lr=lr)
    damic_model = train_network(damic_model,
                               pretrain_dataset_with_label,
                               None,
                               optimizer,
                               n_epochs,
                               device,
                               experiment,
                               train_classifier=False,
                               train_damic=True)

    print("== DAMIC training done!")
    torch.save(damic_model, Constants.PATH_TO_MODEL + "DAMIC_MODEL" + str(experiment.get_key()) + '.pth')
    
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
