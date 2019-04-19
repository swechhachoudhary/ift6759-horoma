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
from models.nrm import NRM


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

    train_loader = DataLoader(train, batch_size=configs[
                              'batch_size'], shuffle=True)
    labeled_loader = DataLoader(labeled, batch_size=configs[
                                'labeled_batch_size'], shuffle=True)
    eval_loader = DataLoader(valid_data, batch_size=configs[
                             'labeled_batch_size'], shuffle=True)

    print("Shape of training set: ", train.data.shape)
    print("Shape of validation set: ", valid_data.data.shape)

    n_iterations = np.floor(
        labeled.data.shape[0] / configs['labeled_batch_size'])
    device = 'cuda'

    net = NRM('AllConv13', batch_size=configs['labeled_batch_size'], num_class=17, use_bias=configs['use_bias'], use_bn=configs[
              'use_bn'], do_topdown=configs['do_topdown'], do_pn=configs['do_pn'], do_bnmm=configs['do_bnmm']).to(device)
    net.apply(weights_init)
    best_f1, best_acc, best_model = train_nrm(net, train_loader, labeled_loader, eval_loader, configs[
                                              'n_epochs'], configs, n_iterations, experiment)

    experiment.log_metric('best_accuracy', best_acc)
    experiment.log_metric('best_f1-score', best_f1)
    experiment.log_metric('best_model_epoch', best_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, default=Constants.DATAPATH,
                        help="Path to dataset folder")
    parser.add_argument("--encoder_path", type=str, default=None)

    parser.add_argument("--config", type=str, default="NRM",
                        help="To select configuration from config.json")
    args = parser.parse_args()
    config_key = args.config
    datapath = args.datapath
    path_to_model = args.encoder_path

    with open(Constants.CONFIG_PATH, 'r') as f:
        configuration = json.load(f)[config_key]

    # Parse configuration file
    seed = configuration['seed']

    # Set all seeds for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists('experiments'):
        print('mkdir ', 'experiments')
        os.mkdir('experiments')

    experiment = OfflineExperiment(project_name="ali", workspace='timothynest',  # Replace this with appropriate comet workspace
                                   offline_directory=str('experiments/' + configuration['experiment']))
    experiment.set_name(
        name=configuration['experiment'])
    experiment.log_parameters(configuration)

    experiment.add_tag(configuration['experiment'])

    MODEL_PATH = 'experiments/' + configuration['experiment'] + '/models'

    if not os.path.exists(MODEL_PATH):
        print('mkdir ', MODEL_PATH)
        os.mkdir(MODEL_PATH)

    configuration['MODEL_PATH'] = MODEL_PATH

    # Initiate experiment
    main(datapath, configuration, experiment)
