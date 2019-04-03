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
from models.mlp_classifier import MLPClassifier


def main(datapath, encoding_model, classifier_model, batch_size, n_epochs, lr, device,
         train_unlabeled_split, valid_split, train_labeled_split,
         experiment, path_to_model=None):
    """
    :param datapath: path to the directory containing the samples
    :param classifier_model: which classifier model to use
    :param encoding_model: which encoding model to use, convolutional, variational or simple autoencoders.
    :param batch_size: batch size
    :param n_epochs: number of epochs
    :param lr: learning rate
    :param device: use CUDA device if available else CPU .
    :param experiment: track experiment
    :param path_to_model: path to the directory containing saved models.
    """
    train_unlabeled = HoromaDataset(datapath, split=train_unlabeled_split)
    train_labeled = HoromaDataset(datapath, split=train_labeled_split)
    valid_data = HoromaDataset(datapath, split=valid_split)

    # train_unlabeled_loader = loop_over_unlabeled_data(train_unlabeled, batch_size)
    # train_labeled_loader = loop_over_labeled_data(train_labeled, batch_size)
    # valid_loader = DataLoader(valid_data, batch_size=batch_size)

    n_labeled_batch = len(train_labeled) / batch_size
    n_unlabeled_batch = n_labeled_batch

    # Train and apply encoding model
    train_enc, encoding_model = encoding_model.fit(train, valid_data, batch_size=batch_size, n_epochs=n_epochs,
                                                   lr=lr, device=device, experiment=experiment)

    # train network
    best_loss = np.inf
    key = experiment.get_key()
    best_model = None
    parameters = [
        {'params': encoding_model.parameters()},
        {'params': classifier_model.parameters(), 'lr': 1e-3}
    ]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.5)
    for epoch in range(n_epochs):
        scheduler.step()
        # train_loss = _train_one_epoch(
        #     model, train_loader, optimizer, epoch, device, experiment)
            """Train one epoch for model."""
        encoding_model.train()
        classifier_model.train()

        running_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device)

            outputs = encoding_model(inputs)
            criterion = nn.MSELoss(reduction='sum')
            loss = criterion(outputs, inputs)
            running_loss += loss

        for batch_idx, (inputs, labels) in enumerate(train_labeled_loader):
            inputs = inputs.to(device)
            train_enc = encode_dataset(
                encoding_model, train_labeled, batch_size, device)
            outputs = classifier_model()
            criterion = nn.MSELoss(reduction='sum')
            loss = criterion(outputs, inputs)
            running_loss += loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(inputs),
                                                                           len(train_loader) *
                                                                           len(inputs),
                                                                           100. * batch_idx /
                                                                           len(train_loader),
                                                                           loss.item() / len(inputs)))

        train_loss = running_loss / len(train_loader.dataset)
        experiment.log_metric("Train loss", train_loss, step=epoch)
        valid_loss = _test(model, test_loader, epoch, device, experiment)

        try:
            if valid_loss < best_loss:
                torch.save({
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "model": model.state_dict(),
                    "loss": valid_loss
                }, "experiment_models/" + str(key) + '.pth')
                best_loss = valid_loss
                best_model = deepcopy(model)  # Keep best model thus far
        except FileNotFoundError as e:
            print(
                "Directory for logging experiments does not exist. Launch script from repository root.")
            raise e

        print("Training loss after {} epochs: {:.6f}".format(epoch, train_loss))
        print("Validation loss after {} epochs: {:.6f}".format(epoch, valid_loss))

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

    parser.add_argument("--config", type=str, default="CAE_BASE",
                        help="To select configuration from config.json")
    args = parser.parse_args()
    config_key = args.config
    datapath = args.datapath
    path_to_model = args.encoder_path

    with open(Constants.CONFIG_PATH, 'r') as f:
        configuration = json.load(f)[config_key]

    encoding_model = configuration['enc_model']
    batch_size = configuration['batch_size']
    seed = configuration['seed']
    n_epochs = configuration['n_epochs']
    lr = configuration['lr']
    train_split = configuration['train_split']
    valid_split = configuration['valid_split']
    train_labeled_split = configuration['train_labeled_split']
    encode = configuration['encode']
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
    experiment = OfflineExperiment(
        "z15Um8oxWZwiXQXZxZKGh48cl", workspace='swechhachoudhary', offline_directory="swechhas_experiments")

    # Set up Comet Experiment tracking
    # experiment = OfflineExperiment(project_name='general',
    #                                workspace='benjaminbenoit',  # Replace this with appropriate comet workspace
    #                                offline_directory="experiments")
    experiment.set_name(
        name=args.config + "_dim={}_split={}".format(latent_dim, train_split))
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
        classifier_model = MLPClassifier()

    # Initiate experiment
    main(datapath, encoding_model, classifier_model, batch_size, n_epochs, lr, flattened, device, train_split, valid_split, train_labeled_split,
         experiment, path_to_model=path_to_model)
