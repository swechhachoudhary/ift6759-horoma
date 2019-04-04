import torch
import datetime
from models.encoders import *
from models.clustering import *
from utils.utils import *
from utils.constants import Constants
from comet_ml import OfflineExperiment
from torch.utils.data import ConcatDataset
from data.dataset import LocalHoromaDataset

def get_class_prediction(encoding_model, clustering_model, encoded_unlabeled_train, unlabeled_train, labeled_train,
                           labeled_valid, batch_size, device, experiment):
    # Apply a clustering model algorithm on an embedded space provided by the encoding_model
    # Return a class prediction for each sample within encoded_unlabeled_train
    print("Start encoding of labeled dataset...")
    encoded_labeled_train = encode_dataset(encoding_model, labeled_train, batch_size, device)
    print("Done")
    print("Start encoding of labeled dataset...")
    encoded_labeled_valid = encode_dataset(encoding_model, labeled_valid, batch_size, device)
    print("Done")
    print("Start kmean training on unlabeled...")
    clustering_model.train(encoded_unlabeled_train)
    print("Done")
    cluster_labels = assign_labels_to_clusters(clustering_model, encoded_labeled_train, labeled_train.targets)
    _, accuracy, f1 = eval_model_predictions(clustering_model, encoded_labeled_valid, labeled_valid.targets, cluster_labels)
    experiment.log_metric('accuracy', accuracy)
    experiment.log_metric('f1-score', f1)
    
    unlabeled_target_pred_by_cluster = clustering_model.predict_cluster(encoded_unlabeled_train)

    tensor_unla_train = torch.Tensor(unlabeled_train.data)
    tensor_unla_target_pred_by_cluster = torch.Tensor(unlabeled_target_pred_by_cluster)
    numpy_unla_train = tensor_unla_train.cpu().numpy()
    numpy_unla_target_pred_by_cluster = tensor_unla_target_pred_by_cluster.cpu().numpy()
    
    return numpy_unla_train, numpy_unla_target_pred_by_cluster


def _get_encoding_model(encoding_model_name, latent_dim, device, seed):
    if encoding_model_name == "cvae":
        now = datetime.datetime.now()
        pth_filename = "autoencoder_pretrain_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
        encoding_model = CVAE(latent_dim=latent_dim,
                              folder_save_model="damic_models/",
                              pth_filename_save_model=pth_filename).to(device)
    elif encoding_model_name == "convae":
        encoding_model = ConvAE(latent_dim=latent_dim).to(device)
    else:
        print('No encoding model specified. Using CVAE.')
        encoding_model = CVAE(seed)
    return encoding_model
        
    
def _get_clustering_model(n_clusters, seed):
    return KMeansClustering(n_clusters, seed)


def _initialize_damic_conv_clustering_net_weights(damic_model, conv_net_pretrain_config, numpy_unla_train,
                                                  numpy_unla_target_pred_by_cluster, labeled_train_and_valid, device, experiment):
    print("Start pre-training of clustering convolutional model...")
    lr = conv_net_pretrain_config["lr"]
    batch_size = conv_net_pretrain_config["batch_size"]
    n_epoch = conv_net_pretrain_config["n_epochs"]
    pretrain_dataset_with_label = LocalHoromaDataset(numpy_unla_train, numpy_unla_target_pred_by_cluster)
    
    pretrain_dataset_predicted_label_loader = DataLoader(pretrain_dataset_with_label, batch_size=batch_size)
    valid_and_train_real_label_loader = DataLoader(labeled_train_and_valid, batch_size=batch_size)
    
    clust_network_params = list(damic_model.clustering_network.parameters()) + list(damic_model.output_layer_conv_net.parameters())
    optimizer = torch.optim.Adam(clust_network_params, lr=lr)
    print("Done")

    # Used to save the convolutional network model at the end of the pre training
    now = datetime.datetime.now()
    pth_filename = "conv_net_pretrain_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
    return train_network(damic_model, pretrain_dataset_predicted_label_loader, valid_and_train_real_label_loader, optimizer, n_epoch,
                         device, experiment, train_classifier=True, folder_save_model="damic_models/", pth_filename_save_model=pth_filename)
 

def _get_class_predictions_for_damic_pretraining(datapath, train_subset, overlapped, ae_pretrain_config, device, experiment, seed):
    unlabeled_train, labeled_train, labeled_valid, labeled_train_and_valid = load_original_horoma_datasets(datapath, 
                                                                                                           train_subset=train_subset, 
                                                                                                           overlapped=overlapped)
    print("Shape of unlabeled training set: ", unlabeled_train.data.shape)
    print("Shape of labeled training set: ", labeled_train.data.shape)
    print("Shape of labeled valid set: ", labeled_valid.data.shape)
    print("Shape of labeled train and valid set: ", labeled_train_and_valid.data.shape)
    
    latent_dim = ae_pretrain_config["latent_dim"]
    encoding_model = _get_encoding_model(ae_pretrain_config["enc_model"], latent_dim, device, seed)
    n_clusters = ae_pretrain_config["n_clusters"]
    clustering_model = _get_clustering_model(n_clusters, seed)
    
    batch_size = ae_pretrain_config['batch_size']
    n_epochs = ae_pretrain_config['n_epochs']
    lr = ae_pretrain_config['lr']
    if ae_pretrain_config['train_encoder']:
        print("Start training of pre-train auto-encoder...")
        encoded_unlabeled_train, encoding_model = encoding_model.fit(data=unlabeled_train, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                                     device=device, experiment=experiment)
        print("Done")
    else:
        # Load encoding model and apply encoding
        print("Load pre-train auto-encoder...")
        path_to_model = ae_pretrain_config["encoder_path"]
        encoding_model.load_state_dict(torch.load(path_to_model)["model"])
        print("Done")
        print("Start encoding of unlabeled dataset...")
        encoded_unlabeled_train = encode_dataset(encoding_model, unlabeled_train, batch_size, device)
        print("Done")
    
    a, b = get_class_prediction(encoding_model, clustering_model, encoded_unlabeled_train, unlabeled_train, labeled_train, labeled_valid,
                                batch_size, device, experiment)
    return a, b, labeled_train_and_valid    

def _initialize_damic_autoencoders_weights(damic_model, damic_autoencoders_pretrain_config, numpy_unla_train, 
                                           numpy_unla_target_pred_by_cluster, device, experiment):
    lr = damic_autoencoders_pretrain_config["lr"]
    n_epochs = damic_autoencoders_pretrain_config["n_epochs"]
    batch_size = damic_autoencoders_pretrain_config["batch_size"]
    now = datetime.datetime.now()
    pth_filename = "autoencoders_pretrain_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
    for i in range(17):
        print("Start pre-training of auto encoder for cluster ...")
        indexes_of_class_i = np.where(np.isin(numpy_unla_target_pred_by_cluster,[i]))
        data_of_class_i = numpy_unla_train[indexes_of_class_i]
        _, damic_model.autoencoders[i] = damic_model.autoencoders[i].fit(data=data_of_class_i, batch_size=batch_size, n_epochs=n_epochs,
                                                       lr=lr, device=device, experiment=experiment)
        print("Done")
    now = datetime.datetime.now()
    pth_filename = "damic_models/autoencoders_pretrain_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
    torch.save({
        "model": damic_model.state_dict(),
    }, pth_filename + ".pth")
    return damic_model

def execute_damic_pre_training(datapath, damic_model, train_subset, overlapped, ae_pretrain_config, conv_net_pretrain_config,
                               damic_autoencoders_pretrain_config, experiment, seed):
    """
    See 'Deep clustering based on a mixture of autoencoders' paper
    DAMIC needs a pre-training to initialize both weights for the autoencoders and the
    convolutional clustering network.
    At first we train one autoencoder (encoding_model) on all the unlabeled data available
    Then, we apply one clustering model (clustering_model) to label those unlabeled data
    After that, we train each autoencoder of DAMIC (1 encoder for each cluster/targets) only
    on the data of the same class. Ex: autoencoder at index 0 will be only trained on data
    labeled by the clustering model as being of class 0.
    Finally we initialize the weights of the convolutional clustering network of DAMIC by using
    CrossEntropyLoss on the labeled data.
    
    :param datapath: datapath for the data to do the pre training on
    :param damic_model: the model we want to do the pre-training on
    :param train_subset: how many data from the dataset we should do the training on
    :param overlapped: if True, we use the overlapped datasets
    :param ae_pretrain_config: configuration dictionary for the auto-encoder used during pre-training
    :param conv_net_pretrain_config: configuration dictionary for the convolutional clustering network for pre-tranining
    :param damic_autoencoders_pretrain_config: configuration dictionary for all the autoencoders part of DAMIC for pre-training
    :param experiment: comet-ml experiment to save training results
    :seed: seed for reproducible results
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("== Start DAMIC pre-training ...")
    numpy_unla_train, numpy_unla_target_pred_by_cluster, labeled_train_and_valid = \
                                            _get_class_predictions_for_damic_pretraining(datapath, train_subset, overlapped,
                                                                                         ae_pretrain_config, device, experiment, seed)

    
    
    # Use the k-means clustering to initialize the clustering network parameters
    # If no model path is specified for the pretraining, we train it from scratch, otherwise we load it
    conv_net_pretrain_path = conv_net_pretrain_config["conv_net_pretrain_path"]
    if conv_net_pretrain_path == "":
        damic_model = _initialize_damic_conv_clustering_net_weights(damic_model, conv_net_pretrain_config, numpy_unla_train, 
                                                                numpy_unla_target_pred_by_cluster, labeled_train_and_valid, device, 
                                                                    experiment)
    else:
        damic_model.load_state_dict(torch.load(conv_net_pretrain_path)["model"])

    # Train each auto encoder of each cluster on his own data class
    autoencoders_pretrain_path = damic_autoencoders_pretrain_config["autoencoders_pretrain_path"]
    if autoencoders_pretrain_path == "":
        damic_model = _initialize_damic_autoencoders_weights(damic_model, damic_autoencoders_pretrain_config, numpy_unla_train, 
                                               numpy_unla_target_pred_by_cluster, device, experiment)
    else:
        damic_model.load_state_dict(torch.load(autoencoders_pretrain_path)["model"])

    print("== DAMIC Pre-training done!")
    return damic_model, numpy_unla_train, numpy_unla_target_pred_by_cluster, labeled_train_and_valid


def execute_damic_training(damic_model, configuration, numpy_unla_train, numpy_unla_target_pred_by_cluster, labeled_train_and_valid,
                           device, experiment):
    """
    See 'Deep clustering based on a mixture of autoencoders' paper
    We simultaneously train all the auto-encoders part of DAMIC as well as the Convolutional clustering network.
    
    :param damic_model: the model we want to do the pre-training on
    :param configuration: dictionnary containing all the keys/values part of DAMIC from a config file
    :param numpy_unla_train: numpy array of the unlabeled training dataset
    :param numpy_unla_target_pred_by_cluster: numpy array with the targets for each sample of the unlabeled traning dataset
    :param device: cuda (training is done on gpu) or cpu
    :param experiment: comet-ml experiment to save training results
    """
    print("== Start DAMIC training ...!")
    
    damic_train_config = configuration['damic_train']
    lr = damic_train_config["lr"]
    n_epochs = damic_train_config["n_epochs"]
    batch_size = damic_train_config["batch_size"]
    damic_train_path = damic_train_config["damic_train_path"]
    
    if damic_train_path != "":
        damic_model.load_state_dict(torch.load(damic_train_path)["model"])
    
    pretrain_dataset_with_label = LocalHoromaDataset(numpy_unla_train, numpy_unla_target_pred_by_cluster)
    pretrain_dataset_with_label_loader = DataLoader(pretrain_dataset_with_label, batch_size=batch_size)
    valid_and_train_real_label_loader = DataLoader(labeled_train_and_valid, batch_size=batch_size)
    
    clust_network_params = list(damic_model.clustering_network.parameters()) + list(damic_model.output_layer_conv_net.parameters())
    autoencoders_params = list()
    for i in range(17):
        autoencoders_params = autoencoders_params + list(damic_model.autoencoders[i].parameters())
    damic_parameters = clust_network_params + autoencoders_params
    
    optimizer = torch.optim.Adam(damic_parameters, lr=lr)
    now = datetime.datetime.now()
    pth_filename = "damic_train_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
    damic_model = train_network(damic_model,
                               pretrain_dataset_with_label_loader,
                               valid_and_train_real_label_loader,
                               optimizer,
                               n_epochs,
                               device,
                               experiment,
                               train_classifier=False,
                               train_damic=True, folder_save_model="damic_models/", pth_filename_save_model=pth_filename)

    print("== DAMIC training done!")
