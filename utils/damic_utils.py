import torch
import time
import datetime
import numpy as np
from models.encoders import CVAE, ConvAE
from models.clustering import KMeansClustering
from utils.utils import assign_labels_to_clusters, eval_model_predictions, compute_metrics, load_original_horoma_datasets
from utils.model_utils import encode_dataset, train_network
from torch.utils.data import DataLoader
from utils.dataset import LocalHoromaDataset


def get_class_prediction(encoding_model, clustering_model, encoded_unlabeled_train, unlabeled_train, labeled_train,
                         labeled_valid, batch_size, device, experiment):
    """
    Apply a clustering model algorithm on an embedded space provided by the encoding_model
    Return a class prediction for each sample within encoded_unlabeled_train

    :param encoding_model: model used to do the encoding
    :param clustering_model: model use to do the clustering
    :param encoded_unlabeled_train: dataset with the unlabeled samples encoded
    :param unlabeled_train: dataset with the unlabeled samples
    :param labeled_train: dataset with labeled samples for training
    :param labeled_valid: dataset with labeled samples for validation
    :param batch_size: batch size
    :param device: cpu or cuda
    :param experiment: comet experiment to log results
    :return: numpy array of the unlabeled data, numpy array with the predict class for each sample
    """

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
    _, accuracy, f1 = eval_model_predictions(
        clustering_model, encoded_labeled_valid, labeled_valid.targets, cluster_labels)
    experiment.log_metric('accuracy', accuracy)
    experiment.log_metric('f1-score', f1)

    unlabeled_target_pred_by_cluster = clustering_model.predict_cluster(encoded_unlabeled_train)

    tensor_unla_train = torch.Tensor(unlabeled_train.data)
    tensor_unla_target_pred_by_cluster = torch.Tensor(unlabeled_target_pred_by_cluster)
    numpy_unla_train = tensor_unla_train.cpu().numpy()
    numpy_unla_target_pred_by_cluster = tensor_unla_target_pred_by_cluster.cpu().numpy()

    return numpy_unla_train, numpy_unla_target_pred_by_cluster


def _get_encoding_model(encoding_model_name, latent_dim, device, seed):
    """
    Train an encoding model

    :param encoding_model_name: name of the encoding model being trained
    :param latent_dim: dimension of the encoded samples
    :param device: cpu or cuda
    :param seed
    :return: a trained encoding model of type encoding_model_name
    """
    if encoding_model_name == "cvae":
        now = datetime.datetime.now()
        pth_filename = "autoencoder_pretrain_" + str(now.month) + "_" + \
            str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
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
    """
    Part of the pretraining for Damic is to initialize the weights for the convolutional clustering network.

    :param damic_model
    :param conv_net_pretrain_config: dictionary of configuration for the conv net pretraining
    :param numpy_unla_train: numpy array of unlabeled samples
    :param numpy_unla_target_pred_by_cluster: numpy array of targets for the unlabeled samples
    :param labeled_train_and_valid: dataset composed of the labeled trained and validation set
    :param device: cpu or cuda
    :param experiment: comet experiment to log results
    :return: a damic_model with his convolutional clustering network weights initialized
    """

    print("Start pre-training of clustering convolutional model...")
    lr = conv_net_pretrain_config["lr"]
    batch_size = conv_net_pretrain_config["batch_size"]
    n_epoch = conv_net_pretrain_config["n_epochs"]
    pretrain_dataset_with_label = LocalHoromaDataset(numpy_unla_train, numpy_unla_target_pred_by_cluster)

    pretrain_dataset_predicted_label_loader = DataLoader(pretrain_dataset_with_label, batch_size=batch_size)
    valid_and_train_real_label_loader = DataLoader(labeled_train_and_valid, batch_size=batch_size)

    optimizer = torch.optim.Adam(damic_model.parameters(), lr=lr)
    print("Done")

    # Used to save the convolutional network model at the end of the pre training
    now = datetime.datetime.now()
    pth_filename = "conv_net_pretrain_" + str(now.month) + "_" + str(now.day) + \
        "_" + str(now.hour) + "_" + str(now.minute)
    return train_network(damic_model, pretrain_dataset_predicted_label_loader, valid_and_train_real_label_loader, optimizer, n_epoch,
                         device, experiment, train_classifier=True, folder_save_model="damic_models/", pth_filename_save_model=pth_filename)


def _get_class_predictions_for_damic_pretraining(datapath, train_subset, overlapped, ae_pretrain_config, device, experiment, seed):
    """
    For pre-training purposes, Damic needs a first class prediction on the unlabeled samples

    :param datapah
    :param train_subset: if we use a subset of the samples
    :param overlapped: if True, we use the overlapped dataset
    :param ae_pretrain_config: dictionary of configuration for this step
    :param device: cpu or cuda
    :param experiment: comet experiment to log results
    :return: numpy array of unlabeled samples, numpy array of target for each sample, dataset of labeled train+valid samples
    """

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

    array_unlabeled_samples, array_targets = \
        get_class_prediction(encoding_model, clustering_model, encoded_unlabeled_train, unlabeled_train,
                             labeled_train, labeled_valid,
                             batch_size, device, experiment)
    return array_unlabeled_samples, array_targets, labeled_train_and_valid


def _initialize_damic_autoencoders_weights(damic_model, damic_autoencoders_pretrain_config, numpy_unla_train,
                                           numpy_unla_target_pred_by_cluster, device, experiment):
    """
    Part of the pretraining for Damic is to initialize the weights for each autoencoders.

    :param damic_model
    :param damic_autoencoders_pretrain_config: dictionary of configuration for this step
    :param numpy_unla_train: numpy array of unlabeled samples
    :param numpy_unla_target_pred_by_cluster: numpy array of targets for the unlabeled samples
    :param device: cpu or cuda
    :param experiment: comet experiment to log results
    :return: a damic_model with each autoencoders weights initialized
    """

    lr = damic_autoencoders_pretrain_config["lr"]
    n_epochs = damic_autoencoders_pretrain_config["n_epochs"]
    batch_size = damic_autoencoders_pretrain_config["batch_size"]
    now = datetime.datetime.now()
    pth_filename = "autoencoders_pretrain_" + str(now.month) + "_" + \
        str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)

    # Didn't do a loop because it seems encapsulating the auto encoders into
    # an array led into issues with optimizing the parameters
    _, damic_model.ae1 = damic_model.ae1.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [0]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae2 = damic_model.ae2.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [1]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae3 = damic_model.ae3.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [2]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae4 = damic_model.ae4.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [3]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae5 = damic_model.ae5.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [4]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae6 = damic_model.ae6.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [5]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae7 = damic_model.ae7.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [6]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae8 = damic_model.ae8.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [7]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae9 = damic_model.ae9.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [8]))],
                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae10 = damic_model.ae10.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [9]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae11 = damic_model.ae11.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [10]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae12 = damic_model.ae12.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [11]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae13 = damic_model.ae13.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [12]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae14 = damic_model.ae14.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [13]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae15 = damic_model.ae15.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [14]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae16 = damic_model.ae16.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [15]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)
    _, damic_model.ae17 = damic_model.ae17.fit(data=numpy_unla_train[np.where(np.isin(numpy_unla_target_pred_by_cluster, [16]))],
                                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, device=device, experiment=experiment)

    now = datetime.datetime.now()
    pth_filename = "damic_models/autoencoders_pretrain_" + \
        str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
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
    :param labeled_train_and_valid: dataset composed of the labeled trained and validation set
    :param device: cuda (training is done on gpu) or cpu
    :param experiment: comet-ml experiment to save training results
    :return: trained damic_model
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

    damic_model_parameters = damic_model.parameters()

    optimizer = torch.optim.Adam(damic_model_parameters, lr=lr)
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
    return damic_model


def get_accuracy_f1_scores_from_damic_model(damic_model, labeled_train_and_valid, device):
    """
    Predict labels and compare to true labels to compute the accuracy and F1 score.

    :param damic_model: the model we want to do the pre-training on
    :param labeled_train_and_valid: dataset composed of the labeled trained and validation set
    :param device: cuda (training is done on gpu) or cpu
    :return: predictions made by damic, accuracy and f1 score
    """
    print("Evaluating DAMIC model ...")
    start_time = time()

    valid_and_train_real_label_loader = DataLoader(labeled_train_and_valid, batch_size=len(labeled_train_and_valid))

    with torch.no_grad():
        for inputs, labels in valid_and_train_real_label_loader:
            inputs = inputs.to(device)
            labels = labels.long()
            labels = labels.squeeze()
            print("Accuracy predictions")
            damic_predictions = damic_model(inputs)
            _, damic_predictions = damic_predictions.max(1)
            print("DAMIC predictions results")
            print(damic_predictions)
            print("Expected results")
            print(labels)

    accuracy, f1 = compute_metrics(labels, damic_predictions.cpu())

    print(
        "Done in {:.2f} sec | Accuracy: {:.2f} - F1: {:.2f}".format(time() - start_time, accuracy * 100, f1 * 100))

    return damic_predictions, accuracy, f1
