import torch
from models.encoders import *
from models.clustering import *
from utils.utils import *
from utils.constants import Constants
from comet_ml import OfflineExperiment
from data.dataset import LocalHoromaDataset

def get_class_prediction(encoding_model, clustering_model, encoded_unlabeled_train, unlabeled_train, labeled_train,
                           labeled_valid, batch_size, device, experiment):
    # Apply a clustering model algorithm on an embedded space provided by the encoding_model
    # Return a class prediction for each sample within encoded_unlabeled_train
    print("Start encoding of labeled dataset...")
    encoded_labeled_train = encode_dataset(encoding_model, labeled_train, batch_size, device, for_pre_training=True)
    print("Done")
    print("Start encoding of labeled dataset...")
    encoded_labeled_valid = encode_dataset(encoding_model, labeled_valid, batch_size, device, for_pre_training=True)
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


def execute_damic_pre_training(datapath, damic_model, encoding_model, train_encoder, clustering_model,
                                batch_size, n_epochs, lr, flattened, device, overlapped,
                                 experiment, train_subset, path_to_model):
    """
    See 'Deep clustering based on a mixture of autoencoders' paper
    DAMIC needs a pre-training to initialize both weights for the autoencoders and the
    convolutional clustering network.
    At first we train one autoencoder (encoding_model) on all the unlabeled data available
    We apply one clustering model (clustering_model) to label those unlabeled data
    After that, we train each autoencoder of DAMIC (1 encoder for each cluster/targets) only
    on the data of the same class. Ex: autoencoder at index 0 will be only trained on data
    labeled by the clustering model as being of class 0.
    Finally we initialize the weights of the convolutional clustering network of DAMIC by using
    CrossEntropyLoss on the labeled data.
    
    :param datapath: datapath for the data to do the pre training on
    :param damic_model: the model we want to do the pre-training on
    :param encoding_model: which encoding model to use for the pre-training of DAMIC
    :param train_autoencoder: if True, the auto encoder is trained from scratch
    :param clustering_model: clustering model used for the pre-training of DAMIC
    """
    
    unlabeled_train, labeled_train, labeled_valid = load_original_horoma_datasets(datapath, train_subset=train_subset,
                                   flattened=flattened, overlapped=overlapped)

    print("Shape of unlabeled training set: ", unlabeled_train.data.shape)
    print("Shape of labeled training set: ", labeled_train.data.shape)
    print("Shape of labeled valid set: ", labeled_valid.data.shape)
    
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
        encoded_unlabeled_train = encode_dataset(encoding_model, unlabeled_train, batch_size, device, for_pre_training=True)
        print("Done")
        
    numpy_unla_train, numpy_unla_target_pred_by_cluster = get_class_prediction(encoding_model, clustering_model, encoded_unlabeled_train,
                                                                               unlabeled_train, labeled_train, labeled_valid, batch_size,
                                                                               device, experiment)


    # Use the k-means clustering to initialize the clustering network parameters
    print("Start pre-training of clustering convolutional model...")
    pretrain_dataset_with_label = LocalHoromaDataset(numpy_unla_train, numpy_unla_target_pred_by_cluster)
    clust_network_params = list(damic_model.clustering_network.parameters()) + list(damic_model.output_layer_conv_net.parameters())
    optimizer = torch.optim.Adam(clust_network_params, lr=lr)
    damic_model = train_network(damic_model, pretrain_dataset_with_label, None, optimizer, 1, device, experiment, train_classifier=True)
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
    return damic_model, numpy_unla_train, numpy_unla_target_pred_by_cluster
