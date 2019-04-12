import os
import sys
import argparse

import torch
import numpy as np
from joblib import load  # You can use Pickle or the serialization technique of your choice

sys.path.append("../")
from data.dataset import OriginalHoromaDataset
import models.transformer_net as transformer_net


def eval_model(model_path, dataset_dir, split):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # SETUP DATASET # #
    # Load requested dataset
    dataset = OriginalHoromaDataset(dataset_dir, split=split)
    data = dataset[:][0]
    data = data.reshape([data.shape[0], 1, 3072])

    # # SETUP MODEL # #
    # Load your best model
    print("\nLoading model from ({}).".format(model_path))

    out_size = 17
    n_layers = 0
    hidden_size = 256
    kernel_size = 8
    pool_size = 4
    dropout = 0.2
    n_heads = 8
    key_dim = 128
    val_dim = 128
    inner_dim = 128

    model = transformer_net.TransformerNet(
        1, out_size, hidden_size, n_layers, kernel_size=kernel_size, pool_size=pool_size,
        n_heads=n_heads, key_dim=key_dim, val_dim=val_dim, inner_dim=inner_dim, dropout=dropout
    ).to('cuda')
    
    resume = torch.load(model_path, map_location='cuda')
    
    if ('module' in list(resume['state_dict'].keys())[0]) \
            and not (isinstance(model, torch.nn.DataParallel)):
        new_state_dict = OrderedDict()
        for k, v in resume['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(resume['state_dict'])

    # # PREDICTIONS # #
    # Return the predicted classes as a numpy array of shape (nb_exemple, 1)

    y_pred = model(data.to(device))

    y_pred = y_pred.cpu()
    _, y_pred = y_pred.max(1)
    return y_pred.detach().numpy()


if __name__ == "__main__":

    # Put your group name here
    group_name = "b3phot5"

    # model_path should be the absolute path on shared disk to your best model.
    # You need to ensure that they are available to evaluators on Helios.
    model_path = "/rap/jvb-000-aa/COURS2019/etudiants/submissions/b3phot5/model/transformer_net.pt"

    #########################
    # DO NOT MODIFY - BEGIN #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str, default="/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/", help="Absolute path to the dataset directory.")
    parser.add_argument("-s", "--dataset_split", type=str, choices=['valid', 'test', 'train'], default="valid", help="Which split of the dataset should be loaded from `dataset_dir`.")
    parser.add_argument("-r", "--results_dir", type=str, default="./", help="Absolute path to where the predictions will be saved.")
    args = parser.parse_args()

    # Arguments validation
    if group_name is "b1phutN":
        print("'group_name' is not set.\nExiting ...")
        exit(1)

    if model_path is None or not os.path.exists(model_path):
        print("'model_path' ({}) does not exists or unreachable.\nExiting ...".format(model_path))
        exit(1)

    if args.dataset_dir is None or not os.path.exists(args.dataset_dir):
        print("'dataset_dir' does not exists or unreachable..\nExiting ...")
        exit(1)

    y_pred = eval_model(model_path, args.dataset_dir, args.dataset_split)

    assert type(y_pred) is np.ndarray, "Return a numpy array"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = os.path.join(args.results_dir, "{}_pred_{}.txt".format(group_name, args.dataset_split))

    print('\nSaving results to ({})'.format(results_fname))
    np.savetxt(results_fname, y_pred, fmt='%s')
    # DO NOT MODIFY - END #
    #######################