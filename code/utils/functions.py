import json
import logging
import os
import shutil

import torch


def set_logger(log_path):
    """Set the logger to log info in terminal and file log_path.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example
    -------
        $ logging.info("Starting training...")

    Parameters
    ----------
    log_path : str
        Where to log.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:\n%(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Parameters
    ----------
    d : dict
        Dictionary of float-castable values (np.float, int, float, etc.).
    json_path : str
        Path to JSON file.
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v.dump() for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(model, state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Parameters
    ----------
    state : dict
        Contains model's state_dict, may contain other keys such as epoch, optimizer state_dict.
    is_best : bool
        True if it is the best model seen till now.
    checkpoint : str
        Folder where parameters are to be saved.
    """
    state_filepath = os.path.join(checkpoint, 'last.pth.tar')
    model_filepath = os.path.join(checkpoint, 'last_model.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, state_filepath)
    torch.save(model, model_filepath)
    if is_best:
        shutil.copyfile(state_filepath, os.path.join(checkpoint, 'best.pth.tar'))
        shutil.copyfile(model_filepath, os.path.join(checkpoint, 'best_model.pth'))


def load_checkpoint(checkpoint, model, params, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Parameters
    ----------
    checkpoint : str
        Filename which needs to be loaded.
    model : torch.nn.Module
        Model for which the parameters are loaded.
    params: object
        Parameters object that handles configuration.
    optimizer: torch.optim, optional
        Optimizer to resume from checkpoint.
    scheduler: torch.optim, optional
        Scheduler to resume from checkpoint.

    Returns
    -------
    checkpoint : dict
        Dictionary with saved states.
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    params.init_epoch = checkpoint['epoch']

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_dict'])

    return checkpoint
