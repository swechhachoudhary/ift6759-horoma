import os
import json
import argparse

import torch
import numpy as np

from utils.dataset import HoromaDataset
from utils.mnist_dataset import CustomMNIST
from utils.factories import ModelFactory, OptimizerFactory, TrainerFactory
from utils.transforms import HoromaTransforms
from utils.mnist_dataset import CustomlabeledMNIST


def main(config, resume, test_run=False, helios_run=None, horoma_test=False):
    """
    Execute a training for a model.

    :param config: the configuration of the optimizer, model and trainer.
    :param resume: path to the checkpoint of a model.
    :param test_run: whether it's a test run or not. In case of test run,
    uses custom mnist dataset.
    :param helios_run: start datetime of a run on helios.
    :param horoma_test: whether to use the test horoma dataset or not.
    """
    np.random.seed(config["numpy_seed"])
    torch.manual_seed(config["torch_seed"])
    torch.cuda.manual_seed_all(config["torch_seed"])

    # setup data_loader instances
    if not test_run:
        unlabeled = HoromaDataset(
            **config["data"]["dataset"],
            split='train',
            transforms=HoromaTransforms()
        )
        unlabeled_overlapped = HoromaDataset(
            **config["data"]["dataset"],
            split='train_overlapped',
            transforms=HoromaTransforms()
        )
        train_labeled = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='train_labeled',
            transforms=HoromaTransforms()
        )
        train_labeled_overlapped = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='train_labeled_overlapped',
            transforms=HoromaTransforms()
        )
        valid = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='valid',
            transforms=HoromaTransforms()
        )
        valid_overlapped = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='valid_overlapped',
            transforms=HoromaTransforms()
        )
        train_labeled = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='train_labeled',
            transforms=HoromaTransforms()
        )
    elif horoma_test:

        unlabeled = HoromaDataset(
            **config["data"]["dataset"],
            split='train_overlapped',
            transforms=HoromaTransforms(),
            subset=5000
        )

        labeled = HoromaDataset(
            data_dir=config["data"]["dataset"]['data_dir'],
            flattened=False,
            split='valid_overlapped',
            transforms=HoromaTransforms(),
            subset=1000
        )
    else:
        unlabeled = CustomMNIST(**config["data"]["dataset"], subset=5000)
        labeled = CustomlabeledMNIST(**config["data"]["dataset"],
                                     subset=1000)

    train_all_dataset = HoromaDataset(config['data']['dataset']['data_dir'],
                                      split="train_all", subset=None, skip=0,
                                      flattened=False, transforms=None)

    model = ModelFactory.get(config)

    print(model)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OptimizerFactory.get(config, trainable_params)

    trainer = TrainerFactory.get(config)(
        model,
        optimizer,
        resume=resume,
        config=config,
        unlabeled=train_all_dataset,
        helios_run=helios_run,
        **config['trainer']['options']
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoEncoder Training')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--config', default=None, type=str,
                       help='config file path (default: None)')
    group.add_argument('-r', '--resume', default=None, type=str,
                       help='path to latest checkpoint (default: None)')
    group.add_argument('--test-run', action='store_true',
                       help='execute a test run on MNIST')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument("--helios-run", default=None, type=str,
                        help='if the train is run on helios with '
                             'the run_experiment script,'
                             'the value should be the time at '
                             'which the task was submitted')
    args = parser.parse_args()

    horoma_test = False

    if args.config:
        # load config file
        config = json.load(open(args.config))
    elif args.resume:
        # load config file from checkpoint, in case new config file
        # is not given.
        # Use '--config' and '--resume' arguments together to
        # load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    elif args.test_run:
        config = json.load(open("./configs/variational_mnist.json"))
    else:
        args.test_run = True
        dataset = 'mnist'

        filename = "./configs/test_{}.json".format(dataset)
        config = json.load(open(filename))

        horoma_test = dataset == 'horoma'

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume, args.test_run, args.helios_run, horoma_test)
