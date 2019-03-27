import argparse


def get_damic_argparser(self):
    # Return an argparser used for training a DAMIC
    parser = argparse.ArgumentParser(description='Damic Training')
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
    return parser
