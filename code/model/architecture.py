"""Define models and generator functions which receives params as parameter, then add model to available models"""
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class HumanwareDenseNet(DenseNet):
    """DenseNet implementation"""
    def __init__(self, params):
        super().__init__(params.growth_rate, params.block_config, params.num_init_features,
                         params.bn_size, params.drop_rate, params.num_classes)


class HumanwareResNet(ResNet):
    """ResNet implementation"""
    def __init__(self, params):
        assert params.block in ['basic', 'bottleneck'], "The desired block does not exist"
        if params.block == 'basic':
            block = BasicBlock
        else:
            block = Bottleneck
        super().__init__(block, params.layers, params.num_classes)


# Add all the models to the dictionary below. This is called in train using the desired model as input.
models = {
    '1': HumanwareDenseNet,
    '2': HumanwareResNet,
}
