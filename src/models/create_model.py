from models.layers import (
    InputLayer, StandardizeLayer, ResNetPreactivationBlock, DenseBlock, PaddingLayer, RotatePhiLayer,
)

from utils.utils import EMPTY_INT, EMPTY_FLOAT, embedding_expected_inputs
import torch


def init_layers(continous_features, categorical_features, config):
    # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
    eps = 0.5e-5
    # activate weight normalization on linear layer weights
    normalize = False

    # helper where all layers are defined
    # std layers are filled when statitics are known
    std_layer = StandardizeLayer()

    continuous_padding = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
    categorical_padding = PaddingLayer(padding_value=config["empty_value"], mask_value=EMPTY_INT)
    # rotation_layer = RotatePhiLayer(
    #     columns=list(map(str, continous_features)),
    #     ref_phi_columns=config["ref_phi_columns"],
    #     rotate_columns=config["rotate_columns"],
    # )
    input_layer = InputLayer(
        continuous_inputs=continous_features,
        categorical_inputs=categorical_features,
        embedding_dim=4,
        expected_categorical_inputs=embedding_expected_inputs,
        empty=config["empty_value"],
        std_layer=std_layer,
        # rotation_layer=rotation_layer,
        rotation_layer=None,
        padding_categorical_layer=categorical_padding,
        padding_continous_layer=continuous_padding,
    )

    resnet_blocks = [
        ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=eps, normalize=normalize)
        for num_blocks in range(4)
        ]

    model = torch.nn.Sequential(
        input_layer,
        DenseBlock(input_nodes = input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize), # noqa
        *resnet_blocks,
        torch.nn.Linear(config["nodes"], 3),
        # no softmax since this is already part of loss
    )
    return input_layer, model
