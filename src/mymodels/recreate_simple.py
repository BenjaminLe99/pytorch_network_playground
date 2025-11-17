from models.layers import (
    InputLayer, StandardizeLayer, ResNetPreactivationBlock, DenseBlock, PaddingLayer, RotatePhiLayer,
)

from utils.utils import EMPTY_INT, EMPTY_FLOAT, embedding_expected_inputs
import torch
from mymodels.mylayers import (DenseNetwork)

def init_layers(continous_features, categorical_features, config):
    # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
    eps = config["eps"]
    # activate weight normalization on linear layer weights
    normalize = config["linear_layer_normalization"]

    # helper where all layers are defined
    # std layers are filled when statitics are known
    std_layer = StandardizeLayer(mean=config["mean"], std=config["std"])

    continuous_padding = PaddingLayer(padding_value=-4, mask_value=EMPTY_FLOAT)
    categorical_padding = PaddingLayer(padding_value=config["empty_value"], mask_value=EMPTY_INT)
    # rotation_layer = RotatePhiLayer(
    #     columns=list(map(str, continous_features)),
    #     ref_phi_columns=config["ref_phi_columns"],
    #     rotate_columns=config["rotate_columns"],
    # )
    input_layer = InputLayer(
        continuous_inputs=continous_features,
        categorical_inputs=categorical_features,
        embedding_dim=config["embedding_dim"],
        expected_categorical_inputs=embedding_expected_inputs,
        empty=config["empty_value"],
        std_layer=std_layer,
        # rotation_layer=rotation_layer,
        rotation_layer=None,
        padding_categorical_layer=categorical_padding,
        padding_continous_layer=continuous_padding,
    )

    model = torch.nn.Sequential(
        input_layer,
        DenseNetwork(
        input_nodes=input_layer.ndim,
        hidden_nodes=config["layers_and_nodes"],
        output_nodes=3,
        activation=config["activation_functions"],
        eps=config["eps"],
        normalize=config["linear_layer_normalization"],
        )
        # no softmax since this is already part of loss
    )

    # model = SimpleNetRecreated(
    #     input_layer=input_layer,
    #     hidden_nodes=config["layers_and_nodes"],
    #     activation=config["activation_functions"],
    #     eps=config["eps"],
    #     normalize=config["linear_layer_normalization"],
    # )

    # resnet_blocks = [
    #     ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=eps, normalize=normalize)
    #     for num_blocks in range(config["num_resblocks"])
    #     ]

    # model = torch.nn.Sequential(
    #     input_layer,
    #     DenseBlock(input_nodes = input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize), # noqa
    #     *resnet_blocks,
    #     torch.nn.Linear(config["nodes"], 3),
    #     # no softmax since this is already part of loss
    # )
    return input_layer, model
