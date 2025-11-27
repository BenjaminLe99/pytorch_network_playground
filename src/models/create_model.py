from models.layers import (
    InputLayer, StandardizeLayer, ResNetPreactivationBlock, DenseBlock, PaddingLayer, RotatePhiLayer, DenseNetBlock
)

from utils.utils import EMPTY_INT, EMPTY_FLOAT, embedding_expected_inputs
import torch

def init_weights(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.kaiming_normal(model.weight)


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
        empty=None,
        std_layer=std_layer,
        # rotation_layer=rotation_layer,
        rotation_layer=None,
        padding_categorical_layer=categorical_padding,
        padding_continous_layer=continuous_padding,
    )

    resnet_blocks = [
        ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=eps, normalize=normalize)
        for num_blocks in range(config["num_resblocks"])
        ]

    model = torch.nn.Sequential(
        input_layer,
        DenseBlock(input_nodes = input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize), # noqa
        *resnet_blocks,
        torch.nn.Linear(config["nodes"], 3),
        # no softmax since this is already part of loss
    )
    return input_layer, model

class BNet(torch.nn.Module):
    def __init__(self, continous_features, categorical_features, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(config=config, continous_features=continous_features, categorical_features=categorical_features)

    def init_layers(self, continous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        # eps = 0.5e-5
        eps = 0.001
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        self.std_layer = StandardizeLayer(mean=config["mean"], std=config["std"])

        self.continuous_padding = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
        self.categorical_padding = PaddingLayer(padding_value=config["empty_value"], mask_value=EMPTY_INT)
        # self.rotation_layer = RotatePhiLayer(
        #     columns=list(map(str, continous_features)),
        #     ref_phi_columns=config["ref_phi_columns"],
        #     rotate_columns=config["rotate_columns"],
        # )
        self.input_layer = InputLayer(
            continuous_inputs=continous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=embedding_expected_inputs,
            empty=config["empty_value"],
            std_layer=self.std_layer,
            # rotation_layer=rotation_layer,
            rotation_layer=None,
            padding_categorical_layer=self.categorical_padding,
            padding_continous_layer=self.continuous_padding,
        )

        self.transition_dense_1 = DenseBlock(input_nodes = self.input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize) # noqa
        self.resnet_block_1 = ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=eps, normalize=normalize)
        self.resnet_block_2 = ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=eps, normalize=normalize)
        self.resnet_block_3 = ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=eps, normalize=normalize)
        self.last_linear = torch.nn.Linear(config["nodes"], 3)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.last_linear(x)
        return x


class BNetDenseNet(torch.nn.Module):
    def __init__(self, continous_features, categorical_features, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(config=config, continous_features=continous_features, categorical_features=categorical_features)

    def init_layers(self, continous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        eps = 0.001
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        if config["mean"] is None and config["std"] is None:
            self.std_layer = StandardizeLayer(
                mean = torch.zeros(len(continous_features)),
                std = torch.ones(len(continous_features))
            )
        else:
            self.std_layer = StandardizeLayer(mean=config["mean"], std=config["std"])

        self.continuous_padding = PaddingLayer(padding_value=-4, mask_value=EMPTY_FLOAT)

        if config["categorical_padding_value"] is None:
            self.categorical_padding = None
        else:
            self.categorical_padding = PaddingLayer(padding_value=config["categorical_padding_value"], mask_value=EMPTY_INT)

        # self.rotation_layer = RotatePhiLayer(
        #     columns=list(map(str, continous_features)),
        #     ref_phi_columns=config["ref_phi_columns"],
        #     rotate_columns=config["rotate_columns"],
        # )
        self.input_layer = InputLayer(
            continuous_inputs=continous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=embedding_expected_inputs,
            empty=config["categorical_padding_value"],
            std_layer=self.std_layer,
            # rotation_layer=rotation_layer,
            rotation_layer=None,
            padding_categorical_layer=self.categorical_padding,
            padding_continous_layer=self.continuous_padding,
        )

        dense_config = {
            "skip_connection_init":config["skip_connection_init"],
            "freeze_skip_connection":config["freeze_skip_connection"],
            "activation_functions":"ELU",
            "eps":eps,
            "normalize":False,
        }

        self.transition_dense_1 = DenseBlock(input_nodes = self.input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize) # noqa
        self.dense_block_1 = DenseNetBlock(input_nodes = self.transition_dense_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_2 = DenseNetBlock(input_nodes = self.dense_block_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_3 = DenseNetBlock(input_nodes = self.dense_block_2.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_4 = DenseNetBlock(input_nodes = self.dense_block_3.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_5 = DenseNetBlock(input_nodes = self.dense_block_4.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, 3)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = self.dense_block_4(x)
        x = self.dense_block_5(x)
        x = self.last_linear(x)
        return x

class AddActFnToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.act_func = self._get_attr(torch.nn.modules.activation, act_fn)(dim=1)

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.model(categorical_inputs, continuous_inputs)
        x = self.act_func(x)
        return x
