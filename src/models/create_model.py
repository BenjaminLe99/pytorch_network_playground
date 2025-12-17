import torch

from models.layers import (
    InputLayer, StandardizeLayer, ResNetPreactivationBlock, DenseBlock, PaddingLayer, RotatePhiLayer,
    DenseNetBlock, LBN_DNN, TemperaturCalibrationLayer
)
from utils.utils import EMPTY_INT, EMPTY_FLOAT
from data.features import expected_embedding_inputs



class BNet(torch.nn.Module):
    def __init__(self, continous_features, categorical_features, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(config=config, continous_features=continous_features, categorical_features=categorical_features)

    def init_layers(self, continous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        # eps = 0.5e-5

        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        std_layer = StandardizeLayer(mean=config["mean"], std=config["std"])

        continuous_padding = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
        categorical_padding = PaddingLayer(padding_value=config["empty_value"], mask_value=EMPTY_INT)

        if config["enable_rotation"]:
            rotation_layer = RotatePhiLayer(
                columns=list(map(str, continous_features)),
                ref_phi_columns=config["ref_phi_columns"],
                rotate_columns=config["rotate_columns"],
            )
        else:
            rotation_layer = None

        self.input_layer = InputLayer(
            continuous_inputs=continous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=expected_embedding_inputs(),
            empty=config["empty_value"],
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continous_layer=continuous_padding,
        )

        self.transition_dense_1 = DenseBlock(input_nodes = self.input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=config["batch_norm_eps"], normalize=normalize) # noqa
        self.resnet_block_1 = ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=config["batch_norm_eps"], normalize=normalize)
        self.resnet_block_2 = ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=config["batch_norm_eps"], normalize=normalize)
        self.resnet_block_3 = ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=config["batch_norm_eps"], normalize=normalize)
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
        self.categorical_features = categorical_features
        self.continous_features = continous_features

    def init_layers(self, continous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        eps = 0.001
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        if config["mean"] is None and config["std"] is None:
            std_layer = StandardizeLayer(
                mean = torch.zeros(len(continous_features)),
                std = torch.ones(len(continous_features))
            )
        else:
            std_layer = StandardizeLayer(mean=config["mean"], std=config["std"])


        if config["continous_padding_value"] is None:
            continuous_padding = None
        else:
            continuous_padding = PaddingLayer(padding_value=-4, mask_value=EMPTY_FLOAT)


        if config["categorical_padding_value"] is None:
            categorical_padding = None
        else:
            categorical_padding = PaddingLayer(padding_value=config["categorical_padding_value"], mask_value=EMPTY_INT)


        if config["enable_rotation"]:
            rotation_layer = RotatePhiLayer(
                columns=list(map(str, continous_features)),
                ref_phi_columns=config["ref_phi_columns"],
                rotate_columns=config["rotate_columns"],
            )
        else:
            rotation_layer = None

        self.input_layer = InputLayer(
            continuous_inputs=continous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=expected_embedding_inputs(),
            empty=config["categorical_padding_value"],
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continous_layer=continuous_padding,
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

class BNetLBNDenseNet(torch.nn.Module):
    def __init__(self, continous_features, categorical_features, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(config=config, continous_features=continous_features, categorical_features=categorical_features)
        self.categorical_features = categorical_features
        self.continous_features = continous_features

    def init_layers(self, continous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        eps = 0.001
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        if config["mean"] is None and config["std"] is None:
            std_layer = StandardizeLayer(
                mean = torch.zeros(len(continous_features)),
                std = torch.ones(len(continous_features))
            )
        else:
            std_layer = StandardizeLayer(mean=config["mean"], std=config["std"])


        if config["continous_padding_value"] is None:
            continuous_padding = None
        else:
            continuous_padding = PaddingLayer(padding_value=-4, mask_value=EMPTY_FLOAT)


        if config["categorical_padding_value"] is None:
            categorical_padding = None
        else:
            categorical_padding = PaddingLayer(padding_value=config["categorical_padding_value"], mask_value=EMPTY_INT)

        if config["enable_rotation"]:
            rotation_layer = RotatePhiLayer(
                columns=list(map(str, continous_features)),
                ref_phi_columns=config["ref_phi_columns"],
                rotate_columns=config["rotate_columns"],
            )
        else:
            rotation_layer = None

        self.input_layer = InputLayer(
            continuous_inputs=continous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=expected_embedding_inputs(),
            empty=config["categorical_padding_value"],
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continous_layer=continuous_padding,
        )

        self.lbn = LBN_DNN(
            continous_features = continous_features,
            M  = config["LBN_M"],
            N  = 5,
            weight_init_scale = 1.0,
            clip_weights = False,
            eps = 1.0e-5,
        )

        dense_config = {
            "skip_connection_init":config["skip_connection_init"],
            "freeze_skip_connection":config["freeze_skip_connection"],
            "activation_functions":"ELU",
            "eps":eps,
            "normalize":False,
        }

        self.transition_dense_1 = DenseBlock(input_nodes = self.input_layer.ndim + self.lbn.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize) # noqa
        self.dense_block_1 = DenseNetBlock(input_nodes = self.transition_dense_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_2 = DenseNetBlock(input_nodes = self.dense_block_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_3 = DenseNetBlock(input_nodes = self.dense_block_2.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_4 = DenseNetBlock(input_nodes = self.dense_block_3.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_5 = DenseNetBlock(input_nodes = self.dense_block_4.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, 3)

    def forward(self, categorical_inputs, continuous_inputs):
        # preprocessing
        x = torch.concatenate(
            (self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs),
            self.lbn(continuous_inputs)),
            axis=1
        )

        # dnn
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
        self.categorical_features = model.categorical_features
        self.continous_features = model.continous_features

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

# TODO finished calib layer
class AddCalibrationToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.categorical_features = model.categorical_features
        self.continous_features = model.continous_features
        self.calibration_layer = TemperaturCalibrationLayer()


    def calibrate(self, sampler, num_samples):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.calibration_layer.temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

        # for i in range(num_samples):
        #     sampler.get
        # # Create tensors
        # logits_list = torch.cat(logits_list).to(device)
        # labels_list = torch.cat(labels_list).to(device)

        # def _eval():
        # loss = criterion(T_scaling(logits_list, temperature), labels_list)
        # loss.backward()
        # return loss

        # optimizer.step(_eval)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.model(categorical_inputs, continuous_inputs)
        x = self.act_func(x)
        return x
