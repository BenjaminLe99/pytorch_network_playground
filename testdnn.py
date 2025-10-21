import torch
import numpy as np
import matplotlib as plt

from src.loss import WeightedCrossEntropy, FocalLoss
from src.utils import (
embedding_expected_inputs, get_standardization_parameter, normalized_weight_decay
)
from src.optimizer import SAM

from src.models.layers import (
    InputLayer, StandardizeLayer, ResNetPreactivationBlock, DenseBlock, PaddingLayer, RotatePhiLayer,
)
from src.models.create_model import init_layers
from src.data.extract_features import (categorical_features, continous_features)

model = init_layers(categorical_features, continous_features, )

def torch_export(model, dst_path, input_tensors):
    from pathlib import Path
    model.eval()

    cat, con = input_tensors

    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    dim = torch.export.dynamic_shapes.Dim.AUTO
    dynamic_shapes = {
        "input": ((dim, cat.shape[-1]), (dim, con[-1]))
    }

    exp = torch.export.export(
        model,
        args=((categorical_input, continous_inputs),),
        dynamic_shapes=dynamic_shapes,
    )

    p = Path(f"{dst_path}").with_suffix(".pt2")
    torch.export.save(exp, p)


def run_exported_tensor_model(pt2_path, input_tensors):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(input_tensors)
    return scores