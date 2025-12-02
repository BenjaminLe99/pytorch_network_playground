import torch
import os
from pathlib import Path
from models.create_model import AddActFnToModel


def torch_export(model, dst_path, input_tensors):
    from pathlib import Path
    from models.create_model import AddActFnToModel
    model = AddActFnToModel(model, "softmax")
    model = model.eval()

    categorical_input, continuous_inputs = input_tensors

    continuous_inputs = continuous_inputs.to(torch.float32)
    categorical_input = categorical_input.to(torch.int32)

    # batch size 1 is an edge case and overwrites dynamic batch size, thus, inflate input to prevent this
    if categorical_input.shape[0] == 1:
        continuous_inputs = torch.concatenate((continuous_inputs, continuous_inputs))
        categorical_input = torch.concatenate((categorical_input, categorical_input))

    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    # dim = torch.export.dynamic_shapes.Dim.AUTO
    dim = torch.export.Dim("batch")

    dynamic_shapes = {
        # "categorical_inputs": (dim, categorical_input.shape[-1]),
        # "continuous_inputs" : (dim, continuous_inputs.shape[-1]),
        "categorical_inputs": {0:dim, 1:categorical_input.shape[-1]},
        "continuous_inputs" : {0:dim, 1:continuous_inputs.shape[-1]},
    }

    exp = torch.export.export(
        model,
        args=(categorical_input, continuous_inputs),
        dynamic_shapes=dynamic_shapes,
    )

    p = Path(f"{dst_path}").with_suffix(".pt2")
    torch.export.save(exp, p, pickle_protocol=4)

def torch_export_v2(model, name, fold):
    DEVICE=torch.device("cpu")
    model = AddActFnToModel(model, "softmax")
    model = model.to(DEVICE)
    model = model.eval()
    # create dummy features from shape
    # batch size 1 is an edge case and overwrites dynamic batch size, thus, inflate input to prevent this
    num_cat = len(model.categorical_features)
    num_cont = len(model.continous_features)

    categorical_input = torch.zeros(2, num_cat).to(torch.int32).to(DEVICE)
    continuous_inputs = torch.zeros(2, num_cont).to(torch.float32).to(DEVICE)


    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    # dim = torch.export.dynamic_shapes.Dim.AUTO
    dim = torch.export.Dim("batch")

    dynamic_shapes = {
        "categorical_inputs": {0:dim, 1:categorical_input.shape[-1]},
        "continuous_inputs" : {0:dim, 1:continuous_inputs.shape[-1]},
    }

    exp = torch.export.export(
        model,
        args=(categorical_input, continuous_inputs),
        dynamic_shapes=dynamic_shapes,
    )

    base_dir = os.environ["MODELS_DIR"]
    dst = (Path(base_dir) / f"{name}_fold{fold}").with_suffix(".pt2")
    torch.export.save(exp, dst, pickle_protocol=4)

def torch_save(model, name, fold):
    base_dir = os.environ["MODELS_DIR"]
    dst = (Path(base_dir) / f"{name}_fold{fold}").with_suffix(".pt")
    torch.save(model, dst)


def run_exported_tensor_model(pt2_path, cat, cont):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(cat, cont)
    return scores


if __name__ == "__main__":
    import pathlib
    import argparse

    p = argparse.ArgumentParser(description="Export model to torch-export (.pt2) with dynamic batch dim")
    p.add_argument("--model_path", "-m", required=True, help="Path to the model file (file or checkpoint) to load")
    p.add_argument("--fold", "-f", required=True, help="Fold number", default=0)

    args = p.parse_args()

    model = torch.load(args.model_path, weights_only=False)
    model.eval()
    torch_export_v2(model, name=pathlib.Path(args.model_path).stem, fold=args.fold)

    from IPython import embed; embed(header="string - 94 in export.py ")
