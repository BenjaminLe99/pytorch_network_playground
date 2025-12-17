import torch
import os
from pathlib import Path
from models.create_model import AddActFnToModel
import inspect
from abc import abstractmethod

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

def torch_export_v2(model, name, fold, base_dir=None, add_softmax=True):
    DEVICE=torch.device("cpu")
    if add_softmax:
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

    if base_dir is None:
        base_dir = os.environ["MODELS_DIR"]

    dst = (Path(base_dir) / f"{name}_fold{fold}").with_suffix(".pt2")
    torch.export.save(exp, dst, pickle_protocol=4)
    return dst


class TorchExport():
    def __init__(self, model_inst, model_name, fold=0, device="cpu", save_dir=None):
        self.device = torch.device(device)
        self.model = model_inst
        self.save_dir = save_dir
        self.model_name = model_name
        self.fold = fold

    @abstractmethod
    def _input(self):
        """ Should return a tuple of tensors that can be used to run the model """
        pass

    def dst(self):
        if self.save_dir is None:
            save_dir = os.environ["MODELS_DIR"]
        return (Path(save_dir) / f"{self.model_name}_fold{self.fold}").with_suffix(".pt2")

    def export(self):
        model = self.model.to(self.device)
        model = model.eval()

        forward_signature = list(inspect.signature(model.forward).parameters.keys())
        _input = self._input()

        dim = torch.export.Dim("batch")
        dynamic_shapes = {forward_arg: {0: dim, 1: _inp_tensor.shape[-1]} for forward_arg, _inp_tensor in zip(forward_signature, _input)}

        exp = torch.export.export(
            model,
            args=_input,
            dynamic_shapes=dynamic_shapes,
        )

        torch.export.save(exp, self.dst(), pickle_protocol=4)
        print(f"successfully export to {self.dst()}")

class TorchExportFullModel(TorchExport):
    def _input(self):
        num_cat = len(model.categorical_features)
        num_cont = len(model.continous_features)

        categorical_input = torch.zeros(2, num_cat).to(torch.int32).to(self.device)
        continuous_inputs = torch.zeros(2, num_cont).to(torch.float32).to(self.device)
        return categorical_input, continuous_inputs


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
    import os

    p = argparse.ArgumentParser(description="Export model to torch-export (.pt2) with dynamic batch dim")
    p.add_argument("--model_path", "-m", required=True, help="Path to the model file (file or checkpoint) to load")
    p.add_argument("--fold", "-f", required=True, help="Fold number", default=0)

    args = p.parse_args()

    model_path = pathlib.Path(args.model_path)
    # if given path is a only a name search for name in enviroment dir
    if len(model_path.parts) == 1:
        model_path = pathlib.Path(os.environ["MODELS_DIR"]).with_stem(args.model_path).with_suffix(".pt2")

    model = torch.load(args.model_path, weights_only=False)
    model.eval()
    torch_export_v2(model, name=str(model_path), fold=args.fold)
    print("Done exporting")
