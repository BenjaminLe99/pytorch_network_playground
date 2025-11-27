import torch

def torch_export(model, name, input_tensors):
    save_path = "/afs/desy.de/user/l/lebenjam/Master/neuralnetwork/mlmodels/test/"
    from pathlib import Path
    model = model.eval()

    categorical_input, continous_inputs = input_tensors

    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    # dim = torch.export.dynamic_shapes.Dim.AUTO
    # dynamic_shapes = {
    #     "input": ((dim, categorical_input.shape[-1]), (dim, continous_inputs.shape[-1]))
    # }

    dynamic_shapes = {
    "categorical_inputs": {0: torch.export.Dim("batch")},
    "continuous_inputs":  {0: torch.export.Dim("batch")},
    }

    exp = torch.export.export(
        model,
        args=(categorical_input, continous_inputs),
        dynamic_shapes=dynamic_shapes,
    )

    p = Path(save_path + f"{name}").with_suffix(".pt2")
    torch.export.save(exp, p, pickle_protocol=4)

def run_exported_tensor_model(pt2_path, cat, cont):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(cat,cont)
    return scores