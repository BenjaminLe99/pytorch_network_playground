import torch

def torch_export(model, name, input_tensors):
    save_path = "/afs/desy.de/user/l/lebenjam/Master/neuralnetwork/mlmodels/test/"
    from pathlib import Path
    from mymodels.mylayers import AddActFnToModel
    model = AddActFnToModel(model, "softmax")
    model = model.eval()

    categorical_input, continuous_inputs = input_tensors

    continuous_inputs = continuous_inputs.to(torch.float32)
    categorical_input = categorical_input.to(torch.int32)

    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    if categorical_input.shape[0] == 1:
        continuous_inputs = torch.concatenate((continuous_inputs, continuous_inputs))
        categorical_input = torch.concatenate((categorical_input, categorical_input))

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

    p = Path(save_path + f"{name}").with_suffix(".pt2")
    torch.export.save(exp, p, pickle_protocol=4)
    print(p)

def run_exported_tensor_model(pt2_path, cat, cont):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(cat,cont)
    return scores