import torch

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



def run_exported_tensor_model(pt2_path, input_tensors):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(input_tensors)
    return scores
