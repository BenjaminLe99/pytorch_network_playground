import torch

def torch_export(model, dst_path, input_tensors):
    from pathlib import Path
    model = model.eval()

    categorical_input, continous_inputs = input_tensors

    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    dim = torch.export.dynamic_shapes.Dim.AUTO
    dynamic_shapes = {
        "input": ((dim, categorical_input.shape[-1]), (dim, continous_inputs.shape[-1]))
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
