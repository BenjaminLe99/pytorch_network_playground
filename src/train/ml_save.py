import torch

save_path = "/afs/desy.de/user/l/lebenjam/Master/neuralnetwork/mlmodels/test/"


def torch_export(model, name, input_tensors):
     from pathlib import Path
     model.eval()

     con, cat, _ = input_tensors

     # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
     dim = torch.export.dynamic_shapes.Dim.AUTO
     dynamic_shapes = {
         "input": ((dim, cat.shape[-1]), (dim, con.shape[-1]))
     }

     exp = torch.export.export(
         model,
         args=((cat.to(torch.int32), con.to(torch.float32)),),
         dynamic_shapes=dynamic_shapes,
     )

     p = Path(save_path + f"{name}").with_suffix(".pt2")
     torch.export.save(exp, p)

def run_exported_tensor_model(pt2_ptrath, input_tensors):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(input_tensors)
    return scores