import torch
from models import layers

class DenseNetwork(torch.nn.Module):
    def __init__(
            self,
            input_nodes: float,
            hidden_nodes: list,
            output_nodes: float,
            activation: str,
            eps: float = 1e-5,
            normalize: bool = True,
        ):
        super().__init__()
        self.activation = activation

        # Hidden layers (super-dense connections)
        layer_input_dims = [input_nodes]  # first hidden layer takes input_dim
        for h in hidden_nodes[:-1]:
            layer_input_dims.append(layer_input_dims[-1] + hidden_nodes[len(layer_input_dims)-1])

        self.hidden_layers = torch.nn.ModuleList([
            layers.DenseBlock(layer_input_dims[i], hidden_nodes[i], activation, eps, normalize) for i in range(len(hidden_nodes))
        ])

        # Final layer sees input to first hidden + all hidden outputs
        total_hidden = sum(hidden_nodes)
        self.output_layer = torch.nn.Linear(input_nodes + total_hidden, output_nodes)

    def forward(self, x):
        prev_outputs = [x]  # x comes from your input layer
        for layer in self.hidden_layers:
            input = torch.cat(prev_outputs, dim=1)
            output = layer(input)
            prev_outputs.append(output)

        final_input = torch.cat(prev_outputs, dim=1)
        return self.output_layer(final_input)