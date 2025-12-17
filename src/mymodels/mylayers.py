import torch
from models import layers

class DenseNetwork(torch.nn.Module):
    def __init__(
            self,
            input_layer,
            input_nodes: float,
            hidden_nodes: list,
            output_nodes: float,
            activation: str,
            eps: float = 1e-5,
            normalize: bool = True,
        ):
        super().__init__()
        self.input_layer = input_layer
        self.activation = activation
        
        # First hidden layer takes only input_dim
        self.first_hidden = layers.DenseBlock(input_nodes, hidden_nodes[0], activation, eps, normalize)

        # Subsequent layers: each gets all previous hidden outputs
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(1, len(hidden_nodes)):
            prev_total = sum(hidden_nodes[:i])
            self.hidden_layers.append(layers.DenseBlock(prev_total, hidden_nodes[i], activation, eps, normalize))

        # Final output layer: sees all hidden outputs concatenated
        total_hidden = sum(hidden_nodes)
        self.last_linear = torch.nn.Linear(total_hidden, output_nodes)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs, continuous_inputs)
        # First hidden layer (normal connection)
        layers = [self.first_hidden(x)]

        # Dense connections only among hidden layers
        for i, layer in enumerate(self.hidden_layers):
            input = torch.cat(layers, dim=1)
            output = layer(input)
            layers.append(output)

        # Output sees all hidden layers concatenated
        final_input = torch.cat(layers, dim=1)
        return self.last_linear(final_input)  # raw logits (no activation) 
    
class AddActFnToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        # self.categorical_features = model.categorical_features
        # self.continous_features = model.continous_features

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