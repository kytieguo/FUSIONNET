import torch
from torch import nn


def get_activation(activation, hidden_units=None):
	if isinstance(activation, str):
		if activation.lower() in ["prelu", "dice"]:
			assert type(hidden_units) == int
		if activation.lower() == "relu":
			return nn.ReLU()
		elif activation.lower() == "sigmoid":
			return nn.Sigmoid()
		elif activation.lower() == "tanh":
			return nn.Tanh()
		elif activation.lower() == "softmax":
			return nn.Softmax(dim=-1)
		else:
			return getattr(nn, activation)()
	elif isinstance(activation, list):
		if hidden_units is not None:
			assert len(activation) == len(hidden_units)
			return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
		else:
			return [get_activation(act) for act in activation]
	return activation


class MLP_Block(nn.Module):
	def __init__(self,
				 input_dim,
				 hidden_units=[],
				 hidden_activations="ReLU",
				 output_dim=None,
				 output_activation=None,
				 dropout_rates=0.0,
				 batch_norm=False,
				 layer_norm=False,
				 norm_before_activation=True,
				 use_bias=True):
		super(MLP_Block, self).__init__()
		dense_layers = []
		if not isinstance(dropout_rates, list):
			dropout_rates = [dropout_rates] * len(hidden_units)
		if not isinstance(hidden_activations, list):
			hidden_activations = [hidden_activations] * len(hidden_units)
		hidden_activations = get_activation(hidden_activations, hidden_units)
		hidden_units = [input_dim] + hidden_units
		for idx in range(len(hidden_units) - 1):
			dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
			if norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if hidden_activations[idx]:
				dense_layers.append(hidden_activations[idx])
			if not norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if dropout_rates[idx] > 0:
				dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
		if output_dim is not None:
			dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
		if output_activation is not None:
			dense_layers.append(get_activation(output_activation))
		self.mlp = nn.Sequential(*dense_layers)  # * used to unpack list

	def forward(self, inputs):
		return self.mlp(inputs)




class CrossNetV2(nn.Module):
	def __init__(self, input_dim, num_layers, w=0.2):
		super(CrossNetV2, self).__init__()
		self.num_layers = num_layers
		self.w = w
		self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
										  for _ in range(self.num_layers))
		self.attention = nn.Parameter(torch.ones(num_layers, 1, 1) / num_layers)

	def forward(self, X_0):
		X_i = X_0
		for i in range(self.num_layers):
			X_i = self.w * X_i + (1 - self.w) * X_0 * self.cross_layers[i](X_i)
		return X_i


