import torch
import torch.nn as nn


class MultiHeadMLP(nn.Module):
    def __init__(self, cfg, device):
        """
        input_window: number of heads
        input_dim: dimension that gets fead to each head
        output_window: output dimension
        hidden_dim: dimension of each head
        """

        super(MultiHeadMLP, self).__init__()

        self.cfg = cfg
        self.device = device
        self.input_window = cfg['input_window']
        self.output_window = cfg['output_window']
        cfg = cfg['mlp']
        self.input_dim = 1 + cfg['token_dim']
        self.hidden_dim = cfg['hidden_dim']

        if cfg['activation'] == 'relu':
            self.activation = nn.ReLU()
        elif cfg['activation'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif cfg['activation'] == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")
        self.input_layer = nn.Linear(
            self.input_dim, self.hidden_dim).to(self.device)
        self.output_layer = nn.Linear(
            self.input_window * self.hidden_dim, self.output_window).to(self.device)
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, input_time, input_embedding, return_hidden_state=False):
        if len(input_time.shape) == 2:
            input_time = input_time.unsqueeze(-1)
        if len(input_embedding.shape) == 2:
            input_embedding = input_embedding.unsqueeze(1)

        x = torch.cat((input_time, input_embedding), dim=-1).to(self.device)
        hidden_state = self.activation(self.input_layer(x))
        output = self.output_layer(self.dropout(
            hidden_state.view(hidden_state.shape[0], -1)))

        if return_hidden_state:
            return output, hidden_state
        return output
