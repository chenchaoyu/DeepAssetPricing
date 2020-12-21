from torch import nn
import torch
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        """The lstm block used in both sdf_net and conditional_net

        Args:
            input_size (int): number of feature
            hidden_size (int): number of hidden unit
            num_layers (int): number of layers
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x):
        """forward function

        Args:
            x (Tensor of shape[seq_len, 1(batch_size==1), feature]): The macroeconomic states

        Output:
            hn of shape(num_layers * num_directions, batch, hidden_size)
        """

        h0 = torch.randn(self.num_layers, 1, self.hidden_size)
        c0 = torch.randn(self.num_layers, 1, self.hidden_size)
        out, (hn,cn) = self.rnn(x, (h0,c0))
        return hn
