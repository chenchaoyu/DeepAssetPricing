import torch
import torch.nn as nn
from Network.lstm import LSTM
from Network.feedforward_net import FeedforwardNet

class ConditionalNet(nn.Module):
    def __init__(self, macro_size, char_size, 
                 lstm_hidden_size, lstm_num_layers, 
                 ff_hidden_size, ff_num_layers):
        super().__init__()
        self.MomentRNN = LSTM(macro_size, lstm_hidden_size, lstm_num_layers)
        ff_input_size = lstm_hidden_size + char_size
        self.FeedForward = FeedforwardNet(ff_input_size, ff_hidden_size, ff_num_layers)

    def forward(self, macro_x, char_x, ret_x):
        """forward function

        Args:
            macro_x (tensor of shape[seq_len, input_size]): macroeconomic data from t1 to tn
            char_x (tensor of shape[input_size]): char data for a stock
        """
        macro_x = torch.unsqueeze(macro_x,1)
        hidden_g = self.MomentRNN(macro_x)
        hidden_g = torch.squeeze(hidden_g,0)
        hidden_g = hidden_g.repeat(char_x.shape[0],1)
        hidden_g_char = torch.cat((hidden_g, char_x), 1)
        out = self.FeedForward(hidden_g_char)
        out = out.squeeze(1)
        return torch.mul(ret_x, out)
       