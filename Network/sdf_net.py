from Network.lstm import LSTM
from Network.feedforward_net import FeedforwardNet
import torch.nn as nn 
import torch
class SDFNet(nn.Module):

    def __init__(self, macro_size, char_size, 
                 lstm_hidden_size, lstm_num_layers, 
                 ff_hidden_size, ff_num_layers):
        
        super().__init__()
        self.StateRNN = LSTM(macro_size, lstm_hidden_size, lstm_num_layers)
        ff_input_size = lstm_hidden_size + char_size
        self.FeedForward = FeedforwardNet(ff_input_size, ff_hidden_size, ff_num_layers)

    def forward(self, macro_x, char_x, ret_x):
        """forward function

        Args:
            macro_x (tensor of shape[seq_len, input_size]): macroeconomic data from t1 to tn
            char_x (tensor of shape[stock_num, input_size]): char data for a stock
            ret_x (tensor of shape[stock_num]): char data for a stock
        """

        macro_x = torch.unsqueeze(macro_x,1)
        hidden = self.StateRNN(macro_x)
        hidden = torch.squeeze(hidden,0)
        hidden = hidden.repeat(char_x.shape[0],1)
        hidden_char = torch.cat((hidden, char_x), 1)
        w = self.FeedForward(hidden_char)

        w = w/(torch.mean(w)*char_x.shape[0])
        w = torch.squeeze(w,1)
        sdf = 1-torch.dot(ret_x,w)
        return sdf
        
        
        