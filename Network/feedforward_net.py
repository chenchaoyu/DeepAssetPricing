import torch.nn as nn


class FeedforwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """init function

        Args:
            input_size (int): input size
            hidden_size (list of length num_layers): for example [10,20,10] for three hidden layers 
            num_layers (int): number of layers
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.module_list = []
        for i in range(self.num_layers):
            if(i == 0):
                self.module_list.append(nn.Linear(input_size, hidden_size[i]))
                self.module_list.append(nn.ReLU())
            else:
                self.module_list.append(
                    nn.Linear(hidden_size[i-1], hidden_size[i]))
                self.module_list.append(nn.ReLU())
        self.module_list.append(nn.Linear(hidden_size[num_layers-1], 1))
        self.net = nn.Sequential(*self.module_list)

    def forward(self, x):
        w = self.net(x)
        return w
