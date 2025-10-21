import torch
import torch.nn as nn
import torch.nn.functional as F

# we are giving it 84*84 picture for input to start
class NeuralNetworkForQValues(nn.Module):
    def __init__(self, output, input_channels=4, input_height=84, input_width=84, freeze=False):
        super().__init__()

        # Select device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)


        self.CNN_layers = nn.Sequential(
            # we ArE going throgh the input of size [4,84,84] 4 channel come from stacking
            # with 8*8 matrix, step size 4, it will output output of size [32, 20,20]
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # now this input is [32,20,20], W_out would be [64,9,9]
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # input is [64,9,9], w_out= [64,7,7]
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.CNN_layers(sample_input)
            self.flattened_size = conv_out.view(1, -1).size(1)


        self.linearLayer = nn.Sequential(
            nn.Flatten(),
            # the output coming from CNN is [64,7,7] before we flattened it
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, output),
        )

        # Combine both CNN and linear layers into one network
        self.network = nn.Sequential(
            self.CNN_layers,
            self.linearLayer
        )

        # Optionally freeze parameters (for target network)
        if freeze:
            for p in self.network.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.network(x)
