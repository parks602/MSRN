import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        h1 = nn.Linear(4, 10)
        h2 = nn.Linear(10, 20)
        h3 = nn.Linear(20, 40)
        h4 = nn.Linear(40, 20)
        h5 = nn.Linear(20, 10)
        h6 = nn.Linear(10, 6)
        h7 = nn.Linear(6, 4)
        h8 = nn.Linear(4, 2)
        h9 = nn.Linear(2, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.LeakyReLU(),
            h2,
            nn.LeakyReLU(),
            h3,
            nn.LeakyReLU(),
            h4,
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            h5,
            nn.LeakyReLU(),
            h6,
            nn.LeakyReLU(),
            h7,
            nn.LeakyReLU(),
            h8,
            nn.LeakyReLU(),
            h9
        )
        #self.hidden = self.hidden.cuda()

    def forward(self, x):
        o = self.hidden(x)
        return o.view(-1)

