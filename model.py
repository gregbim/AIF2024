import torch
import torch.nn as nn
import torch.nn.functional as F
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5)    # inchannel de 1, outchannel de 1,8 filtres, noyau carré : 5
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5)      # inchannel de 1, 16 filtres, noyau carré de 5
        self.fc1 = nn.Linear(in_features =16*4*4,out_features=128)
        self.fc2 = nn.Linear(in_features =128,out_features=64)
        self.fc3 = nn.Linear(in_features =64,out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))       # First convolution followed by
        x = self.pool(x)              # a relu activation and a max pooling#
        x = F.relu(self.conv2(x))       #BMB : second convolution followed by
        x = self.pool(x)                #BMB : relu activation and pool
      
        x = self.flatten(x)
       
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        return x
   
if __name__ == "__main__":
    x = torch.rand(1,1,28,28)
    print(x.shape)
    model = MNISTNet()
    h = model(x)
    print('model',h.shape)