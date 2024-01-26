from torch import nn

class xrayCNNv0(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # L1
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - 1)
        )

        # L2
        self.layer2 = nn.Sequential(
            nn.Conv3d(24, 56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - 1)
        )

        # L3
        self.layer3 = nn.Sequential(
            nn.Conv3d(56, 120, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=1 - 1)
        )

        self.fc1 = nn.Linear(4 * 4 * 120, 627, bias=True)

        nn.init.xavier_uniform(self.fc1.weight) ## Need more deep dive in this function
        
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - 1))
        
        # L5 Final FC 627 inputs -> 12 outputs
        self.fc2 = nn.Linear(627, 12, bias=True)

        # initialize parameters
        nn.init.xavier_uniform_(self.fc2.weight) 

    def forward(self, y):
        output = self.layer1(y)
        output = self.layer2(output)
        output = self.layer3(output)

         # Flatten them for FC
        output = output.view(output.size(0), -1)  
        output = self.fc1(output)
        output = self.fc2(output)
        
        return output