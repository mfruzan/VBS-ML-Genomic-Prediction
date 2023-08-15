import torch.nn as nn


class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()

        self.fc1 = nn.Sequential(
             nn.Linear(17114, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            #nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),        )
        # self.fc1 = nn.Linear(17114, 2 * 16384)
        # self.fc2 = nn.Linear(2 * 16384, 4 * 16384)
        # # self.fc3 = nn.Linear(4 * 16384, 8 * 16384)
        # # self.fc4 = nn.Linear(8 * 16384, 16 * 16384)
        # self.fc3 = nn.Linear(4 * 16384, 2 * 16384)
        # self.fc4 = nn.Linear(2 * 16384, 16384)
        # self.fc5 = nn.Linear(16384, 8192)
        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 2048)
        # self.fc8 = nn.Linear(2048, 1024)
        # self.fc9 = nn.Linear(1024, 256)
        # self.fc9 = nn.Linear(256, 64)
        # self.fc9 = nn.Linear(64, 16)
        # self.fc9 = nn.Linear(16, 1)

    def forward(self, input):
        y = self.fc1(input)
        #y1 =self.fc2(input)

        #y =y +y1

        return y