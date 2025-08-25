import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolutionnal_layer(nn.Module):
    def __init__ (self):
        super().__init__()
        self.con2d = nn.Conv2d(in_channels=19,out_channels=256,kernel_size=3,padding=1)
        self.batch = nn.BatchNorm2d(256)
    
    def forward(self, x):
        # input (B,19,15,15)
        x = self.con2d(x)
        x = self.batch(x)
        x = F.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.batch_1 = nn.BatchNorm2d(256)

        self.conv2d_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.batch_2 = nn.BatchNorm2d(256)
    
    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.batch_1(out)
        out = F.relu(out)

        out = self.conv2d_2(out)
        out = self.batch_2(out)
        # skip conection
        return F.relu( x + out )

class Extract_policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=256,out_channels=2,kernel_size=1,padding=0)
        self.batch = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 15 * 15, 225)
    
    def forward(self, x):
        # input (B,256,15,15)
        x = self.depthwise(x)
        x = self.batch(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class AlphaZero(nn.Module):
    def __init__(self):
        super(AlphaZero, self).__init__()
        self.conv_layer = Convolutionnal_layer()
        self.res_block = nn.Sequential(*[ResBlock() for _ in range(19)])
        self.policy_head = Extract_policy()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        self.loss_function = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.conv_layer(x)
        x = self.res_block(x)
        x = self.policy_head(x)
        return x

    def startTraining(self, dataPack_loader, epochs = 2000):
        self.train()
        for epoch in range(epochs):
            num_batches = 0
            for input, label in dataPack_loader:
                self.optimizer.zero_grad()
                outputs = self(input)
                loss = self.loss_function(outputs, label)
                loss.backward()
                self.optimizer.step()
                num_batches += 1
                if num_batches % 10 == 0:
                    print(f"Batch [{num_batches}], Loss: {loss.item():.4f}")
                if loss.item() < 0.2:
                    print(f"Early stopping at epoch {epoch}, batch {num_batches} with loss {loss.item():.4f}")
                    torch.save(self.state_dict(), 'model_save.pth')
                    return
            print(f"Epoch [{epoch + 1}], Loss: {loss.item():.4f}")
            if epoch % 2 == 0:
                torch.save(self.state_dict(), 'model_save.pth')

                
