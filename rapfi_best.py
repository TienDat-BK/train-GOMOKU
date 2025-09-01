import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DirConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, direction, kernel_size=3, padding=0):
        super().__init__()
        self.weight_raw = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        nn.init.kaiming_uniform_(self.weight_raw, a=math.sqrt(5))
        self.padding = padding

        self.direction = direction
        self.bias = nn.Parameter(torch.zeros(out_channels))  # Thêm bias nếu cần

        # Tạo mask và đăng ký làm buffer
        mask = self._create_directional_mask(direction, out_channels, in_channels, kernel_size)
        self.register_buffer('mask', mask)

    def _create_directional_mask(self, direction, out_ch, in_ch, k):
        m = torch.zeros((out_ch, in_ch, k, k), dtype=torch.float32)
        center = k // 2

        if direction == 'horizontal':
            m[:, :, center, :] = 1.0
        elif direction == 'vertical':
            m[:, :, :, center] = 1.0
        elif direction == 'diagonal':
            for i in range(k):
                m[:, :, i, i] = 1.0
        elif direction == 'anti-diagonal':
            for i in range(k):
                m[:, :, i, k - 1 - i] = 1.0
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return m


    def forward(self, x):
        # mask đã là buffer, nằm trong state_dict và chuyển device tự động
        weight = self.weight_raw * self.mask
        return F.conv2d(x, weight, bias=self.bias, padding=self.padding)

    
class ResBlock(nn.Module):
    def __init__(self, in_channels, directions, kernel_size=3, padding=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.DirConv2d = DirConv2d(in_channels, in_channels, directions, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.DirConv2d(x)
        out = self.bn1(out)
        out = F.silu(out)
        out = self.conv2d(out)
        out = self.bn2(out)
        out = F.silu(out)
        return out + x

class OutputBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv2d_2 = nn.Conv2d(in_channels,in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv2d_1(x)
        out = F.silu(out)
        out = self.conv2d_2(out)
        out = F.silu(out)
        return out + x

class ExtractDirectionalFeatures(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.listDirection = ['horizontal', 'vertical', 'diagonal', 'anti-diagonal']
        self.listModels = nn.ModuleList()  

        for direction in self.listDirection:
            model = nn.Sequential(
                DirConv2d(in_channels, mid_channels, direction, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(mid_channels),
                ResBlock(mid_channels, direction, kernel_size=kernel_size, padding=padding),
                ResBlock(mid_channels, direction, kernel_size=kernel_size, padding=padding),
                ResBlock(mid_channels, direction, kernel_size=kernel_size, padding=padding),   
                ResBlock(mid_channels, direction, kernel_size=kernel_size, padding=padding),
                OutputBlock(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
            )
            self.listModels.append(model)

    def forward(self, x):
        outputs = [model(x) for model in self.listModels]
        return torch.cat(outputs, dim=1)  # (B, 4*out_channels, H, W)

class IncrementalModel(nn.Module):
    def __init__(self,in_channels=32):
        super().__init__()
        self.in_channels = in_channels
        # half = in_channels // 2
        # self.depthwise_conv = nn.Conv2d(half, half, kernel_size=3, padding=1, groups=half)

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, x):
        # input x: (B, C, H, W)
        # x1,x2 = x.chunk(2, dim=1)  # chia 2 channel đầu vào
        # x1 = self.depthwise_conv(x1)
        # return torch.cat([x1, x2], dim=1)

        x = self.depthwise_conv(x)  # (B, C, H, W)
        return x  # (B, C, H, W)

class ExtractPolicy_2(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.Conv2d_x5_x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.FinalConv = nn.Conv2d(in_channels, 4, kernel_size=1, padding=0)
        self.FinalLinear = nn.Sequential(
            # nn.Linear(4 * 15 * 15, 8 * 15 * 15),
            # nn.ReLU(),
            nn.Linear(4 * 15 * 15, 2 * 15 * 15),
            nn.ReLU(),
            nn.Linear(2 * 15 * 15, 15 * 15)  # output (B, C*H*W)

        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.Conv2d_x5_x3(x)
        x = self.FinalConv(x)
        x = x.reshape(x.size(0), -1)  # (B, C*H*W)
        x = self.FinalLinear(x)  # (B, C*H*W)
        x = x.view(x.size(0), 1, 15, 15)
        return x # logits output (B, 1, 15, 15)


class ExtractPolicy(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.AvgPool_and_Linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),  
            nn.ReLU(),
            nn.Linear(in_channels, in_channels) # output (B, in_channels, 1, 1)
        )
        self.FinalConv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)

    def forward(self, x):
        # x: (B, C, H, W)
        x2 = self.AvgPool_and_Linear(x)
        x2 = x2.reshape(x2.size(0), self.in_channels, 1, 1) # reshape to (B, in_channels, 1, 1)

        out = self.FinalConv( F.relu(x2 + x) ) # (B, 1, 15, 15)
        return out  

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Input x: (B, 2, 15, 15)
        self.extractDirFeatures = ExtractDirectionalFeatures(in_channels=2, mid_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.incrementalModel = IncrementalModel(in_channels=16*4)
        self.extractPolicy = ExtractPolicy_2(in_channels=16*4)

    def forward(self, x):
        # x: (B, 2, 15, 15)
        x = self.extractDirFeatures(x)   # (B, 64, 15, 15)
        # x = self.incrementalModel(x)      # (B, 64, 15, 15)
        x = self.extractPolicy(x)         # (B, 1, 15, 15)
        x = x.view(x.size(0), -1)         # (B, 225) — chưa softmax

        return x                        # logits output

    def train_model(self, dataloader):
            # target_index: (B,) — chỉ số ô đúng (0~224)
            loss_func = nn.CrossEntropyLoss()
            self.train()
            optimizer = torch.optim.Adam([
                {"params": self.extractDirFeatures.parameters(), "lr": 0.002},
                # {"params": self.incrementalModel.parameters(), "lr": 0.002},
                {"params": self.extractPolicy.parameters(), "lr": 0.002, "weight_decay" : 1e-4},   # phần policy nhẹ hơn
            ])
            

            cnt_mini_batch = 0

            for epoch in range(4000):
                print(f"Epoch {epoch} processing...")
                for data_input, target_probs in dataloader:

                    optimizer.zero_grad()
                    logits = self.forward(data_input)           # (B, 225)
                    loss = loss_func(logits,target_probs)

                    # target: (B,)
                    if loss < 0.1:
                        print(f"Early stop at epoch {epoch} with loss {loss.item():.4f}")
                        torch.save(self.state_dict(), f"model_rapfi_save_best.pth")
                        print(f"Model saved at epoch {epoch}")
                        return
                    loss.backward()
                    optimizer.step()

                    cnt_mini_batch += 1
                    if cnt_mini_batch  == 10:
                        print(f"mini-batch loss [{cnt_mini_batch}]: {loss.item():.4f}")

                cnt_mini_batch = 0
                if epoch % 1 == 0:
                    torch.save(self.state_dict(), f"model_rapfi_save_best.pth")
                    print(f"Model saved at epoch {epoch}")
                
            super().train(mode=False)