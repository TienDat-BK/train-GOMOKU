
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda'

class DirConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, direction, kernel_size=3, padding=0):
        super().__init__()
        self.weight_raw = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        nn.init.kaiming_uniform_(self.weight_raw, a=math.sqrt(5))
        self.padding = padding

        self.direction = direction
        self.bias = nn.Parameter(torch.zeros(out_channels))  # Thêm bias 

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

class ResBlock_dir(nn.Module):
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
                ResBlock_dir(mid_channels, direction, kernel_size=kernel_size, padding=padding),
                ResBlock_dir(mid_channels, direction, kernel_size=kernel_size, padding=padding),
                ResBlock_dir(mid_channels, direction, kernel_size=kernel_size, padding=padding),   
                ResBlock_dir(mid_channels, direction, kernel_size=kernel_size, padding=padding),
                OutputBlock(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
            )
            self.listModels.append(model)

    def forward(self, x):
        outputs = [model(x) for model in self.listModels]
        return outputs[0] + outputs[1] + outputs[2] + outputs[3] # (B, out_channels, H, W)

class ResBlock(nn.Module):
    def __init__(self,in_out_channel = 256):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_out_channel,out_channels=in_out_channel,kernel_size=3,padding=1)
        self.batch_1 = nn.BatchNorm2d(in_out_channel)

        self.conv2d_2 = nn.Conv2d(in_channels=in_out_channel,out_channels=in_out_channel,kernel_size=3,padding=1)
        self.batch_2 = nn.BatchNorm2d(in_out_channel)
    
    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.batch_1(out)
        out = F.relu(out)

        out = self.conv2d_2(out)
        out = self.batch_2(out)
        # skip conection
        return F.relu( x + out )

class IncrementalModel(nn.Module):
    def __init__(self,in_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.half = in_channels // 2

        self.depthWise = nn.Conv2d(in_channels=self.half,out_channels=self.half,kernel_size=3,padding=1,groups=self.half)
    def forward(self, x):
        # input (B,C,15,15)
        first = x[:,:self.half, :, :]
        second = x[:, self.half:, :, :]

        x = torch.concat([self.depthWise(first), second], dim=1)
        return x  # (B, C, H, W)

class DynamicPointwise(nn.Module):
    def __init__(self, in_ch=32, out_ch=16):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.extract_WeightBias = nn.Sequential(
            nn.Linear(in_features=in_ch, out_features=2*in_ch),
            nn.ReLU(),
            nn.Linear(in_features=2*in_ch, out_features=(out_ch*in_ch + out_ch))
        )
        
    def forward(self, x):
        """
        x: (B, in_ch, H, W)
        out: (B, out_ch, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.in_ch, f"Expect input channel {self.in_ch}, got {C}"
        
        # Global pooling để lấy context (B, in_ch)
        pooled = F.adaptive_avg_pool2d(x, 1).view(B, C)
        
        # Sinh kernel và bias động
        params = self.extract_WeightBias(pooled)              # (B, out_ch*in_ch + out_ch)
        weight = params[:, :self.out_ch*C].view(B, self.out_ch, C)   # (B, out_ch, in_ch)
        bias   = params[:, self.out_ch*C:].view(B, self.out_ch, 1)   # (B, out_ch, 1)
        
        # Reshape input để nhân batch
        x_flat = x.view(B, C, -1)                             # (B, in_ch, H*W)
        
        # Dynamic pointwise conv bằng bmm
        out = torch.bmm(weight, x_flat) + bias                # (B, out_ch, H*W)
        
        out = F.relu(out)

        # Reshape lại (B, out_ch, H, W)
        out = out.view(B, self.out_ch, H, W)
        return out
    
class ExtractPolicy(nn.Module):
    def __init__(self, in_ch = 16):
        super().__init__()

        self.conv1x1 = nn.Conv2d(in_channels=in_ch,out_channels=1,kernel_size=1,padding=0)
    
    def forward(self,x):
        # input (B, in_ch, H, W)
        x = self.conv1x1(x)
        return x


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (B, 3, 15, 15)
        self.extractDirFeatures = ExtractDirectionalFeatures(in_channels=3, mid_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.incrementalModel = IncrementalModel(in_channels=256)
        self.dynamic = DynamicPointwise(in_ch=256,out_ch=128)
        self.extractPolicy = ExtractPolicy(in_ch=128)

    def forward(self, x):
        # x: (B, 3, 15, 15)

        x = self.extractDirFeatures(x)   # (B, 64, 15, 15)
        x = self.incrementalModel(x)      # (B, 64, 15, 15)
        x = self.dynamic(x)
        policy = self.extractPolicy(x)         # (B, 1, 15, 15)
        policy = policy.view(x.size(0), -1)         # (B, 225) — chưa softmax
        return policy               # logits output


    def train_model(self, dataloader):
        # target_index: (B,) — chỉ số ô đúng (0~224)
        loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        cnt_eval = 0
        global tensor_input_eval, tensor_target_eval, target_index_eval

        self.train()
        lr = 0.002
        optimizer = torch.optim.Adam([
            {"params": self.extractDirFeatures.parameters(), "lr": lr},
            {"params": self.incrementalModel.parameters(), "lr": lr},
            {"params": self.dynamic.parameters(), "lr": lr},
            {"params": self.extractPolicy.parameters(), "lr": lr}  # phần policy nhẹ hơn
        ],weight_decay=1e-4)
        

        cnt_mini_batch = 0

        for epoch in range(4000):
            print(f"Epoch {epoch} processing...")
            for data_input, target_probs in dataloader:
                # data_input = data_input.to(device)
                # target_index = target_index.to(device)

                optimizer.zero_grad()
                logits = self.forward(data_input)           # (B, 225)
                loss = loss_func(logits,target_probs)
                # loss = loss = F.kl_div(
                #     input=F.log_softmax(logits, dim=1),
                #     target=target_probs,
                #     reduction='batchmean'
                # )      

                # print("Target sample:", target_probs[0])
                # print("Pred sample:", F.softmax(logits, dim=1)[0])
                # print("KL loss:", loss.item())

                # input()

                # target: (B,)
                if loss < 0.1:
                    print(f"Early stop at epoch {epoch} with loss {loss.item():.4f}")
                    torch.save(self.state_dict(), f"model_rapfi_save_3.pth")
                    print(f"Model saved at epoch {epoch}")
                    return
                
                loss.backward()
                optimizer.step()

                cnt_mini_batch += 1
                if cnt_mini_batch % 20 == 0:
                    print(f"mini-batch loss [{cnt_mini_batch}]: {loss.item():.4f}")

            cnt_mini_batch = 0
            if epoch % 1 == 0:
                torch.save(self.state_dict(), f"model_rapfi_save_3.pth")
                print(f"Model saved at epoch {epoch}")
            
        super().train(mode=False)

