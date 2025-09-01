from Model import AlphaZero
import torch.utils.data as dataLoader
import torch
from rapfi_best import MyModel

tensor_data = torch.load('tensor_data_rapfi.pt')
tensor_target = torch.load('tensor_target_rapfi.pt')

dataPack = dataLoader.TensorDataset(
    tensor_data,
    tensor_target
)
dataPack_loader = dataLoader.DataLoader(dataPack, batch_size=64, shuffle=True)

my_model = MyModel().to(device='cuda')
# my_model.load_state_dict(torch.load('model_rapfi_save_3.pth'))
my_model.train_model(dataPack_loader)


