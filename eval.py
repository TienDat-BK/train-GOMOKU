from Model import AlphaZero
import torch.utils.data as dataLoader
import torch

tensor_data_eval = torch.load('tensor_data_eval.pt')
tensor_target_eval = torch.load('tensor_target_eval.pt')

dataPack_eval = dataLoader.TensorDataset(
    tensor_data_eval,
    tensor_target_eval
)
dataPack_loader_eval = dataLoader.DataLoader(dataPack_eval, batch_size=64, shuffle=False)

my_model = AlphaZero().to(device='cuda')
my_model.load_state_dict(torch.load('model_save.pth'))
my_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data, target in dataPack_loader_eval:
        data = data.to('cuda')
        target = target.to('cuda')
        output = my_model(data)  # output shape: (B, 255)
        pred = output.argmax(dim=1)  # predicted class indices, shape: (B,)
        correct += (pred == target).sum().item()
        total += target.size(0)

accuracy = correct / total
print(f'Evaluation accuracy: {accuracy:.4f}')