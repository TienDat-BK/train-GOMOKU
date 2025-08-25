import os
import torch
from Model import AlphaZero
import random
import torch.utils.data as dataLoader

def extract_data_train_file(file_path):
    # mỗi file là một ván
    turn = 0
    board_tensor = torch.zeros((3,15,15), dtype=torch.float32).to(device='cuda')

    for line in open(file_path,'r').readlines():
        x,y = line.strip().split(',')
        x = int(x) - 1
        y = int(y) - 1

        if x < 0 or y < 0 or x >= 15 or y >= 15:
            print(f"Invalid move ({x}, {y}) in file {file_path}. Skipping...")
            continue

        # lấy label
        label = x + 15 * y
        target_data.append(label)
        training_data.append(board_tensor.clone())

        # cập nhật dữ liệu
        board_tensor[turn, x, y] = 1.0
        turn = 1 - turn
        board_tensor[2,:,:] = turn

    print(f"Processed {file_path} DONE!")
        
def extract_data_train_folder(folder_path, num_files=50):
    list_of_file = os.listdir(folder_path)
    random.shuffle(list_of_file)
    num_processed_files = 0
    for file_path in list_of_file:
        if num_processed_files >= num_files:
            break
        full_path = os.path.join(folder_path, file_path)
        if os.path.isfile(full_path):
            extract_data_train_file(full_path)
            num_processed_files += 1

    print(f"\tProcessed {folder_path} DONE!")



training_data = []
target_data = []

extract_data_train_folder('dataTrain\\Freestyle15_1', num_files=2000)
# extract_data_train('dataTrain\\Freestyle15_2')

tensor_data = torch.stack(training_data, dim=0).to(device='cuda')
tensor_target = torch.tensor(target_data, dtype=torch.long).to(device='cuda')

print(f"tensor_data shape: {tensor_data.shape}")

dataPack = dataLoader.TensorDataset(
    torch.stack(training_data, dim=0),
    torch.tensor(target_data, dtype=torch.long).to(device='cuda')
)

dataPack_loader = dataLoader.DataLoader(dataPack, batch_size=64, shuffle=True)

print(f"Total training data: {len(training_data)}")

torch.save(tensor_data, 'tensor_data_rapfi.pt')
torch.save(tensor_target, 'tensor_target_rapfi.pt')