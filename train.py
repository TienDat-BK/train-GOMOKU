import os
import torch
from Model import AlphaZero
import torch.utils.data as dataLoader

def extract_data_train(folder_path):
    list_of_file = os.listdir(folder_path)
    num_files = 0
    for file_path in list_of_file:
        num_files += 1
        if num_files > 50:
            break
        # mỗi file là một ván
        data = torch.zeros((17, 15, 15), dtype=torch.float32).to(device='cuda')
        turn = 0
        move_buffer = [ [(-1, -1) for _ in range(8)] for _ in range(2) ]

        for line in open(folder_path + '\\' + file_path,'r').readlines():
            x,y = line.strip().split(',')
            x = int(x) - 1
            y = int(y) - 1

            if x < 0 or y < 0 or x >= 15 or y >= 15:
                print(f"Invalid move ({x}, {y}) in file {file_path}. Skipping...")
                continue

            # lấy label
            label = x + 15 * y
            target_data.append(label)
            training_data.append(data.clone())

            # cập nhật dữ liệu
            data = torch.zeros((17, 15, 15), dtype=torch.float32).to(device='cuda')
            move_buffer[turn].append( (x,y) )

            for team in range(2):
                for i in range(8):
                    if move_buffer[team][-(i+1)] == (-1, -1):
                        continue
                    xx, yy = move_buffer[team][-(i+1)]
                    data[team * 8 + i, xx, yy] = 1.0
            data[16,:,:] = turn
            turn = 1 - turn

        
    print(f"Processed {folder_path} DONE!")



training_data = []
target_data = []

extract_data_train('dataTrain\\Freestyle15_1')
# extract_data_train('dataTrain\\Freestyle15_2')

tensor_data = torch.stack(training_data, dim=0)
print(f"tensor_data shape: {tensor_data.shape}")

dataPack = dataLoader.TensorDataset(
    torch.stack(training_data, dim=0),
    torch.tensor(target_data, dtype=torch.long).to(device='cuda')
)

dataPack_loader = dataLoader.DataLoader(dataPack, batch_size=64, shuffle=True)

print(f"Total training data: {len(training_data)}")

model = AlphaZero().to(device='cuda')
model.startTraining(dataPack_loader, epochs=2000)

