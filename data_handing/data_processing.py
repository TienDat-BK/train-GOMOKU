import numpy as np  
import os
import torch

num_state = 0
mp_state = dict()

def soft_max(list_label : list, threshold=0):
    softmax_list = []
    for label in list_label:
        exp_label = np.exp(label - np.max(label))
        softmax_label = exp_label / np.sum(exp_label)
        softmax_label[softmax_label < threshold] = 0

        softmax_list.append(softmax_label)

    return softmax_list

def scale_liner(list_label : list):
    scale_list = []
    for label in list_label:
        scale_label = (label -  np.min(label)) / (np.max(label) - np.min(label)) 
        scale_label = scale_label / np.sum(scale_label)
        scale_list.append(scale_label)
    return scale_list

def rotate90(pos:tuple) -> tuple:
    return (pos[1], 14 - pos[0])

def rotate_label(pos: tuple, n) -> tuple:
    for _ in range(n):
        pos = rotate90(pos)
    return pos

def flip(board: np.ndarray, pos_label: tuple):
    """
    Trả về tuple gồm 4 kiểu flip khác nhau của cả board và pos_label:
    (flip dọc, flip ngang, flip qua đường chéo chính, flip qua đường chéo phụ)
    đã bỏ 1 flip  ngang và chéo
    """
    x, y = pos_label
    # Flip dọc (theo trục y)
    board_v = np.flip(board, axis=2)
    pos_v = (x, 14 - y)
    # Flip ngang (theo trục x)

    # board_h = np.flip(board, axis=1)
    # pos_h = (14 - x, y)

    # Flip qua đường chéo chính (x=y)
    board_diag_main = np.transpose(board, (0,2,1))
    pos_diag_main = (y, x)
    # Flip qua đường chéo phụ (x+y=14)

    # board_diag_anti = np.flip(np.transpose(board, (0,2,1)), axis=2)
    # pos_diag_anti = (14 - y, 14 - x)
    return (
        (board_v, pos_v),
        # (board_h, pos_h),
        (board_diag_main, pos_diag_main),
        # (board_diag_anti, pos_diag_anti)
    )

def rotate_board(board: np.ndarray, n: int) -> np.ndarray:
    """
    Xoay bàn cờ (3,15,15) n lần 90 độ.
    board[0] và board[1] được xoay, board[2] giữ nguyên.
    """
    rotated = np.empty_like(board)
    rotated[0] = np.rot90(board[0], k=n)
    rotated[1] = np.rot90(board[1], k=n)
    # ko dung turn nưa
    # rotated[2] = board[2].copy()
    return rotated

def check_trung(board: np.ndarray, pos_label : tuple) -> bool:
    global mp_state, list_label, list_state
    board_hash = hash(board.tobytes())

    x, y = pos_label

    if board_hash in mp_state.keys():
        # nếu đã trùng state
        idx = mp_state[board_hash]
        list_label[idx][x + 15*y] += 1.5 if cnt % 2 == whoWin else 1
        return True
    else:
        # đưa vào pack dữ liệu
        list_state.append(board)
        # tao label
        label = np.zeros((225))
        label[x + 15*y] = 1.5 if cnt % 2 == whoWin else 1
        list_label.append(label)
        # đưa hash vào 
        mp_state[board_hash] = len(list_label) - 1

cnt = -1
whoWin = 1

def extract_state_unique_file(file_path):
    global num_state, mp_state, list_label, list_state, cnt ,whoWin
    try:
        f = open(file_path, mode='r')
    except Exception as e:
        print(f"Cannot open file {file_path}: {e}")
        return
    
    turn = 0
    # == 0 là chẵn thắng, == 1 là lẻ thắng
    list_line = f.readlines()
    whoWin = len(list_line) % 2
    f.seek(0)
    game_state = np.zeros((2,15,15))
    cnt = 0

    if list_line[0].strip() !='8,8':
        return

    for line in list_line:
        num_state+=1
        cnt +=1
        x, y = line.strip().split(',')
        x=int(x) - 1
        y=int(y) - 1

        label = np.zeros((225))
        label[x + 15*y] += 2 if cnt % 2 == whoWin else 1 

        #xoay
        for i in range(1):
            # chi lay cac move cua WIN
            if cnt % 2 == whoWin:
                check_trung(rotate_board(game_state,i),rotate_label((x,y),i))
        #lật
        # for board_flip, pos_label in flip(game_state,(x,y)):
        #     check_trung(board_flip,pos_label)

        # cập nhật dữ liệu
        game_state[turn][x, y] = 1
        turn = 1 - turn
        # game_state[2][...] = turn

        # xoay phe lai
        tmp = game_state[0].copy()
        game_state[0] = game_state[1].copy()
        game_state[1] = tmp.copy()


    # print(f" Processing {file_path} DONE!")


def extract_folder(folder_path):
    list_file = os.listdir(folder_path)
    for file_path in list_file:
        extract_state_unique_file(folder_path + '//' + file_path)
    print(f"Processing {folder_path} DONE!")


def kl_divergence_uniform(p: np.ndarray):
    """
    Tính KL-Divergence giữa phân phối p và phân phối đều
    Input:
        p: ndarray shape (n,) - phân phối xác suất (tổng = 1)
    Output:
        KL divergence float
    """
    p = np.asarray(p, dtype=np.float64)
    p = p / np.sum(p)  # chuẩn hóa thành xác suất
    n = len(p)
    q = np.ones(n) / n  # phân phối đều

    # Tránh log(0) → dùng epsilon
    eps = 1e-12
    p_safe = np.clip(p, eps, 1)
    q_safe = np.clip(q, eps, 1)

    kl = np.sum(p_safe * np.log(p_safe / q_safe))
    return kl

def change_to_onehot():
    global list_label
    new_list_label = []
    for label in list_label:
        new_list_label.append(np.argmax(label))
    return new_list_label

# main
list_state = []
list_label = []

if os.path.exists("dataTrain\\Freestyle15_1"):
    path_1 = "dataTrain\\Freestyle15_1"
    path_2 = "dataTrain\\Freestyle15_2"
else:
    path_1 = "/content/train-GOMOKU/dataTrain/Freestyle15_1"
    path_2 = "/content/train-GOMOKU/dataTrain/Freestyle15_1"

extract_folder(path_1)
# extract_folder('dataTrain\\Freestyle15_2')

# list_label = soft_max(list_label)
list_label = scale_liner(list_label)
_cnt= 0
# for idx, label in enumerate(list_label):
    
#     if np.count_nonzero(label) == 1 :
#         _cnt+=1
        


print(f"dong deu: {_cnt}")

print(f'NumState: {num_state}')
print(f'list_state: {len(list_state)}')
print(f'list_label: {len(list_label)}')
# Stack list_state and list_label, convert to tensor
print(f"Stacking tensor data!")

device = torch.device("cuda")

states_tensor = torch.tensor(np.stack(list_state), dtype=torch.float32).to(device=device)
# labels_tensor = torch.tensor(np.stack(list_label), dtype=torch.float32).to(device=device)
onehot_label = np.array(change_to_onehot())
labels_tensor = torch.tensor(onehot_label, dtype=torch.int64).to(device=device)


print(f'states_tensor shape: {states_tensor.shape}')
print(f'labels_tensor shape: {labels_tensor.shape}')

torch.save(states_tensor,"tensor_data_rapfi.pt")
torch.save(labels_tensor,"tensor_target_rapfi.pt")
print("Saved!")

print(hash(np.zeros((2,15,15)).tobytes()) in mp_state.keys())

