import numpy as np

label = np.random.rand(255)  # ví dụ
label = label / np.sum(label)  # chuẩn hóa thành xác suất

uniform = np.ones(255) / 255
diff = np.abs(label - uniform)
mean_diff = np.mean(diff)

print("Mean absolute difference:", mean_diff)