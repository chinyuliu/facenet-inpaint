import time
from functools import wraps
import os
import matplotlib.pyplot as plt

def print_compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    days = elapsed_time // (3600 * 24)
    hours = (elapsed_time % (3600 * 24)) // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"程式執行時間: {int(days)} 天 {int(hours)} 小時 {int(minutes)} 分鐘 {int(seconds)} 秒")

def compute_time(func):
    """裝飾器：測量函式執行時間並印出來"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print_compute_time(start_time, end_time)
        return result
    return wrapper

def save_comparison(inputs, outputs, targets, save_path, num_images=3):  # 保存輸入、生成和真實圖片的比較
    inputs = inputs.cpu().detach()
    outputs = outputs.cpu().detach()
    targets = targets.cpu().detach()

    batch_size = min(inputs.size(0), num_images)
    _, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    # 處理單張圖片的情況
    if batch_size == 1:
        axes = [axes]  # 將一維 axes 包裝成列表，模擬二維數組

    for idx in range(batch_size):
        axes[idx][0].imshow(inputs[idx].permute(1, 2, 0))
        axes[idx][0].set_title('Input (Masked)')
        axes[idx][0].axis('off')

        axes[idx][1].imshow(outputs[idx].permute(1, 2, 0))
        axes[idx][1].set_title('Generated')
        axes[idx][1].axis('off')

        axes[idx][2].imshow(targets[idx].permute(1, 2, 0))
        axes[idx][2].set_title('Ground Truth')
        axes[idx][2].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
