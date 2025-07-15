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

def save_checkpoint(G, Glo_D, Loc_D, G_opt=None, Glo_D_opt=None, Loc_D_opt=None,
                    epoch=None, G_losses=None, Glo_D_losses=None, Loc_D_losses=None, Val_losses=None,
                    Glo_G_Adv_losses=None, Loc_G_Adv_losses=None, path=None, name=None, best_loss=None,
                    early_stop_counter=None, full_checkpoint=False):
    os.makedirs(path, exist_ok=True)
    checkpoint = {'G_state_dict': G.state_dict(), 'Glo_D_state_dict': Glo_D.state_dict(),
        'Loc_D_state_dict': Loc_D.state_dict(),
    }
    if full_checkpoint:
        checkpoint.update({
            'G_optimizer_state_dict': G_opt.state_dict() if G_opt else None,
            'Glo_D_optimizer_state_dict': Glo_D_opt.state_dict() if Glo_D_opt else None,
            'Loc_D_optimizer_state_dict': Loc_D_opt.state_dict() if Loc_D_opt else None,
            'epoch': epoch,
            'G_losses': G_losses, 'Glo_D_losses': Glo_D_losses, 'Loc_D_losses': Loc_D_losses,
            'Val_losses': Val_losses,
            'Glo_G_Adv_losses': Glo_G_Adv_losses,
            'Loc_G_Adv_losses': Loc_G_Adv_losses,
            'best_loss': best_loss,
            'early_stop_counter': early_stop_counter
        })
    tc.save(checkpoint, os.path.join(path, name))

def load_checkpoint(G, Glo_D, Loc_D, G_opt=None, Glo_D_opt=None, Loc_D_opt=None, path=None, name=None, device=None):
    checkpoint_path = os.path.join(path, name)
    if not os.path.exists(checkpoint_path):
        print("沒有找到檢查點，將從頭開始訓練")
        return 0, [], [], [], [], [], [], float('inf'), 0

    checkpoint = tc.load(checkpoint_path, map_location=device)
    G.load_state_dict(checkpoint['G_state_dict'])
    Glo_D.load_state_dict(checkpoint['Glo_D_state_dict'])
    Loc_D.load_state_dict(checkpoint['Loc_D_state_dict'])

    if 'G_optimizer_state_dict' in checkpoint and G_opt and checkpoint['G_optimizer_state_dict']:
        G_opt.load_state_dict(checkpoint['G_optimizer_state_dict'])
        Glo_D_opt.load_state_dict(checkpoint['Glo_D_optimizer_state_dict'])
        Loc_D_opt.load_state_dict(checkpoint['Loc_D_optimizer_state_dict'])
        print(f"成功載入檢查點：從第 {checkpoint['epoch'] + 1} epoch 繼續訓練")
        return (checkpoint['epoch'] + 1,
                checkpoint.get('G_losses', []),
                checkpoint.get('Glo_D_losses', []),
                checkpoint.get('Loc_D_losses', []),
                checkpoint.get('Val_losses', []),
                checkpoint.get('Glo_G_Adv_losses', []),
                checkpoint.get('Loc_G_Adv_losses', []),
                checkpoint.get('best_loss', float('inf')),
                checkpoint.get('early_stop_counter', 0))
    else:
        print(f"成功載入檢查點：僅模型參數，無法繼續訓練")
        return 0, [], [], [], [], [], [], float('inf'), 0