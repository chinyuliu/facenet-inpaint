import os
import torch
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm
import argparse

def mask_center(image_tensor, trim_ratio=0.1):
    masked = image_tensor.clone()
    _, H, W = image_tensor.shape

    # 中心遮蔽區塊的座標
    mask_size = 32
    center = H // 2
    hstart = center - mask_size // 2
    hend = center + mask_size // 2
    wstart = center - mask_size // 2
    wend = center + mask_size // 2

    # 擴張區域的座標
    expand = 20
    ext_hstart = max(hstart - expand, 0)
    ext_hend = min(hend + expand, H)
    ext_wstart = max(wstart - expand, 0)
    ext_wend = min(wend + expand, W)

    # 擷取擴張區域
    expanded_patch = image_tensor[:, ext_hstart:ext_hend, ext_wstart:ext_wend].clone()

    # 將中心遮蔽區塊清除（設為 NaN 以便排除計算）
    mask_hstart = hstart - ext_hstart
    mask_hend = hend - ext_hstart
    mask_wstart = wstart - ext_wstart
    mask_wend = wend - ext_wstart
    expanded_patch[:, mask_hstart:mask_hend, mask_wstart:mask_wend] = float('nan')

    # 計算修剪均值（去掉極端值後的平均值）
    C, _, _ = expanded_patch.shape
    mean_val = torch.zeros(C, 1, 1, device=expanded_patch.device)
    for c in range(C):  # 對每個通道單獨計算
        channel_data = expanded_patch[c].flatten()
        valid_data = channel_data[~torch.isnan(channel_data)]  # 移除 NaN
        if valid_data.numel() == 0:
            # 若無有效數據，使用全圖該通道的均值
            valid_data = image_tensor[c].flatten()
            print("警告：擴張區域無有效像素，使用全圖均值")
        if valid_data.numel() > 0:
            # 按值排序，移除最大和最小的 trim_ratio 比例數據
            sorted_data, _ = torch.sort(valid_data)
            trim_count = int(len(sorted_data) * trim_ratio / 2)  # 每端修剪數量
            trimmed_data = sorted_data[trim_count:len(sorted_data) - trim_count]
            if trimmed_data.numel() > 0:
                mean_val[c, 0, 0] = trimmed_data.mean()
            else:
                mean_val[c, 0, 0] = valid_data.mean()  # 若修剪後無數據，退回普通均值
        else:
            mean_val[c, 0, 0] = image_tensor[c].mean()  # 若無數據，使用全圖均值

    # 用修剪均值填補原圖中間區域
    masked[:, hstart:hend, wstart:wend] = mean_val

    return masked

def preprocess_celeba(image_folder, output_root, image_size=128, num_train=80_000, num_val=10_000, num_test=10_000, split_mode='shuffle', trim_ratio=0.1):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # 創建輸出資料夾
    os.makedirs(output_root, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{output_root}/{split}', exist_ok=True)

    # 讀取圖片清單
    images = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.jpg', '.png'))]
    print(f"總共有 {len(images)} 張圖片需要處理")

    # 檢查圖片總數是否足夠
    total_needed = num_train + num_val + num_test
    if total_needed > len(images):
        raise ValueError(f"要求的總數 {total_needed} 超過可用圖片數 {len(images)}")

    # 根據 split_mode 選擇分割方式
    if split_mode == 'shuffle':
        random.seed(7)  # 設定隨機種子以確保可重現性
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)
        images_to_process = shuffled_images
    else:  # ordered
        images_to_process = images

    # 分割資料集
    split_dict = {
        'train': images_to_process[:num_train],
        'val': images_to_process[num_train:num_train + num_val],
        'test': images_to_process[num_train + num_val:num_train + num_val + num_test]
    }

    # 處理每個資料集
    for split, split_images in split_dict.items():
        for idx, img_name in tqdm(enumerate(split_images), total=len(split_images), desc=f"Processing {split}"):
            try:
                img_path = os.path.join(image_folder, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                masked_tensor = mask_center(img_tensor, trim_ratio=trim_ratio)
                save_path = os.path.join(f'{output_root}/{split}', f"{idx}.pt")
                torch.save((img_tensor, masked_tensor), save_path)
            except Exception as e:
                print(f"處理圖片 {img_name} 失敗: {e}")

    # 輸出資料集分配結果
    print("\n資料預處理完成，資料集分配如下：")
    print(f"  訓練集（train）: {len(split_dict['train'])} 張")
    print(f"  驗證集（val）  : {len(split_dict['val'])} 張")
    print(f"  測試集（test） : {len(split_dict['test'])} 張")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CelebA dataset")
    parser.add_argument('--image_folder', default='data/img_align_celeba/img_align_celeba', help='Path to CelebA images')
    parser.add_argument('--output_root', default='data/img_align_celeba/processed/CelebA', help='Output path for processed data')
    parser.add_argument('--image_size', type=int, default=128, help='Image size for resizing')
    parser.add_argument('--num_train', type=int, default=80_000, help='Number of training images')
    parser.add_argument('--num_val', type=int, default=10_000, help='Number of validation images')
    parser.add_argument('--num_test', type=int, default=10_000, help='Number of test images')
    parser.add_argument('--split_mode', choices=['ordered', 'shuffle'], default='shuffle', help='Mode to split dataset: ordered or shuffle')
    parser.add_argument('--trim_ratio', type=float, default=0.1, help='Ratio of extreme values to trim from each end (0.0 to 0.5)')
    args = parser.parse_args()
    preprocess_celeba(image_folder=args.image_folder, output_root=args.output_root, image_size=args.image_size, num_train=args.num_train, num_val=args.num_val, num_test=args.num_test,
                      split_mode=args.split_mode, trim_ratio=args.trim_ratio
                     )
#依序分割資料集
#python preprocess_celeba.py --split_mode ordered --image_folder data/img_align_celeba/img_align_celeba --output_root data/img_align_celeba/processed/CelebA --image_size 128 --num_train 80000 --num_val 10000 --num_test 10000 --trim_ratio 0.1

#python preprocess_celeba.py --split_mode ordered  --num_train 90000 --num_val 11250 --num_test 11250


#隨機分割資料集（預設）
#python preprocess_celeba.py --split_mode shuffle --image_folder data/img_align_celeba/img_align_celeba --output_root data/img_align_celeba/processed/CelebA --image_size 128 --num_train 80000 --num_val 10000 --num_test 10000 --trim_ratio 0.1
'''
def mask_center(image_tensor):
    masked = image_tensor.clone()
    _, H, W = image_tensor.shape

    # 中心遮蔽區塊的座標
    mask_size = 32
    center = H // 2
    hstart = center - mask_size // 2
    hend = center + mask_size // 2
    wstart = center - mask_size // 2
    wend = center + mask_size // 2

    # 擴張區域的座標
    expand = 20
    ext_hstart = max(hstart - expand, 0)
    ext_hend = min(hend + expand, H)
    ext_wstart = max(wstart - expand, 0)
    ext_wend = min(wend + expand, W)

    # 擷取擴張區域
    expanded_patch = image_tensor[:, ext_hstart:ext_hend, ext_wstart:ext_wend].clone()

    # 將中心遮蔽區塊清除（設為 NaN 以便排除計算）
    mask_hstart = hstart - ext_hstart
    mask_hend = hend - ext_hstart
    mask_wstart = wstart - ext_wstart
    mask_wend = wend - ext_wstart
    expanded_patch[:, mask_hstart:mask_hend, mask_wstart:mask_wend] = float('nan')

    # 計算 NaN 忽略的平均值
    mean_val = torch.nanmean(expanded_patch, dim=(1, 2), keepdim=True)  # shape: (C, 1, 1)  沿著高度和寬度維度 (dim=(1, 2)) 計算每個通道的平均值

    # 用平均值填補原圖中間區域
    masked[:, hstart:hend, wstart:wend] = mean_val

    return masked

def preprocess_celeba(image_folder, output_root, image_size=128):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    os.makedirs(f'{output_root}/ordered', exist_ok=True)
    os.makedirs(f'{output_root}/shuffle', exist_ok=True)

    images = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.jpg', '.png'))]
    print(f"總共有 {len(images)} 張圖片需要處理")

    # Ordered
    for idx, img_name in tqdm(enumerate(images), total=len(images), desc="Processing ordered"):
        try:
            img_path = os.path.join(image_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            masked_tensor = mask_center(img_tensor)
            torch.save((img_tensor, masked_tensor), os.path.join(f'{output_root}/ordered', f"{idx}.pt"))  # (原圖(縮小後) , 要填補的圖)
        except Exception as e:
            print(f"處理圖片 {img_name} 失敗: {e}")

    # Shuffle
    random.seed(7)
    shuffled_images = images.copy()
    random.shuffle(shuffled_images)
    for idx, img_name in tqdm(enumerate(shuffled_images), total=len(images), desc="Processing shuffle"):
        try:
            img_path = os.path.join(image_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            masked_tensor = mask_center(img_tensor)
            torch.save((img_tensor, masked_tensor), os.path.join(f'{output_root}/shuffle', f"{idx}.pt"))
        except Exception as e:
            print(f"處理圖片 {img_name} 失敗: {e}")

    print("資料預處理完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CelebA dataset")
    parser.add_argument('--image_folder', default='data/img_align_celeba/img_align_celeba', help='Path to CelebA images')
    parser.add_argument('--output_root', default='data/img_align_celeba/processed/CelebA', help='Output path for processed data')
    parser.add_argument('--image_size', type=int, default=128, help='Image size for resizing')  # 128*128
    args = parser.parse_args()
    preprocess_celeba(args.image_folder, args.output_root, args.image_size)
'''