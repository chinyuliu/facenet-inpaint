import os
import torch as tc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from model import Generator
from config import HyperParameters
from PIL import Image
from preprocess_celeba import mask_center
import argparse

plt.switch_backend('agg')  # 確保使用非交互式後端，避免繪圖混亂

def parse_args():
    parser = argparse.ArgumentParser(description="圖像填補測試")
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='G_best.pth',  # 預設值
        help='指定要載入的生成器模型檔案名稱'
    )
    return parser.parse_args()

def plot_compare(original, masked, inpainted, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(masked.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title('Masked Input')
    axs[1].axis('off')

    axs[2].imshow(inpainted.permute(1, 2, 0).cpu().numpy())
    axs[2].set_title('Inpainted by Generator')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # 關閉畫布，避免堆疊

def remove_module_prefix(state_dict):  # 移除 module. 前綴再載入
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def test(model_name):
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    print(f"使用 {device} 進行測試")

    H = HyperParameters()

    # 載入生成器
    model_path = os.path.join(H.model_path, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型檔案不存在：{model_path}")
    
    G = Generator(H.dc).to(device)
    # 載入生成器參數
    checkpoint = tc.load(model_path, map_location=device)

    if False:  # 是否使用DDP做訓練()
        G.load_state_dict(remove_module_prefix(checkpoint['G_state_dict']))
    else:
        G.load_state_dict(checkpoint['G_state_dict'])

    G.eval()
    print(f"成功載入 Generator {model_path}")

    # 讀取 test_img 資料夾中的圖像
    test_img_folder = H.test_data_path
    if not os.path.exists(test_img_folder):
        raise FileNotFoundError(f"測試圖像資料夾不存在：{test_img_folder}")
    
    img_paths = [
        os.path.join(test_img_folder, fname)
        for fname in os.listdir(test_img_folder)
        if fname.endswith(('.png', '.jpg', '.jpeg'))
    ]

    # 檢查圖像數量
    print(f"找到 {len(img_paths)} 張圖像")
    if len(img_paths) == 0:
        raise ValueError("測試資料夾中無圖像")

    # 創建結果資料夾
    os.makedirs(H.result_path, exist_ok=True)
    os.makedirs(H.test_data_result_path, exist_ok=True)

    # 限制處理的圖像數量
    max_images = getattr(H, 'max_test_images', 10)  # 預設最多處理10張
    img_paths = img_paths[:max_images]
    print(f"將處理 {len(img_paths)} 張圖像")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    for idx, img_path in enumerate(tqdm(img_paths, desc="Testing")):
        # 讀取圖像並轉為RGB
        original = cv2.imread(img_path)
        if original is None:
            print(f"無法載入圖像：{img_path}")
            continue
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # 將NumPy陣列轉為PIL圖像以應用transforms
        original_pil = Image.fromarray(original)

        # 應用transforms：調整大小並轉為張量
        original_tensor = transform(original_pil).unsqueeze(0).to(device)

        # 從張量轉回NumPy以供後續cv2操作
        original = (original_tensor[0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # 驗證遮罩範圍（圖像已調整為128x128）
        hstart, hend = H.cut_range
        wstart, wend = H.cut_range


        if hstart >= hend or wstart >= wend or hend > 128 or wend > 128:
            print(f"無效的遮罩範圍 {H.cut_range}，圖像尺寸為 128x128")
            continue

        # 原始圖像張量形狀: [1, C, H, W]
        single_image = original_tensor[0]  # shape: [C, H, W]

        # 建立遮蔽圖
        masked = single_image.clone()
        masked[:, hstart:hend, wstart:wend] = 0
        masked = mask_center(masked)
        masked = masked.unsqueeze(0)  # shape: [1, C, H, W]

        # 使用生成器填補
        with tc.no_grad():
            gen_img = G(masked)
            inpainted = masked.clone()
            inpainted[:, :, hstart:hend, wstart:wend] = gen_img[:, :, hstart:hend, wstart:wend]

        cv2.imwrite(
            os.path.join(H.test_data_result_path, f'inpainted_{idx}.png'),
            cv2.cvtColor((inpainted[0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        )

        # 生成對比圖
        save_file = os.path.join(H.test_data_result_path, f'test_result_{idx}.png')
        plot_compare(original_tensor[0], masked[0], inpainted[0], save_file)

    print(f"測試完成。結果儲存於 {H.test_data_result_path}")

if __name__ == "__main__":
    args = parse_args()
    test(args.model_name)

# python3 test.py --model_name G_best.pth
