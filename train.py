import os
import torch as tc
from torch import nn
from dataloader import get_dataloader
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Generator, Glo_Discriminator, Loc_Discriminator
from config import HyperParameters
from utils import compute_time, save_comparison, save_checkpoint, load_checkpoint
import random
import csv


def plot_losses(g_losses, glo_d_losses, loc_d_losses, val_losses, save_path):  # 繪製損失曲線
    plt.figure(figsize=(12, 6))
    plt.plot(g_losses, label='Generator Loss(train)')
    plt.plot(glo_d_losses, label='Glo Discriminator Loss')
    plt.plot(loc_d_losses, label='Loc Discriminator Loss')
    plt.plot(val_losses, label='Generator Loss(val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Valid Losses')
    plt.savefig(save_path)
    plt.close()

def validate(G, dataloader, device, H, rec_loss):  # 在驗證集上評估生成器損失
    G.eval()
    val_loss = 0.0
    total_samples = 0
    hstart, hend = H.cut_range
    wstart, wend = H.cut_range
    with tc.no_grad():
        for original_img, corrupted_img in dataloader:
            original_img = original_img.to(device)
            corrupted_img = corrupted_img.to(device)

            gen = G(corrupted_img)[:, :, hstart:hend, wstart:wend]
            g_rec = rec_loss(gen, original_img[:, :,hstart:hend, wstart:wend])
            val_loss += g_rec.item() * original_img.size(0)  # 按批次大小加權(防止最後一個批次不滿 batch_size)
            total_samples += original_img.size(0)
    return val_loss / total_samples

def test(epoch, H, device, G, test_loader):  # 在測試集上生成比較圖片
    G.eval()
    inputs_list = []
    outputs_list = []
    targets_list = []
    hstart, hend = H.cut_range
    wstart, wend = H.cut_range
    with tc.no_grad():
        count = 0
        for original_img, corrupted_img in test_loader:
            original_img = original_img.to(device)
            corrupted_img = corrupted_img.to(device)

            gen = G(corrupted_img)[:, :, hstart:hend, wstart:wend]
            fake_img = corrupted_img.clone()
            fake_img[:, :, hstart:hend, wstart:wend] = gen

            inputs_list.append(corrupted_img.cpu())
            outputs_list.append(fake_img.cpu())
            targets_list.append(original_img.cpu())

            count += original_img.size(0)
            if count == 3:
                break
        # 拼接成 batch
        inputs = tc.cat(inputs_list, dim=0)[:3]  # 取前 3 張
        outputs = tc.cat(outputs_list, dim=0)[:3]
        targets = tc.cat(targets_list, dim=0)[:3]
        save_comparison(inputs=inputs, outputs=outputs, targets=targets, save_path=os.path.join(H.result_path, f'comparison_epoch{epoch+1}.png'), num_images=3)

@compute_time
def train():
    device_ids = [3]
    device = tc.device(f'cuda:{device_ids[0]}' if tc.cuda.is_available() else 'cpu')
    #device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    print(f"使用 {device} 訓練")
    # 設定隨機種子（CPU / CUDA / numpy / Python 隨機）
    seed = 7
    tc.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    random.seed(seed)

    H = HyperParameters()
    os.makedirs(H.result_path, exist_ok=True)  # 確保結果資料夾存在
    os.makedirs(H.model_path, exist_ok=True)
    print(H)

    # 載入資料集
    train_loader = get_dataloader(split='train', batch_size=H.batch_size, shuffle=True, max_samples=90_000)  # 170_000  90_000
    val_loader = get_dataloader(split='val', batch_size=H.batch_size, shuffle=True, max_samples=11_250)  # 9700  11_250
    test_loader = get_dataloader(split='test', batch_size=H.batch_size, shuffle=False, max_samples=10)
    print(f"訓練集大小: {len(train_loader.dataset)}")
    print(f"驗證集大小: {len(val_loader.dataset)}")
    print(f"測試集大小: {len(test_loader.dataset)}")

    # 建立模型與優化器
    G = Generator(H.dc).to(device)
    Glo_D = Glo_Discriminator(H.dc).to(device)
    Loc_D = Loc_Discriminator(H.dc).to(device)
    G_optimizer = tc.optim.Adam(G.parameters(), lr=H.lr, weight_decay=1e-5)  # L2 正則化
    Glo_D_optimizer = tc.optim.Adam(Glo_D.parameters(), lr=H.lr, weight_decay=1e-5)
    Loc_D_optimizer = tc.optim.Adam(Loc_D.parameters(), lr=H.lr, weight_decay=1e-5)
    
    # 學習率調整器（根據 G_loss 調整）
    #scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, mode='min', factor=0.5, patience=10)
    #Glo_D_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(Glo_D_optimizer, mode='min', factor=0.5, patience=10)
    #Loc_D_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(Loc_D_optimizer, mode='min', factor=0.5, patience=10)

    # 嘗試從檢查點繼續訓練
    start_epoch, G_losses, Glo_D_losses, Loc_D_losses, Val_losses, Glo_G_Adv_losses, Loc_G_Adv_losses, best_loss, early_stop_counter = load_checkpoint(
        G, Glo_D, Loc_D, G_optimizer, Glo_D_optimizer, Loc_D_optimizer, H.model_path, 'G_latest.pth', device)

    rec_loss = nn.MSELoss()
    adv_loss = nn.BCELoss()

    # 初始化損失列表（如果從頭開始）
    if start_epoch == 0:
        G_losses = []
        Glo_D_losses = []
        Loc_D_losses = []
        Val_losses = []
        Glo_G_Adv_losses = []
        Loc_G_Adv_losses = []

    # 初始化或覆蓋 CSV 文件，寫入歷史損失（如果從檢查點恢復）
    loss_file = os.path.join(H.result_path, 'losses.csv')  # loss_file = os.path.join(H.result_path, f'losses_{time.strftime("%Y%m%d_%H%M%S")}.csv')
    with open(loss_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'G_Loss', 'Glo_D_Loss', 'Loc_D_Loss', 'Val_Loss', 'Glo_G_Adv', 'Loc_G_Adv', 'Is_Best'])
        for epoch, (g_loss, glo_d_loss, loc_d_loss, val_loss, glo_g_adv, loc_g_adv) in enumerate(
            zip(G_losses, Glo_D_losses, Loc_D_losses, Val_losses, Glo_G_Adv_losses, Loc_G_Adv_losses), 1
        ):
            writer.writerow([epoch, f"{g_loss:.6f}", f"{glo_d_loss:.6f}", f"{loc_d_loss:.6f}",
                             f"{val_loss:.6f}", f"{glo_g_adv:.6f}", f"{loc_g_adv:.6f}", 0])

    early_stop_patience = 200
    best_loss = float('inf') if start_epoch == 0 else best_loss 
    early_stop_counter = 0 if start_epoch == 0 else early_stop_counter
    # 訓練
    G_iter = H.max_iter*3 // 20  # 向下取整
    D_iter = H.max_iter // 6
    for epoch in range(start_epoch, H.max_iter):
        if epoch < G_iter:
            print('-' * 6 + ' Train G ' + '-' * 6)
        elif epoch < D_iter:
            print('-' * 6 + ' Train D ' + '-' * 6)
        else:
            print('-' * 6 + ' Train G & D Alternately ' + '-' * 6)
            
        G.train()
        Glo_D.train()
        Loc_D.train()
        
        g_loss_epoch = 0
        glo_d_loss_epoch = 0
        loc_d_loss_epoch = 0
        glo_g_adv_epoch = 0
        loc_g_adv_epoch = 0
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{H.max_iter}")
        for original_img, masked_img in pbar:
            original_img = original_img.to(device)
            masked_img = masked_img.to(device)
            batch_size = original_img.size(0)
            
            hstart, hend = H.cut_range
            wstart, wend = H.cut_range

            ## ---- Generator ----
            gen = G(masked_img)[:, :, hstart:hend, wstart:wend]  # 只生成缺失區塊
            
            # 建立 fake_img 作為完整圖像（僅中間區塊被生成內容替代）
            fake_img = masked_img.clone()
            fake_img[:, :, hstart:hend, wstart:wend] = gen

            # 判別器輸入
            Glo_real = Glo_D(original_img)
            Glo_fake = Glo_D(fake_img)
            #Loc_real = Loc_D(original_img[:, :, hstart:hend, wstart:wend])
            #Loc_fake = Loc_D(gen)
            v=0
            Loc_real = Loc_D(original_img[:, :, hstart-v:hend+v, wstart-v:wend+v])
            Loc_fake = Loc_D(fake_img[:, :, hstart-v:hend+v, wstart-v:wend+v])

            Glo_real_label = tc.ones_like(Glo_real) * 0.9
            Glo_fake_label = tc.zeros_like(Glo_fake) + 0.1
            Loc_real_label = tc.ones_like(Loc_real) * 0.9
            Loc_fake_label = tc.zeros_like(Loc_fake) + 0.1
            if epoch < G_iter:
                # ------------train G------------
                Glo_D_real = adv_loss(Glo_real, Glo_real_label)  # 計算判別器損失但不更新
                Glo_D_fake = adv_loss(Glo_fake, Glo_fake_label)
                Loc_D_real = adv_loss(Loc_real, Loc_real_label)
                Loc_D_fake = adv_loss(Loc_fake, Loc_fake_label)
                Glo_D_loss = Glo_D_real + Glo_D_fake
                Loc_D_loss = Loc_D_real + Loc_D_fake

                G.zero_grad()
                G_rec = rec_loss(gen, original_img[:, :,hstart:hend, wstart:wend])  # MSE
                G_loss = G_rec
                G_loss.backward()
                G_optimizer.step()
            elif epoch < D_iter:
                # ------------train D------------
                G_rec = rec_loss(gen, original_img[:, :, hstart:hend, wstart:wend].to(device))  # MSE
                Glo_G_adv = adv_loss(Glo_D(fake_img), Glo_real_label)
                #Loc_G_adv = adv_loss(Loc_D(gen), Loc_real_label)
                Loc_G_adv = adv_loss(Loc_D(fake_img[:, :, hstart-v:hend+v, wstart-v:wend+v]), Loc_real_label)

                G_loss = H.Lambda * (Glo_G_adv + Loc_G_adv) + G_rec
                
                Glo_D.zero_grad()
                Loc_D.zero_grad()
                Glo_D_loss = adv_loss(Glo_real, Glo_real_label) + adv_loss(Glo_fake, Glo_fake_label)
                Loc_D_loss = adv_loss(Loc_real, Loc_real_label) + adv_loss(Loc_fake, Loc_fake_label)
                D_loss = Glo_D_loss + Loc_D_loss
                D_loss.backward()
                Glo_D_optimizer.step()
                Loc_D_optimizer.step()
                # 累計對抗損失
                glo_g_adv_epoch += Glo_G_adv.item() * batch_size
                loc_g_adv_epoch += Loc_G_adv.item() * batch_size
            else:
                # ------------alternatively train D and G------------
                Glo_D.zero_grad()
                Loc_D.zero_grad()
                Glo_D_loss = adv_loss(Glo_real, Glo_real_label) + adv_loss(Glo_fake, Glo_fake_label)
                Loc_D_loss = adv_loss(Loc_real, Loc_real_label) + adv_loss(Loc_fake, Loc_fake_label)
                D_loss = Glo_D_loss + Loc_D_loss
                D_loss.backward(retain_graph=True)
                Glo_D_optimizer.step()
                Loc_D_optimizer.step()

                G.zero_grad()
                G_rec = rec_loss(gen, original_img[:, :, hstart:hend, wstart:wend])
                Glo_G_adv = adv_loss(Glo_D(fake_img), Glo_real_label)
                #Loc_G_adv = adv_loss(Loc_D(gen), Loc_real_label)
                Loc_G_adv = adv_loss(Loc_D(fake_img[:, :, hstart-v:hend+v, wstart-v:wend+v]), Loc_real_label)

                G_loss = H.Lambda * (Glo_G_adv + Loc_G_adv) + G_rec
                G_loss.backward()
                G_optimizer.step()
                # 累計對抗損失
                glo_g_adv_epoch += Glo_G_adv.item() * batch_size
                loc_g_adv_epoch += Loc_G_adv.item() * batch_size

            g_loss_epoch += G_loss.item() * batch_size
            glo_d_loss_epoch += Glo_D_loss.item() * batch_size
            loc_d_loss_epoch += Loc_D_loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix({
                'G_loss': f'{G_loss.item():.4f}',
                'Glo_D_loss': f'{Glo_D_loss.item():.4f}',
                'Loc_D_loss': f'{Loc_D_loss.item():.4f}'
            })

        # 計算平均損失
        G_losses.append(g_loss_epoch / total_samples)
        Glo_D_losses.append(glo_d_loss_epoch / total_samples)
        Loc_D_losses.append(loc_d_loss_epoch / total_samples)
        # 僅在 epoch >= G_iter 時記錄對抗損失，否則追加 0
        Glo_G_Adv_losses.append(glo_g_adv_epoch / total_samples if epoch >= G_iter else 0)
        Loc_G_Adv_losses.append(loc_g_adv_epoch / total_samples if epoch >= G_iter else 0)

        # 驗證
        val_loss = validate(G, val_loader, device, H, rec_loss)
        Val_losses.append(val_loss)
        print(f"驗證損失: {val_loss:.4f}")

        # 每 10 個 epoch 進行測試
        if (epoch + 1) % 10 == 0:
            test(epoch, H, device, G, test_loader)

        # --- Early stopping ---
        is_best = 0
        if val_loss < best_loss and epoch > D_iter:
            best_loss = val_loss
            is_best = 1  # 標記為最佳模型
            save_checkpoint( G, Glo_D, Loc_D, path=H.model_path, name='G_best.pth', full_checkpoint=False)  # 僅保存模型參數
            print(f"新的最佳 val loss {best_loss:.4f}，模型已保存")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"連續 {early_stop_patience} 次沒有改善，提早停止訓練")
                break
    
        # 當前 epoch 的損失寫入 CSV
        with open(loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{G_losses[-1]:.6f}", f"{Glo_D_losses[-1]:.6f}", f"{Loc_D_losses[-1]:.6f}",
                             f"{val_loss:.6f}", f"{Glo_G_Adv_losses[-1]:.6f}", f"{Loc_G_Adv_losses[-1]:.6f}", is_best])

        # 每 5 epoch 儲存當前進度
        #if (epoch + 1) % 5 == 0:
        if epoch+1 == G_iter :
            save_checkpoint(
                G, Glo_D, Loc_D, G_optimizer, Glo_D_optimizer, Loc_D_optimizer,
                epoch, G_losses, Glo_D_losses, Loc_D_losses, Val_losses, Glo_G_Adv_losses, Loc_G_Adv_losses,
                H.model_path, 'G_latest.pth', best_loss, early_stop_counter, full_checkpoint=True)    # 保存完整檢查點
        # 每 10 epoch 存一次模型
        if (epoch + 1) % 10 == 0:
            save_checkpoint(G, Glo_D, Loc_D, path=H.model_path, name=f'G_epoch{epoch+1}.pth', full_checkpoint=False)  # 僅保存模型參數
            # 清理舊檢查點
            existing_checkpoints = [f for f in os.listdir(H.model_path) if f.startswith('G_epoch') and f.endswith('.pth')]
            if len(existing_checkpoints) > 18:
                oldest = min(existing_checkpoints, key=lambda x: int(x.split('epoch')[1].split('.pth')[0]))
                os.remove(os.path.join(H.model_path, oldest))
        #scheduler.step(val_loss)  # 根據 val loss 動態調整學習率
        #Glo_D_scheduler.step(glo_d_loss_epoch)
        #Loc_D_scheduler.step(loc_d_loss_epoch)
        tc.cuda.empty_cache()  # 釋放未使用的記憶體

    plot_losses(G_losses, Glo_D_losses, Loc_D_losses, Val_losses, os.path.join(H.result_path, 'loss.png'))
    print("訓練結束")

if __name__ == "__main__":
    train()
