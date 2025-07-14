from torch.utils.data import Dataset, DataLoader
import torch
import os
import multiprocessing

class CelebADataset(Dataset):
    def __init__(self, split='train', max_samples=None):  # split : 'train' 'val' 'test'
        assert split in ['train', 'val', 'test'], f"無效的 split: {split}"
        self.path = f'data/img_align_celeba/processed/CelebA/{split}'
        self.samples = sorted(os.listdir(self.path))

        # 指定 max_samples，選取指定數量的圖像
        if max_samples is not None:
            if max_samples > len(self.samples):
                raise ValueError(f"max_samples ({max_samples}) 超過資料集大小 ({len(self.samples)})")
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        original, corrupted = torch.load(os.path.join(self.path, self.samples[index]))
        return original, corrupted

def get_dataloader(split='train', batch_size=32, shuffle=True, max_samples=None):
    dataset = CelebADataset(split=split, max_samples=max_samples)
    num_workers = min(8, multiprocessing.cpu_count())  # 最多用 8 個 worker
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

#train_loader = get_dataloader(split='train', batch_size=64)  # 訓練集

#val_loader = get_dataloader(split='val', batch_size=64, shuffle=False)  # # 驗證集（不需要 shuffle）

#test_loader = get_dataloader(split='test', batch_size=64, shuffle=False, max_samples=500)  # 測試集（只使用 500 張圖做推論

'''
class CelebADataset(Dataset):
    def __init__(self, shuf=0, max_samples=None):
        if(shuf):  # shuffle = 1(隨機排列)
            self.path = 'data/img_align_celeba/processed/CelebA/shuffle'
        else:
            self.path = 'data/img_align_celeba/processed/CelebA/ordered'
        self.samples = sorted(os.listdir(self.path))

        # 如果指定了 max_samples，選取指定數量的圖像
        if max_samples is not None:
            if max_samples > len(self.samples):
                raise ValueError(f"max_samples ({max_samples}) 超過資料集大小 ({len(self.samples)})")
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        original, masked = torch.load(os.path.join(self.path, self.samples[index]))
        return original, masked

def get_dataloader(batch_size=32, shuffle=False, shuf=0, max_samples=None):
    dataset = CelebADataset(shuf=shuf, max_samples=max_samples)
    num_workers = min(8, multiprocessing.cpu_count())  # 最多用 8 個 worker
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
'''