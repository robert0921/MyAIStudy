"""Kaggle 竞赛优化的数据管线"""
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import pickle
import lmdb
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class KaggleDataset(Dataset):
    """Kaggle 竞赛优化数据集"""
    def __init__(
        self,
        data_dir: str,
        csv_file: Optional[str] = None,
        transforms: Optional[A.Compose] = None,
        cache_images: bool = True,
        use_lmdb: bool = False,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.cache_images = cache_images
        self.use_lmdb = use_lmdb
        self.image_size = image_size
        
        # 加载数据信息
        if csv_file:
            self.df = pd.read_csv(csv_file)
            self.image_paths = [self.data_dir / fname for fname in self.df['filename']]
            self.labels = self.df['label'].values if 'label' in self.df.columns else None
        else:
            # 从文件夹结构推断
            self.image_paths = list(self.data_dir.glob('**/*.jpg')) + \
                              list(self.data_dir.glob('**/*.png'))
            self.labels = None
        
        # 图像缓存
        self.image_cache = {}
        if cache_images and not use_lmdb:
            self._cache_images()
        elif use_lmdb:
            self._setup_lmdb()
    
    def _cache_images(self):
        """预加载图像到内存"""
        print("缓存图像到内存...")
        
        def load_image(path):
            try:
                image = cv2.imread(str(path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)
                return str(path), image
            except Exception as e:
                print(f"加载图像失败 {path}: {e}")
                return str(path), None
        
        # 多线程并行加载
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(load_image, self.image_paths))
        
        for path, image in results:
            if image is not None:
                self.image_cache[path] = image
    
    def _setup_lmdb(self):
        """设置LMDB数据库以获得更快的IO"""
        lmdb_path = self.data_dir.parent / 'cache.lmdb'
        
        if not lmdb_path.exists():
            self._create_lmdb(lmdb_path)
        
        self.lmdb_env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    
    def _create_lmdb(self, lmdb_path: Path):
        """创建LMDB数据库"""
        print("创建LMDB数据库...")
        
        # 估算数据库大小
        map_size = len(self.image_paths) * self.image_size[0] * self.image_size[1] * 3 * 2
        
        env = lmdb.open(str(lmdb_path), map_size=map_size)
        
        with env.begin(write=True) as txn:
            for idx, path in enumerate(self.image_paths):
                try:
                    image = cv2.imread(str(path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.image_size)
                    
                    # 序列化图像
                    image_bytes = pickle.dumps(image)
                    txn.put(str(idx).encode(), image_bytes)
                    
                    if idx % 1000 == 0:
                        print(f"处理了 {idx}/{len(self.image_paths)} 张图像")
                        
                except Exception as e:
                    print(f"处理图像失败 {path}: {e}")
        
        env.close()
        print("LMDB数据库创建完成")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 获取图像
        if self.use_lmdb:
            image = self._load_from_lmdb(idx)
        elif self.cache_images:
            image = self.image_cache.get(str(self.image_paths[idx]))
            if image is None:
                image = self._load_image(self.image_paths[idx])
        else:
            image = self._load_image(self.image_paths[idx])
        
        # 应用数据增强
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 获取标签
        label = self.labels[idx] if self.labels is not None else 0
        
        return image, label
    
    def _load_from_lmdb(self, idx):
        """从LMDB加载图像"""
        with self.lmdb_env.begin() as txn:
            image_bytes = txn.get(str(idx).encode())
            image = pickle.loads(image_bytes)
        return image
    
    def _load_image(self, path):
        """直接从文件加载图像"""
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        return image

class OptimizedDataLoader:
    """优化的数据加载器工厂"""
    
    @staticmethod
    def get_competition_transforms(image_size: Tuple[int, int] = (224, 224)) -> Dict[str, A.Compose]:
        """获取竞赛级数据增强"""
        train_transforms = A.Compose([
            # 几何变换
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.3
            ),
            
            # 颜色变换
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            
            # 模糊和噪声
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.2),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1]),
            ], p=0.2),
            
            # Cutout/Mixup类似效果
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.3
            ),
            
            # 标准化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        val_transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return {'train': train_transforms, 'val': val_transforms}
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = None,
        distributed: bool = False,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ) -> DataLoader:
        """创建优化的数据加载器"""
        
        if num_workers is None:
            num_workers = min(8, mp.cpu_count())
        
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # 当使用sampler时，不能设置shuffle
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=True if shuffle else False
        )

def get_kaggle_loaders(
    train_dir: str,
    val_dir: str,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = None,
    distributed: bool = False,
    cache_images: bool = True,
    use_lmdb: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """获取优化的Kaggle竞赛数据加载器"""
    
    # 获取数据增强
    transforms = OptimizedDataLoader.get_competition_transforms(image_size)
    
    # 创建数据集
    train_dataset = KaggleDataset(
        data_dir=train_dir,
        csv_file=train_csv,
        transforms=transforms['train'],
        cache_images=cache_images,
        use_lmdb=use_lmdb,
        image_size=image_size
    )
    
    val_dataset = KaggleDataset(
        data_dir=val_dir,
        csv_file=val_csv,
        transforms=transforms['val'],
        cache_images=cache_images,
        use_lmdb=use_lmdb,
        image_size=image_size
    )
    
    # 创建数据加载器
    train_loader = OptimizedDataLoader.create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed
    )
    
    val_loader = OptimizedDataLoader.create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        distributed=distributed
    )
    
    return train_loader, val_loader
