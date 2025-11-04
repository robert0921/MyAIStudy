"""训练工具和优化策略模块"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, List
import numpy as np
from pathlib import Path
import time

class TrainerConfig:
    """训练配置类"""
    def __init__(self,
                 max_epochs: int = 100,
                 batch_size: int = 128,
                 learning_rate: float = 0.1,
                 num_workers: int = 4,
                 weight_decay: float = 5e-4,
                 patience: int = 10,
                 grad_clip: float = 5.0,
                 device: str = 'cuda',
                 mixed_precision: bool = True):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = device
        self.mixed_precision = mixed_precision

class Trainer:
    """模型训练器，支持分布式训练和混合精度"""
    def __init__(self,
                 model: nn.Module,
                 config: TrainerConfig,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 local_rank: int = -1):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_rank = local_rank
        
        # 设置设备
        if local_rank != -1:
            self.device = torch.device(f'cuda:{local_rank}')
            self.model = DDP(model.to(self.device), device_ids=[local_rank])
        else:
            self.device = torch.device(config.device)
            self.model = model.to(self.device)
        
        # 优化器设置
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.max_epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 训练状态
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # 指标记录
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型性能"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """完整训练流程"""
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics['loss'])
                self.val_accs.append(val_metrics['accuracy'])
                
                # 早停检查
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0
                    if save_dir:
                        self.save_checkpoint(save_dir, 'best_model.pth')
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.config.patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                
                print(f'Epoch {epoch+1}/{self.config.max_epochs} - '
                      f'Train Loss: {train_metrics["loss"]:.4f}, '
                      f'Train Acc: {train_metrics["accuracy"]:.2f}%, '
                      f'Val Loss: {val_metrics["loss"]:.4f}, '
                      f'Val Acc: {val_metrics["accuracy"]:.2f}%')
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f} seconds')
        
        return {
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs
        }
    
    def save_checkpoint(self, save_dir: str, filename: str):
        """保存检查点"""
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__
        }
        
        torch.save(state, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.best_val_acc = state['best_val_acc']
