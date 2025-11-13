"""训练工具和优化策略模块
包含增强的早停和检查点管理功能
"""
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
import json

class EarlyStopping:
    """早停机制实现 - 增强版本"""
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: 容忍的验证指标未改善的epoch数
            min_delta: 改善的最小变化量
            mode: 'min'表示指标越小越好，'max'表示指标越大越好
            verbose: 是否打印详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = min_delta
    
    def __call__(self, score: float, model: nn.Module, epoch: int, save_path: str):
        """检查是否应该早停"""
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, epoch, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, epoch, save_path)
            self.counter = 0
    
    def save_checkpoint(self, score: float, model: nn.Module, epoch: int, save_path: str):
        """保存模型检查点"""
        if self.verbose:
            score_str = f'{-score:.6f}' if self.mode == 'min' else f'{score:.6f}'
            prev_score = self.val_loss_min
            print(f'   验证指标改善 ({prev_score:.6f} -> {score_str})，保存模型...')
        
        # 创建保存目录
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        torch.save(checkpoint, save_path)
        
        self.val_loss_min = -score if self.mode == 'min' else score


class CheckpointManager:
    """检查点管理器 - 支持多检查点管理和自动清理"""
    def __init__(
        self,
        save_dir: str,
        keep_best_only: bool = True,
        max_keep: int = 3,
        metric_name: str = 'score'
    ):
        """
        Args:
            save_dir: 检查点保存目录
            keep_best_only: 是否只保留最佳检查点
            max_keep: 最多保留的检查点数量
            metric_name: 用于排序的指标名称
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_only = keep_best_only
        self.max_keep = max_keep
        self.metric_name = metric_name
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
        # 加载现有检查点信息
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """加载现有检查点信息"""
        for checkpoint_file in self.save_dir.glob('*.pt'):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                self.checkpoints[checkpoint_file.name] = {
                    'epoch': checkpoint.get('epoch', 0),
                    'score': checkpoint.get('score', 0.0),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'path': str(checkpoint_file)
                }
            except Exception as e:
                print(f"   警告：无法加载检查点 {checkpoint_file}: {e}")
    
    def save(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int, 
        score: float,
        additional_info: Optional[Dict] = None
    ) -> str:
        """保存新的检查点"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'checkpoint_epoch{epoch:03d}_score{score:.4f}_{timestamp}.pt'
        save_path = self.save_dir / filename
        
        # 构建检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'timestamp': timestamp
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # 保存检查点
        torch.save(checkpoint, save_path)
        
        # 更新检查点信息
        self.checkpoints[filename] = {
            'epoch': epoch,
            'score': score,
            'timestamp': timestamp,
            'path': str(save_path)
        }
        
        # 管理检查点数量
        if self.keep_best_only:
            self._keep_best_checkpoints()
        elif len(self.checkpoints) > self.max_keep:
            self._remove_old_checkpoints()
        
        return str(save_path)
    
    def _keep_best_checkpoints(self):
        """只保留最好的检查点"""
        if len(self.checkpoints) <= 1:
            return
            
        # 按分数排序
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # 删除多余的检查点
        for filename, info in sorted_checkpoints[1:]:
            file_path = Path(info['path'])
            if file_path.exists():
                file_path.unlink()
            del self.checkpoints[filename]
    
    def _remove_old_checkpoints(self):
        """删除旧的检查点，保持数量在限制之内"""
        if len(self.checkpoints) <= self.max_keep:
            return
            
        # 按时间戳排序
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # 删除最旧的检查点
        for filename, info in sorted_checkpoints[:-self.max_keep]:
            file_path = Path(info['path'])
            if file_path.exists():
                file_path.unlink()
            del self.checkpoints[filename]
    
    def load_best(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None) -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        if not self.checkpoints:
            print("   没有找到检查点")
            return None
            
        # 按分数排序找到最佳检查点
        best_checkpoint = max(
            self.checkpoints.items(),
            key=lambda x: x[1]['score']
        )
        
        checkpoint_path = best_checkpoint[1]['path']
        if not Path(checkpoint_path).exists():
            print(f"   检查点文件不存在: {checkpoint_path}")
            return None
            
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"   已加载最佳检查点: epoch={checkpoint['epoch']}, score={checkpoint['score']:.4f}")
        return checkpoint
    
    def load_latest(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None) -> Optional[Dict[str, Any]]:
        """加载最新的检查点"""
        if not self.checkpoints:
            return None
            
        # 按时间戳排序找到最新检查点
        latest_checkpoint = max(
            self.checkpoints.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        checkpoint_path = latest_checkpoint[1]['path']
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_summary(self) -> Dict[str, Any]:
        """获取检查点摘要信息"""
        if not self.checkpoints:
            return {'total': 0, 'best': None, 'latest': None}
        
        best = max(self.checkpoints.items(), key=lambda x: x[1]['score'])
        latest = max(self.checkpoints.items(), key=lambda x: x[1]['timestamp'])
        
        return {
            'total': len(self.checkpoints),
            'best': best[1],
            'latest': latest[1],
            'checkpoints': list(self.checkpoints.values())
        }


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
                 mixed_precision: bool = True,
                 use_early_stopping: bool = True,
                 checkpoint_mode: str = 'best'):
        """
        Args:
            checkpoint_mode: 'best' 只保留最佳模型, 'all' 保留所有检查点, 'last_n' 保留最近N个
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = device
        self.mixed_precision = mixed_precision
        self.use_early_stopping = use_early_stopping
        self.checkpoint_mode = checkpoint_mode

class Trainer:
    """模型训练器，支持分布式训练、混合精度和增强的检查点管理"""
    def __init__(self,
                 model: nn.Module,
                 config: TrainerConfig,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 local_rank: int = -1,
                 save_dir: Optional[str] = None):
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
        
        # 早停和检查点管理
        self.save_dir = save_dir or 'checkpoints'
        if config.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                mode='max',  # 准确率越高越好
                verbose=True
            )
        else:
            self.early_stopping = None
        
        # 检查点管理器
        keep_best_only = (config.checkpoint_mode == 'best')
        max_keep = 3 if config.checkpoint_mode == 'last_n' else 999
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            keep_best_only=keep_best_only,
            max_keep=max_keep
        )
    
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
    
    def train(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """完整训练流程 - 增强版本，支持早停和检查点管理"""
        start_time = time.time()
        
        if save_dir:
            self.save_dir = save_dir
            self.checkpoint_manager.save_dir = Path(save_dir)
            self.checkpoint_manager.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"开始训练 - 最大epochs: {self.config.max_epochs}")
        print(f"保存目录: {self.save_dir}")
        print(f"早停: {'启用' if self.early_stopping else '禁用'}")
        print(f"检查点模式: {self.config.checkpoint_mode}")
        print(f"{'='*70}\n")
        
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics['loss'])
                self.val_accs.append(val_metrics['accuracy'])
                
                epoch_time = time.time() - epoch_start
                
                # 打印进度
                print(f'Epoch {epoch+1:03d}/{self.config.max_epochs} ({epoch_time:.1f}s) - '
                      f'Train Loss: {train_metrics["loss"]:.4f}, '
                      f'Train Acc: {train_metrics["accuracy"]:.2f}%, '
                      f'Val Loss: {val_metrics["loss"]:.4f}, '
                      f'Val Acc: {val_metrics["accuracy"]:.2f}%')
                
                # 保存检查点
                checkpoint_path = self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    score=val_metrics['accuracy'],
                    additional_info={
                        'train_loss': train_metrics['loss'],
                        'train_acc': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy']
                    }
                )
                
                # 更新最佳验证准确率
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # 早停检查
                if self.early_stopping:
                    self.early_stopping(
                        score=val_metrics['accuracy'],
                        model=self.model,
                        epoch=epoch,
                        save_path=str(Path(self.save_dir) / 'early_stop_best.pt')
                    )
                    
                    if self.early_stopping.early_stop:
                        print(f'\n早停触发于 epoch {epoch+1}')
                        print(f'最佳验证准确率: {self.best_val_acc:.2f}%')
                        break
            else:
                # 仅训练，无验证
                epoch_time = time.time() - epoch_start
                print(f'Epoch {epoch+1:03d}/{self.config.max_epochs} ({epoch_time:.1f}s) - '
                      f'Train Loss: {train_metrics["loss"]:.4f}, '
                      f'Train Acc: {train_metrics["accuracy"]:.2f}%')
        
        total_time = time.time() - start_time
        
        # 打印训练摘要
        print(f"\n{'='*70}")
        print(f"训练完成!")
        print(f"总用时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 检查点摘要
        checkpoint_summary = self.checkpoint_manager.get_summary()
        print(f"\n检查点摘要:")
        print(f"  总检查点数: {checkpoint_summary['total']}")
        if checkpoint_summary['best']:
            print(f"  最佳检查点: epoch={checkpoint_summary['best']['epoch']}, "
                  f"score={checkpoint_summary['best']['score']:.2f}%")
        print(f"{'='*70}\n")
        
        return {
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'total_time': total_time,
            'checkpoint_summary': checkpoint_summary
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
