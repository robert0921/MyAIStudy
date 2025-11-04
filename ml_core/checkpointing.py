"""早停和模型检查点管理"""
import torch
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path
import json
import time

class EarlyStopping:
    """早停实现"""
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = min_delta
    
    def __call__(self, score: float, model: torch.nn.Module, epoch: int, save_path: str):
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, epoch, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, epoch, save_path)
            self.counter = 0
    
    def save_checkpoint(self, score: float, model: torch.nn.Module, epoch: int, save_path: str):
        """保存模型检查点"""
        if self.verbose:
            score_str = f'{-score:.6f}' if self.mode == 'min' else f'{score:.6f}'
            print(f'Validation score improved ({self.val_loss_min:.6f} --> {score_str}). Saving model...')
        
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
    """检查点管理器"""
    def __init__(
        self,
        save_dir: str,
        keep_best_only: bool = True,
        max_keep: int = 3
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_only = keep_best_only
        self.max_keep = max_keep
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
        # 加载现有检查点信息
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """加载现有检查点信息"""
        for checkpoint_file in self.save_dir.glob('*.pt'):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                self.checkpoints[checkpoint_file.name] = {
                    'epoch': checkpoint['epoch'],
                    'score': checkpoint['score'],
                    'timestamp': checkpoint.get('timestamp', '')
                }
            except Exception as e:
                print(f"警告：无法加载检查点 {checkpoint_file}: {e}")
    
    def save(self, model: torch.nn.Module, epoch: int, score: float):
        """保存新的检查点"""
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'checkpoint_epoch{epoch:03d}_{score:.4f}_{timestamp}.pt'
        save_path = self.save_dir / filename
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'timestamp': timestamp
        }
        torch.save(checkpoint, save_path)
        
        # 更新检查点信息
        self.checkpoints[filename] = {
            'epoch': epoch,
            'score': score,
            'timestamp': timestamp
        }
        
        # 管理检查点数量
        if self.keep_best_only:
            self._keep_best_checkpoints()
        elif len(self.checkpoints) > self.max_keep:
            self._remove_old_checkpoints()
    
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
        for filename, _ in sorted_checkpoints[1:]:
            file_path = self.save_dir / filename
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
        for filename, _ in sorted_checkpoints[:-self.max_keep]:
            file_path = self.save_dir / filename
            if file_path.exists():
                file_path.unlink()
            del self.checkpoints[filename]
    
    def load_best(self, model: torch.nn.Module) -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        if not self.checkpoints:
            return None
            
        # 按分数排序找到最佳检查点
        best_checkpoint = max(
            self.checkpoints.items(),
            key=lambda x: x[1]['score']
        )
        
        checkpoint_path = self.save_dir / best_checkpoint[0]
        if not checkpoint_path.exists():
            return None
            
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint
