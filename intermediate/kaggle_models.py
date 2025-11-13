"""模块化的Kaggle竞赛模型架构"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import timm
from abc import ABC, abstractmethod

class BaseBlock(nn.Module, ABC):
    """基础模块抽象类"""
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @property
    @abstractmethod
    def out_channels(self):
        pass

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel attention
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        b, c, h, w = x.size()
        channel_pool = self.channel_pool(x).view(b, c)
        channel_att = self.sigmoid(self.channel_fc(channel_pool)).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_conv(spatial_input)
        x = x * spatial_att
        
        return x

class ModularResBlock(BaseBlock):
    """模块化残差块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.1,
        attention: str = 'se',  # 'se', 'cbam', 'none'
        activation: str = 'relu'  # 'relu', 'swish', 'mish'
    ):
        super().__init__()
        self.in_channels = in_channels
        self._out_channels = out_channels
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'mish':
            self.activation = nn.Mish(inplace=True)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 注意力机制
        if attention == 'se':
            self.attention = SEBlock(out_channels)
        elif attention == 'cbam':
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()
        
        # 短接连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        
        out += identity
        out = self.activation(out)
        
        return out
    
    @property
    def out_channels(self):
        return self._out_channels

class KaggleBackbone(nn.Module):
    """模块化的主干网络"""
    def __init__(
        self,
        input_channels: int = 3,
        initial_channels: int = 64,
        block_configs: List[Dict[str, Any]] = None,
        global_pool: str = 'adaptive'  # 'adaptive', 'max', 'avg'
    ):
        super().__init__()
        
        if block_configs is None:
            block_configs = [
                {'out_channels': 64, 'num_blocks': 2, 'stride': 1},
                {'out_channels': 128, 'num_blocks': 2, 'stride': 2},
                {'out_channels': 256, 'num_blocks': 2, 'stride': 2},
                {'out_channels': 512, 'num_blocks': 2, 'stride': 2},
            ]
        
        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # 构建主干层
        self.layers = nn.ModuleList()
        in_channels = initial_channels
        
        for config in block_configs:
            layer = self._make_layer(
                in_channels=in_channels,
                **config
            )
            self.layers.append(layer)
            in_channels = config['out_channels']
        
        # 全局池化
        if global_pool == 'adaptive':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif global_pool == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif global_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.feature_dim = in_channels
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        **kwargs
    ) -> nn.Sequential:
        """构建网络层"""
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(ModularResBlock(
            in_channels, out_channels, stride, **kwargs
        ))
        
        # 后续块
        for _ in range(1, num_blocks):
            layers.append(ModularResBlock(
                out_channels, out_channels, 1, **kwargs
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        # 全局池化
        pooled = self.global_pool(x)
        pooled = pooled.view(pooled.size(0), -1)
        
        return pooled, features

class KaggleClassifier(nn.Module):
    """模块化分类器头"""
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        dropout_rate: float = 0.5,
        hidden_dims: Optional[List[int]] = None,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        if hidden_dims is None:
            # 简单的单层分类器
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            # 多层分类器
            layers = []
            in_dim = feature_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Dropout(dropout_rate))
                layers.append(nn.Linear(in_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(in_dim, num_classes))
            
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

class KaggleModel(nn.Module):
    """完整的Kaggle竞赛模型"""
    def __init__(
        self,
        num_classes: int,
        backbone_config: Optional[Dict[str, Any]] = None,
        classifier_config: Optional[Dict[str, Any]] = None,
        pretrained_backbone: Optional[str] = None
    ):
        super().__init__()
        
        if pretrained_backbone:
            # 使用预训练模型
            self.backbone = timm.create_model(
                pretrained_backbone,
                pretrained=True,
                num_classes=0  # 去掉分类头
            )
            feature_dim = self.backbone.num_features
        else:
            # 使用自定义主干网络
            backbone_config = backbone_config or {}
            self.backbone = KaggleBackbone(**backbone_config)
            feature_dim = self.backbone.feature_dim
        
        # 分类器
        classifier_config = classifier_config or {}
        classifier_config['feature_dim'] = feature_dim
        classifier_config['num_classes'] = num_classes
        
        self.classifier = KaggleClassifier(**classifier_config)
        
        # 模型初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if hasattr(self.backbone, 'forward_features'):
            # timm 模型
            features = self.backbone.forward_features(x)
            features = self.backbone.global_pool(features)
            features = features.view(features.size(0), -1)
        else:
            # 自定义模型
            features, _ = self.backbone(x)
        
        logits = self.classifier(features)
        return logits

def create_kaggle_model(
    model_name: str,
    num_classes: int,
    **kwargs
) -> KaggleModel:
    """创建Kaggle竞赛模型的工厂函数"""
    
    model_configs = {
        'resnet50': {
            'pretrained_backbone': 'resnet50',
            'classifier_config': {
                'hidden_dims': [512],
                'dropout_rate': 0.5
            }
        },
        'efficientnet_b3': {
            'pretrained_backbone': 'efficientnet_b3',
            'classifier_config': {
                'hidden_dims': [1024, 512],
                'dropout_rate': 0.3
            }
        },
        'custom_resnet': {
            'backbone_config': {
                'block_configs': [
                    {'out_channels': 64, 'num_blocks': 3, 'stride': 1, 'attention': 'se'},
                    {'out_channels': 128, 'num_blocks': 4, 'stride': 2, 'attention': 'se'},
                    {'out_channels': 256, 'num_blocks': 6, 'stride': 2, 'attention': 'cbam'},
                    {'out_channels': 512, 'num_blocks': 3, 'stride': 2, 'attention': 'cbam'},
                ]
            },
            'classifier_config': {
                'hidden_dims': [1024, 512],
                'dropout_rate': 0.4
            }
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"未知的模型: {model_name}")
    
    config = model_configs[model_name]
    config.update(kwargs)
    
    return KaggleModel(num_classes=num_classes, **config)
