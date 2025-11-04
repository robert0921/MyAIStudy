"""CIFAR-10竞赛模型实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CompetitionNet(nn.Module):
    """CIFAR-10竞赛模型 - 基于ResNet架构"""
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2, dropout_rate)
        self.layer2 = self._make_layer(64, 128, 2, dropout_rate)
        self.layer3 = self._make_layer(128, 256, 2, dropout_rate)
        self.layer4 = self._make_layer(256, 512, 2, dropout_rate)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, dropout_rate):
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(ResBlock(in_channels, out_channels, stride=2, dropout_rate=dropout_rate))
        
        # 后续块保持维度不变
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class ResBlock(nn.Module):
    """残差块实现"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # 短接层
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out += identity
        out = F.relu(out)
        
        return out
