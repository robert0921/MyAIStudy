"""
演示早停和检查点管理功能
展示新的训练系统特性
"""
import torch
import torch.nn as nn
from pathlib import Path

print("="*70)
print("🎯 早停和检查点管理功能演示")
print("="*70)

# 检查PyTorch是否可用
try:
    from ml_core.training import Trainer, TrainerConfig, EarlyStopping, CheckpointManager
    from ml_core.models_torch import CIFAR10Net
    from ml_core.data import get_cifar10_loaders
    
    print("\n✓ 所有必需模块已加载")
    
    # 1. 演示早停配置
    print("\n" + "="*70)
    print("1️⃣ 早停机制配置")
    print("="*70)
    
    config_with_early_stop = TrainerConfig(
        max_epochs=100,
        batch_size=64,
        learning_rate=0.001,
        patience=10,              # 10个epoch未改善则停止
        use_early_stopping=True,  # 启用早停
        checkpoint_mode='best',   # 只保留最佳模型
        mixed_precision=True
    )
    
    print("配置参数:")
    print(f"  • max_epochs: {config_with_early_stop.max_epochs}")
    print(f"  • patience: {config_with_early_stop.patience}")
    print(f"  • use_early_stopping: {config_with_early_stop.use_early_stopping}")
    print(f"  • checkpoint_mode: {config_with_early_stop.checkpoint_mode}")
    print(f"  • mixed_precision: {config_with_early_stop.mixed_precision}")
    
    # 2. 演示检查点模式
    print("\n" + "="*70)
    print("2️⃣ 检查点管理模式")
    print("="*70)
    
    modes = {
        'best': '只保留最佳模型（推荐）',
        'last_n': '保留最近N个检查点',
        'all': '保留所有检查点'
    }
    
    for mode, description in modes.items():
        print(f"  • {mode:10s} - {description}")
    
    # 3. 演示训练配置对比
    print("\n" + "="*70)
    print("3️⃣ 不同场景的配置示例")
    print("="*70)
    
    # 快速原型配置
    print("\n📌 快速原型开发（不使用早停）")
    quick_config = TrainerConfig(
        max_epochs=10,
        batch_size=128,
        learning_rate=0.01,
        use_early_stopping=False,  # 禁用早停
        checkpoint_mode='best',
        mixed_precision=True
    )
    print(f"  • epochs: {quick_config.max_epochs}, early_stop: {quick_config.use_early_stopping}")
    
    # 生产训练配置
    print("\n📌 生产环境训练（启用早停+检查点）")
    production_config = TrainerConfig(
        max_epochs=200,
        batch_size=64,
        learning_rate=0.001,
        patience=15,
        use_early_stopping=True,  # 启用早停
        checkpoint_mode='last_n',  # 保留最近3个
        mixed_precision=True
    )
    print(f"  • epochs: {production_config.max_epochs}, patience: {production_config.patience}")
    print(f"  • early_stop: {production_config.use_early_stopping}, mode: {production_config.checkpoint_mode}")
    
    # 4. 演示检查点管理器
    print("\n" + "="*70)
    print("4️⃣ CheckpointManager 使用示例")
    print("="*70)
    
    # 创建临时检查点目录
    test_dir = Path("demo_checkpoints")
    test_dir.mkdir(exist_ok=True)
    
    manager = CheckpointManager(
        save_dir=str(test_dir),
        keep_best_only=True,
        max_keep=3
    )
    
    print(f"✓ 创建检查点管理器")
    print(f"  • 保存目录: {manager.save_dir}")
    print(f"  • 只保留最佳: {manager.keep_best_only}")
    print(f"  • 最大保留数: {manager.max_keep}")
    
    # 获取摘要
    summary = manager.get_summary()
    print(f"\n检查点摘要:")
    print(f"  • 当前检查点总数: {summary['total']}")
    
    # 5. 模拟训练流程
    print("\n" + "="*70)
    print("5️⃣ 完整训练流程示例")
    print("="*70)
    
    print("\n🔧 创建模型和数据加载器...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"  • 设备: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"  • 设备: {device}")
    
    # 创建小批量数据用于演示
    print("\n  • 创建数据加载器（演示用小批量）")
    try:
        train_loader, val_loader = get_cifar10_loaders(
            batch_size=32,
            num_workers=0,  # 避免Windows的多进程问题
            distributed=False
        )
        print(f"    ✓ 训练集批次数: {len(train_loader)}")
        print(f"    ✓ 验证集批次数: {len(val_loader)}")
    except Exception as e:
        print(f"    ⚠️  数据加载失败: {e}")
        print(f"    💡 提示: 首次运行会自动下载CIFAR-10数据集")
        
        # 清理
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        print("\n" + "="*70)
        print("✅ 演示完成（数据准备阶段）")
        print("="*70)
        exit(0)
    
    print("\n  • 创建CIFAR10模型")
    model = CIFAR10Net()
    print(f"    ✓ 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n🏃 配置训练器...")
    demo_config = TrainerConfig(
        max_epochs=3,  # 仅演示，使用3个epoch
        batch_size=32,
        learning_rate=0.001,
        patience=2,
        use_early_stopping=True,
        checkpoint_mode='best',
        mixed_precision=torch.cuda.is_available()  # 仅GPU启用混合精度
    )
    
    trainer = Trainer(
        model=model,
        config=demo_config,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=str(test_dir)
    )
    
    print(f"  ✓ 训练器配置完成")
    print(f"    • 最大epochs: {demo_config.max_epochs}")
    print(f"    • 早停容忍度: {demo_config.patience}")
    print(f"    • 检查点目录: {test_dir}")
    
    print("\n🚀 开始训练（这将需要几分钟）...")
    print("    提示: 早停将自动监控验证准确率")
    print("    提示: 检查点将自动保存最佳模型")
    
    results = trainer.train()
    
    # 显示训练结果
    print("\n📊 训练结果:")
    print(f"  • 最佳验证准确率: {results['best_val_acc']:.2f}%")
    print(f"  • 总训练时间: {results['total_time']:.2f}秒")
    print(f"  • 最终训练损失: {results['train_loss'][-1]:.4f}")
    print(f"  • 最终验证损失: {results['val_loss'][-1]:.4f}")
    
    # 显示检查点信息
    checkpoint_summary = results['checkpoint_summary']
    print(f"\n💾 检查点信息:")
    print(f"  • 保存的检查点数: {checkpoint_summary['total']}")
    if checkpoint_summary['best']:
        print(f"  • 最佳检查点: epoch {checkpoint_summary['best']['epoch']}, "
              f"准确率 {checkpoint_summary['best']['score']:.2f}%")
    
    # 6. 演示加载最佳模型
    print("\n" + "="*70)
    print("6️⃣ 加载最佳检查点")
    print("="*70)
    
    print("\n🔄 重新加载最佳模型...")
    new_model = CIFAR10Net()
    checkpoint = trainer.checkpoint_manager.load_best(new_model)
    
    if checkpoint:
        print(f"  ✓ 成功加载最佳模型")
        print(f"    • Epoch: {checkpoint['epoch']}")
        print(f"    • 验证准确率: {checkpoint['score']:.2f}%")
        print(f"    • 训练准确率: {checkpoint['train_acc']:.2f}%")
        print(f"    • 时间戳: {checkpoint['timestamp']}")
    
    # 清理
    import shutil
    if test_dir.exists():
        print(f"\n🧹 清理演示文件...")
        shutil.rmtree(test_dir)
        print(f"  ✓ 已删除 {test_dir}")
    
    print("\n" + "="*70)
    print("✅ 演示完成！")
    print("="*70)
    print("\n💡 要点总结:")
    print("  1. use_early_stopping=True 启用早停机制")
    print("  2. patience 参数控制容忍的未改善epoch数")
    print("  3. checkpoint_mode 控制检查点保存策略")
    print("  4. CheckpointManager 自动管理检查点文件")
    print("  5. load_best() 方法可恢复最佳模型")
    print("\n📚 更多信息请查看: ml_core/training.py")
    
except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print("\n💡 请确保已安装必要的依赖:")
    print("   pip install torch torchvision numpy")
except Exception as e:
    print(f"\n❌ 演示过程出错: {e}")
    import traceback
    traceback.print_exc()
