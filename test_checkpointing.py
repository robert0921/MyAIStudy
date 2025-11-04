"""测试早停和检查点管理功能"""
import sys
import os

def test_imports():
    """测试导入"""
    print("\n" + "="*60)
    print("测试1: 检查模块导入")
    print("="*60)
    
    try:
        from ml_core.training import EarlyStopping, CheckpointManager, Trainer, TrainerConfig
        print("✓ 成功导入 EarlyStopping")
        print("✓ 成功导入 CheckpointManager")
        print("✓ 成功导入 Trainer")
        print("✓ 成功导入 TrainerConfig")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_early_stopping():
    """测试早停机制"""
    print("\n" + "="*60)
    print("测试2: 早停机制")
    print("="*60)
    
    try:
        from ml_core.training import EarlyStopping
        
        # 创建早停对象
        early_stopping = EarlyStopping(
            patience=3,
            min_delta=0.01,
            mode='max',
            verbose=True
        )
        
        print("✓ 成功创建 EarlyStopping 对象")
        print(f"  - patience: {early_stopping.patience}")
        print(f"  - min_delta: {early_stopping.min_delta}")
        print(f"  - mode: {early_stopping.mode}")
        print(f"  - counter: {early_stopping.counter}")
        print(f"  - early_stop: {early_stopping.early_stop}")
        
        return True
    except Exception as e:
        print(f"✗ 早停测试失败: {e}")
        return False

def test_checkpoint_manager():
    """测试检查点管理器"""
    print("\n" + "="*60)
    print("测试3: 检查点管理器")
    print("="*60)
    
    try:
        from ml_core.training import CheckpointManager
        from pathlib import Path
        
        # 创建临时目录
        test_dir = Path("test_checkpoints_temp")
        test_dir.mkdir(exist_ok=True)
        
        # 创建检查点管理器
        manager = CheckpointManager(
            save_dir=str(test_dir),
            keep_best_only=True,
            max_keep=3
        )
        
        print("✓ 成功创建 CheckpointManager 对象")
        print(f"  - save_dir: {manager.save_dir}")
        print(f"  - keep_best_only: {manager.keep_best_only}")
        print(f"  - max_keep: {manager.max_keep}")
        
        # 获取摘要
        summary = manager.get_summary()
        print(f"  - 检查点总数: {summary['total']}")
        
        # 清理
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"✗ 检查点管理器测试失败: {e}")
        return False

def test_trainer_config():
    """测试训练配置"""
    print("\n" + "="*60)
    print("测试4: 训练配置")
    print("="*60)
    
    try:
        from ml_core.training import TrainerConfig
        
        # 创建配置对象
        config = TrainerConfig(
            max_epochs=100,
            batch_size=64,
            learning_rate=0.001,
            patience=10,
            use_early_stopping=True,
            checkpoint_mode='best'
        )
        
        print("✓ 成功创建 TrainerConfig 对象")
        print(f"  - max_epochs: {config.max_epochs}")
        print(f"  - batch_size: {config.batch_size}")
        print(f"  - learning_rate: {config.learning_rate}")
        print(f"  - patience: {config.patience}")
        print(f"  - use_early_stopping: {config.use_early_stopping}")
        print(f"  - checkpoint_mode: {config.checkpoint_mode}")
        print(f"  - mixed_precision: {config.mixed_precision}")
        
        return True
    except Exception as e:
        print(f"✗ 训练配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("🧪 早停和检查点管理功能测试")
    print("="*60)
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("早停机制", test_early_stopping()))
    results.append(("检查点管理器", test_checkpoint_manager()))
    results.append(("训练配置", test_trainer_config()))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
