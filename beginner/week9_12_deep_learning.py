"""
第9-12周：深度学习入门与PyTorch（重命名自 beginner_ai）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_classes=2):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x


def week9_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)
    n_train = 1000
    n_test = 200
    X_train = torch.randn(n_train, 1, 28, 28)
    y_train = torch.randint(0, 10, (n_train,))
    X_test = torch.randn(n_test, 1, 28, 28)
    y_test = torch.randint(0, 10, (n_test,))
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)

    # 测试评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    acc = correct / total if total > 0 else 0.0

    # 打印摘要
    total_params = sum(p.numel() for p in model.parameters())
    print(f"week9_mnist: params={total_params}, test_acc={acc:.4f}")
    # 示例预测
    with torch.no_grad():
        sample_data, sample_target = next(iter(test_loader))
        sample_out = model(sample_data.to(device))
        sample_preds = sample_out.argmax(dim=1).cpu().numpy()[:5]
        print("  示例真实 vs 预测:", sample_target.numpy()[:5], "->", sample_preds)

    return model


def week10_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    n_train = 1000
    n_test = 200
    X_train = torch.randn(n_train, 3, 32, 32)
    y_train = torch.randint(0, 10, (n_train,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    model = SimpleCNN(num_classes=10, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 简单评估（使用训练数据的一个小子集作为测试替代）
    model.eval()
    with torch.no_grad():
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        out = model(data)
        preds = out.argmax(dim=1)
        acc = (preds == target).float().mean().item()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"week10_cifar10: params={total_params}, sample_acc={acc:.4f}")
    print("  示例真实 vs 预测:", target.cpu().numpy()[:5], "->", preds.cpu().numpy()[:5])

    return model


def week11_text_classification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    vocab_size = 1000
    seq_length = 50
    n_train = 800
    n_test = 200
    X_train = torch.randint(1, vocab_size, (n_train, seq_length))
    y_train = torch.randint(0, 2, (n_train,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = SimpleRNN(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 简单评估（使用训练数据的一个小子集作为测试替代）
    model.eval()
    with torch.no_grad():
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        out = model(data)
        preds = out.argmax(dim=1)
        acc = (preds == target).float().mean().item()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"week11_text_classification: params={total_params}, sample_acc={acc:.4f}")
    print("  示例真实 vs 预测:", target.cpu().numpy()[:5], "->", preds.cpu().numpy()[:5])

    return model


def week12_comprehensive_project():
    """
    第12周综合项目：从数据加载到模型部署的完整Pipeline
    
    学习重点：
    - 完整的深度学习流程：数据准备 → 模型训练 → 评估 → 保存 → 部署
    - 模型持久化（保存和加载）
    - 使用Gradio构建交互式Web应用
    
    实践输出：
    - 搭建一个网页端手写数字识别应用（Gradio）
    """
    print("\n" + "="*70)
    print("🎯 第12周综合项目 - 完整深度学习Pipeline演示")
    print("="*70)
    
    # 检查Gradio是否安装
    try:
        import gradio as gr
        gradio_available = True
    except (ImportError, Exception) as e:
        print("⚠️  Gradio不可用，将跳过网页演示部分。")
        print(f"   原因: {type(e).__name__}")
        print("   安装命令: pip install gradio")
        gradio_available = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}")
    
    # ==========================================
    # 步骤1: 数据加载与准备
    # ==========================================
    print("\n📊 步骤1: 数据加载与准备")
    print("-" * 70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟MNIST数据集（实际项目中应使用真实数据）
    n_train = 1000
    n_test = 200
    
    print(f"  ✓ 训练集样本数: {n_train}")
    print(f"  ✓ 测试集样本数: {n_test}")
    print(f"  ✓ 图像尺寸: 28x28 灰度图")
    print(f"  ✓ 类别数: 10 (数字 0-9)")
    
    X_train = torch.randn(n_train, 1, 28, 28)
    y_train = torch.randint(0, 10, (n_train,))
    X_test = torch.randn(n_test, 1, 28, 28)
    y_test = torch.randint(0, 10, (n_test,))
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # ==========================================
    # 步骤2: 模型构建
    # ==========================================
    print("\n🏗️  步骤2: 模型构建")
    print("-" * 70)
    
    model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ 模型架构: SimpleMLP")
    print(f"  ✓ 总参数量: {total_params:,}")
    print(f"  ✓ 可训练参数: {trainable_params:,}")
    print(f"  ✓ 模型结构:")
    print(f"     - 输入层: 784 → 隐藏层1: 128 → 隐藏层2: 64 → 输出层: 10")
    print(f"     - 激活函数: ReLU")
    print(f"     - 正则化: Dropout(0.2)")
    
    # ==========================================
    # 步骤3: 模型训练
    # ==========================================
    print("\n🚀 步骤3: 模型训练")
    print("-" * 70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    
    print(f"  ✓ 损失函数: CrossEntropyLoss")
    print(f"  ✓ 优化器: Adam (lr=0.001)")
    print(f"  ✓ 训练轮数: {num_epochs}")
    print()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
        
        avg_loss = epoch_loss / total
        train_acc = correct / total
        print(f"  Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")
    
    # ==========================================
    # 步骤4: 模型评估
    # ==========================================
    print("\n📈 步骤4: 模型评估")
    print("-" * 70)
    
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    
    test_acc = correct / total
    avg_test_loss = test_loss / total
    
    print(f"  ✓ 测试集准确率: {test_acc:.4f} ({correct}/{total})")
    print(f"  ✓ 测试集损失: {avg_test_loss:.4f}")
    
    # 显示几个预测示例
    with torch.no_grad():
        sample_data, sample_target = next(iter(test_loader))
        sample_data = sample_data.to(device)
        sample_output = model(sample_data)
        sample_preds = sample_output.argmax(dim=1).cpu().numpy()[:8]
        sample_true = sample_target.numpy()[:8]
        
        print(f"\n  示例预测:")
        print(f"    真实标签: {sample_true}")
        print(f"    预测标签: {sample_preds}")
        print(f"    预测正确: {(sample_preds == sample_true).sum()}/8")
    
    # ==========================================
    # 步骤5: 模型保存
    # ==========================================
    print("\n💾 步骤5: 模型保存")
    print("-" * 70)
    
    import os
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, "mnist_mlp_week12.pth")
    
    # 保存完整的模型检查点
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': avg_test_loss,
    }
    torch.save(checkpoint, model_path)
    
    print(f"  ✓ 模型已保存至: {model_path}")
    print(f"  ✓ 模型大小: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"  ✓ 保存信息: epoch={num_epochs}, acc={test_acc:.4f}")
    
    # 演示模型加载
    print("\n  演示模型加载...")
    loaded_model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10).to(device)
    checkpoint = torch.load(model_path)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    print(f"  ✓ 模型加载成功！加载的准确率: {checkpoint['test_acc']:.4f}")
    
    # ==========================================
    # 步骤6: Gradio Web应用部署
    # ==========================================
    if not gradio_available:
        print("\n⚠️  步骤6: 跳过Web应用部署（未安装gradio）")
        print("="*70)
        return model
    
    print("\n🌐 步骤6: Gradio Web应用部署")
    print("-" * 70)
    
    def predict_digit(img):
        """
        从Gradio画布输入预测手写数字
        
        Args:
            img: numpy array, Gradio传入的图像数据
            
        Returns:
            dict: 各个数字类别的预测概率
        """
        if img is None:
            return {str(i): 0.0 for i in range(10)}
        
        # 图像预处理
        x = np.array(img).astype(np.float32)
        
        # 如果是RGB图，转为灰度
        if x.ndim == 3:
            x = x.mean(axis=2)
        
        # 归一化到[0,1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # 调整大小到28x28
        try:
            from scipy.ndimage import zoom
            if x.shape != (28, 28):
                zoom_factor = (28 / x.shape[0], 28 / x.shape[1])
                x = zoom(x, zoom_factor, order=1)
        except ImportError:
            # 如果scipy不可用，使用简单的resize
            if x.shape != (28, 28):
                # 使用简单的双线性插值
                from PIL import Image
                try:
                    img = Image.fromarray((x * 255).astype(np.uint8))
                    img = img.resize((28, 28), Image.BILINEAR)
                    x = np.array(img).astype(np.float32) / 255.0
                except ImportError:
                    # PIL也不可用，使用最简单的方法
                    x = np.resize(x, (28, 28))
        
        # 转换为tensor
        x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # 预测
        with torch.no_grad():
            output = loaded_model(x_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = output.argmax(dim=1).item()
        
        # 返回概率字典
        result = {str(i): float(probs[i]) for i in range(10)}
        
        return result
    
    # 创建Gradio界面（兼容Gradio 4.x API）
    try:
        iface = gr.Interface(
            fn=predict_digit,
            inputs=gr.Image(
                type='numpy',
                label='📝 在此绘制数字（0-9）',
                height=280,
                width=280
            ),
            outputs=gr.Label(
                num_top_classes=3,
                label='🎯 预测结果（Top-3）'
            ),
            title='🔢 第12周综合项目：手写数字识别系统',
            description='''
            ### 📚 项目说明
            这是一个完整的深度学习Pipeline演示：
            1. ✅ 数据加载与预处理
            2. ✅ 模型训练（SimpleMLP）
            3. ✅ 模型评估与保存
            4. ✅ Web应用部署（Gradio）
            
            ### 🎨 使用方法
            1. 在左侧上传或绘制一个数字图片（0-9）
            2. 点击"Submit"按钮
            3. 查看右侧的预测结果和概率分布
            
            ### ⚙️ 模型信息
            - 架构：3层全连接神经网络（784→128→64→10）
            - 参数量：''' + f"{total_params:,}" + '''
            - 测试准确率：''' + f"{test_acc:.2%}" + '''
            
            ### 💡 提示
            - 可以上传图片或使用网络摄像头
            - 图片会自动调整到28x28
            - 确保数字清晰可见
            ''',
            examples=[],
            theme='default',
            allow_flagging='never'
        )
    except TypeError as e:
        # 如果API参数不兼容，使用最简单的方式
        print(f"  ⚠️  Gradio API版本兼容性问题: {e}")
        print("  ℹ️  尝试使用简化配置...")
        iface = gr.Interface(
            fn=predict_digit,
            inputs=gr.Image(type='numpy', label='📝 上传或绘制数字图片'),
            outputs=gr.Label(num_top_classes=3, label='🎯 预测结果'),
            title='🔢 手写数字识别系统',
            description='上传图片或绘制数字，查看AI预测结果'
        )
    
    print("  ✓ Gradio Web应用创建成功！")
    print("  ✓ 支持功能: 画布绘制、实时预测、概率分布")
    print("\n  🚀 启动方式:")
    print("     方法1: iface.launch()              # 启动本地服务器")
    print("     方法2: iface.launch(share=True)    # 生成公网链接（临时）")
    print("\n  💻 启动后访问: http://localhost:7860")
    
    print("\n" + "="*70)
    print("✅ 第12周综合项目完成！")
    print("="*70)
    print("\n📝 项目总结:")
    print(f"  • 完整Pipeline: 数据→训练→评估→保存→部署 ✓")
    print(f"  • 模型准确率: {test_acc:.2%}")
    print(f"  • 模型已保存: {model_path}")
    print(f"  • Web应用就绪: 调用 iface.launch() 即可启动")
    print("\n🎓 学习成果:")
    print("  ✓ 掌握了完整的深度学习开发流程")
    print("  ✓ 学会了模型的保存和加载")
    print("  ✓ 能够构建交互式Web应用")
    print("  ✓ 具备了端到端项目开发能力")
    print("\n🎉 恭喜完成第12周学习！准备好进入进阶版了吗？\n")
    
    return iface


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 第9-12周：深度学习入门与PyTorch")
    print("="*70)
    print("\n📋 本模块包含以下内容:")
    print("  Week 9:  MNIST手写数字识别（MLP）")
    print("  Week 10: CIFAR-10图像分类（CNN）")
    print("  Week 11: 文本情感分类（RNN/LSTM）")
    print("  Week 12: 综合项目 - 完整Pipeline与Web部署")
    print("\n" + "="*70 + "\n")
    
    # Week 9
    print("▶️  运行 Week 9 - MNIST手写数字识别")
    print("-" * 70)
    week9_mnist()
    
    print("\n")
    
    # Week 10
    print("▶️  运行 Week 10 - CIFAR-10图像分类")
    print("-" * 70)
    week10_cifar10()
    
    print("\n")
    
    # Week 11
    print("▶️  运行 Week 11 - 文本情感分类")
    print("-" * 70)
    week11_text_classification()
    
    print("\n")
    
    # Week 12
    print("▶️  运行 Week 12 - 综合项目")
    iface = week12_comprehensive_project()
    
    # 可选：自动启动Gradio应用
    # 如果想自动启动Web应用，取消下面的注释
    # if iface is not None:
    #     print("\n🚀 正在启动Gradio Web应用...")
    #     iface.launch(share=False, server_name="127.0.0.1", server_port=7860)
