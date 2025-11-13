"""
第4周：Matplotlib/Seaborn可视化（重命名自 beginner_ai）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_matplotlib_basics():
    print("\n" + "="*60)
    print("📈 第4周：Matplotlib数据可视化")
    print("="*60)

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label='sin(x)')
    plt.plot(x, y2, label='cos(x)', linestyle='--')
    plt.show()


def demonstrate_sales_visualization():
    print("\n" + "="*60)
    print("💰 销售数据可视化示例")
    print("="*60)

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.linspace(1000, 1500, 365)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.randn(365) * 50
    sales = trend + seasonal + noise
    df = pd.DataFrame({'date': dates, 'sales': sales})
    df['sales_ma'] = df['sales'].rolling(window=30).mean()

    # 绘制销售折线图和移动平均
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['sales'], label='日销售额', alpha=0.6)
    plt.plot(df['date'], df['sales_ma'], label='30日移动平均', color='red')
    plt.title('2023 年销售趋势')
    plt.xlabel('日期')
    plt.ylabel('销售额')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return df


def demonstrate_seaborn_advanced():
    print("\n" + "="*60)
    print("🎨 Seaborn高级可视化")
    print("="*60)

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'age': np.random.randint(20, 60, n),
        'income': np.random.randint(3000, 20000, n),
        'spending': np.random.randint(500, 15000, n),
        'education': np.random.choice(['高中', '本科', '硕士'], n)
    })

    # 使用 seaborn 绘制收入 vs 支出 的散点图，并按教育水平着色
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='income', y='spending', hue='education', alpha=0.7)
    plt.title('收入与支出散点图（按教育水平）')
    plt.xlabel('收入')
    plt.ylabel('支出')
    plt.legend(title='教育水平')
    plt.tight_layout()
    plt.show()
    
    # 可选：返回 DataFrame 以供进一步分析
    return df


if __name__ == "__main__":
    demonstrate_matplotlib_basics()
    demonstrate_sales_visualization()
    demonstrate_seaborn_advanced()
