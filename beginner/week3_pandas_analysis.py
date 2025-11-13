"""
第3周：Pandas数据处理（重命名自 beginner_ai）
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ecommerce_data(n_customers=1000, n_orders=5000):
    np.random.seed(42)
    customers = pd.DataFrame({
        'customer_id': [f'C{i:05d}' for i in range(n_customers)],
        'name': [f'用户{i}' for i in range(n_customers)],
        'age': np.random.randint(18, 65, n_customers),
        'gender': np.random.choice(['男', '女'], n_customers),
        'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n_customers),
        'register_date': pd.date_range('2023-01-01', periods=n_customers, freq='6H')
    })

    start_date = datetime(2023, 1, 1)
    orders = pd.DataFrame({
        'order_id': [f'O{i:06d}' for i in range(n_orders)],
        'customer_id': np.random.choice(customers['customer_id'], n_orders),
        'order_date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_orders)],
        'product_category': np.random.choice(['电子产品', '服装', '食品', '家居', '图书'], n_orders),
        'amount': np.random.uniform(10, 1000, n_orders).round(2),
        'status': np.random.choice(['已完成', '已取消', '退款'], n_orders, p=[0.85, 0.10, 0.05])
    })

    return customers, orders


def demonstrate_pandas_basics():
    print("\n" + "="*60)
    print("📊 第3周：Pandas数据处理与分析")
    print("="*60)

    df = pd.DataFrame({
        'name': ['张三', '李四', '王五'],
        'age': [25, 30, 35],
        'city': ['北京', '上海', '广州'],
        'salary': [8000, 12000, 15000]
    })
    print(df)
    return df


def demonstrate_data_cleaning():
    print("\n" + "="*60)
    print("🧹 数据清洗示例")
    print("="*60)

    df = pd.DataFrame({
        'name': ['张三', '李四', None, '王五', '赵六'],
        'age': [25, 30, np.nan, 35, 200],
        'salary': [8000, np.nan, 12000, 15000, 9000],
        'score': [85, 90, 95, np.nan, 88]
    })

    print("\n原始数据:")
    print(df)

    df_filled = df.copy()
    df_filled['name'].fillna('未知', inplace=True)
    df_filled['age'].fillna(df['age'].median(), inplace=True)
    df_filled['salary'].fillna(df['salary'].mean(), inplace=True)
    df_filled['score'].fillna(df['score'].mean(), inplace=True)
    df_cleaned = df_filled.copy()
    df_cleaned.loc[df_cleaned['age'] > 100, 'age'] = df_filled['age'].median()

    print("\n清理异常值后的数据:")
    print(df_cleaned)
    return df_cleaned


def analyze_ecommerce_data():
    print("\n" + "="*60)
    print("🛒 电商用户行为数据分析")
    print("="*60)

    customers, orders = generate_ecommerce_data()
    total_orders = len(orders)
    completed_orders = len(orders[orders['status'] == '已完成'])
    total_revenue = orders[orders['status'] == '已完成']['amount'].sum()
    avg_order_value = orders[orders['status'] == '已完成']['amount'].mean()

    print(f"  总订单数: {total_orders:,}")
    print(f"  完成订单: {completed_orders:,} ({completed_orders/total_orders*100:.1f}%)")
    print(f"  总收入: ¥{total_revenue:,.2f}")
    return customers, orders


if __name__ == "__main__":
    demonstrate_pandas_basics()
    demonstrate_data_cleaning()
    analyze_ecommerce_data()
