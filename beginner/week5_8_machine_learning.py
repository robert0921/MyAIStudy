"""
第5-8周：机器学习基础（重命名自 beginner_ai）
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def week5_classification():
    print("\n" + "="*60)
    print("🌸 第5周：分类任务示例")
    print("="*60)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_model = LogisticRegression(random_state=42, max_iter=200)
    lr_model.fit(X_train_scaled, y_train)
    y_pred = lr_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  准确率: {accuracy:.4f}")
    return lr_model


def week6_regression():
    print("\n" + "="*60)
    print("🏠 第6周：回归任务示例")
    print("="*60)

    np.random.seed(42)
    n_samples = 500
    area = np.random.uniform(50, 200, n_samples)
    rooms = np.random.randint(1, 6, n_samples)
    floor = np.random.randint(1, 30, n_samples)
    year = np.random.randint(1990, 2024, n_samples)
    price = (area * 100 + rooms * 5000 + floor * 200 + (year - 1990) * 300 + np.random.randn(n_samples) * 5000)
    X = np.column_stack([area, rooms, floor, year])
    y = price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


def week7_clustering():
    print("\n" + "="*60)
    print("👥 第7周：聚类示例")
    print("="*60)

    np.random.seed(42)
    n_samples = 300
    X, y_true = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=42)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    return kmeans


def week8_model_tuning():
    print("\n" + "="*60)
    print("⚙️ 第8周：模型调参示例")
    print("="*60)

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_baseline = RandomForestClassifier(random_state=42)
    rf_baseline.fit(X_train, y_train)
    return rf_baseline


if __name__ == "__main__":
    week5_classification()
    week6_regression()
    week7_clustering()
    week8_model_tuning()
