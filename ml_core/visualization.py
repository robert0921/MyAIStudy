"""数据可视化模块，提供数据分析和可视化功能"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import time
from pathlib import Path

@st.cache_data
def load_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    加载和预处理数据,使用缓存提升性能
    
    Args:
        path: 数据文件路径，如果为None则生成示例数据
    """
    if path is None:
        # 生成示例数据
        df = pd.DataFrame(np.random.randn(100000, 4), columns=['A', 'B', 'C', 'D'])
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df))
    else:
        # 分块读取大文件
        df = pd.read_csv(path, chunksize=100000)
        df = pd.concat(df)
    
    # 优化数据类型
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    
    df = df.astype({
        **{col: 'float32' for col in float_cols},
        **{col: 'int32' for col in int_cols}
    })
    
    return df

@st.cache_data
def calculate_stats(df: pd.DataFrame) -> Dict:
    """计算数据统计指标"""
    return {
        'mean': df.mean().to_dict(),
        'median': df.median().to_dict(),
        'std': df.std().to_dict(),
        'min': df.min().to_dict(),
        'max': df.max().to_dict()
    }

def create_time_series(df: pd.DataFrame, 
                      columns: List[str],
                      resample: Optional[str] = '1H') -> go.Figure:
    """
    创建交互式时间序列图表
    
    Args:
        df: 包含时间序列数据的DataFrame
        columns: 要显示的列名列表
        resample: 重采样间隔，如'1H'表示每小时
    """
    # 如果数据点过多，进行重采样
    if resample and 'date' in df.columns:
        df = df.set_index('date')
        df = df.resample(resample).mean()
        df = df.reset_index()
    
    fig = go.Figure()
    
    for col in columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[col],
                name=col,
                mode='lines',
                line=dict(shape='spline', smoothing=0.3)
            )
        )
    
    fig.update_layout(
        title='时间序列分析',
        xaxis_title='日期',
        yaxis_title='数值',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # 使用WebGL渲染提升性能
    fig.update_traces(type='scattergl')
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """创建相关性热力图"""
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='相关性分析',
        width=600,
        height=600
    )
    
    return fig

def create_dashboard(df: pd.DataFrame,
                    date_col: str = 'date',
                    default_columns: Optional[List[str]] = None) -> None:
    """
    创建交互式数据分析仪表盘
    
    Args:
        df: 数据DataFrame
        date_col: 日期列名
        default_columns: 默认显示的列
    """
    st.set_page_config(
        page_title="数据分析仪表盘",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 实时数据分析仪表盘")
    
    # 侧边栏控件
    st.sidebar.header("📈 图表控制")
    
    # 选择要显示的指标
    available_columns = [col for col in df.columns if col != date_col]
    selected_columns = st.sidebar.multiselect(
        "选择要显示的指标",
        options=available_columns,
        default=default_columns or available_columns[:2]
    )
    
    # 日期范围选择
    if date_col in df.columns:
        date_range = st.sidebar.date_input(
            "选择日期范围",
            value=(df[date_col].min(), df[date_col].max())
        )
        
        # 数据过滤
        mask = (df[date_col].dt.date >= date_range[0]) & \
               (df[date_col].dt.date <= date_range[1])
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    # 布局设置
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("时间序列分析")
        start_time = time.time()
        if date_col in df.columns and selected_columns:
            fig1 = create_time_series(filtered_df, selected_columns)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption(f"图表渲染时间: {(time.time() - start_time):.3f} 秒")

    with col2:
        st.subheader("相关性分析")
        if selected_columns:
            start_time = time.time()
            fig2 = create_correlation_heatmap(filtered_df[selected_columns])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"图表渲染时间: {(time.time() - start_time):.3f} 秒")

    # 统计指标
    if selected_columns:
        st.subheader("📊 统计指标")
        stats = calculate_stats(filtered_df[selected_columns])
        
        # 创建多列布局显示统计指标
        stats_cols = st.columns(len(selected_columns))
        for i, col in enumerate(selected_columns):
            with stats_cols[i]:
                st.markdown(f"**{col}**")
                st.metric("平均值", f"{stats['mean'][col]:.2f}")
                st.metric("中位数", f"{stats['median'][col]:.2f}")
                st.metric("标准差", f"{stats['std'][col]:.2f}")

    # 数据表格(分页显示)
    st.subheader("📋 数据明细")
    page_size = st.slider("每页显示行数", 10, 100, 50)
    total_pages = len(filtered_df) // page_size + 1
    page = st.selectbox("选择页码", range(1, total_pages + 1))
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(filtered_df.iloc[start_idx:end_idx])
