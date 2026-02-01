from __future__ import annotations

import datetime as dt

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from main import (
    Config,
    build_features,
    fetch_ashare_daily,
    train_model,
    train_test_split_time_series,
)


st.set_page_config(page_title="A股价格预测与量化分析 Demo", layout="wide")

st.title("A股价格预测与量化分析 Demo")
st.markdown(
    "基于 `akshare` 日线数据 + 多种技术指标（MACD / RSI / 布林带 / KDJ / ATR），"
    "使用随机森林回归预测未来 1 日收盘价，并以交互式图表展示结果。"
)


@st.cache_data(show_spinner=True)
def load_data(cfg: Config):
    df = fetch_ashare_daily(cfg)
    X, y, df_feat = build_features(cfg, df)
    return df, X, y, df_feat


def make_prediction_fig(cfg: Config, X_test, y_test, model):
    """
    使用 Plotly 绘制：价格预测 + MACD + RSI 三联图。
    """
    y_pred = model.predict(X_test)
    y_pred_series = np.array(y_pred)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
    )

    # Row1: 真实 vs 预测价格
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test.values,
            name="真实收盘价",
            line=dict(color="blue", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_pred_series,
            name="预测收盘价",
            line=dict(color="orange", width=1.2, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Row2: MACD
    if {"macd", "macd_signal", "macd_hist"}.issubset(X_test.columns):
        macd = X_test["macd"]
        macd_signal = X_test["macd_signal"]
        macd_hist = X_test["macd_hist"]

        fig.add_trace(
            go.Bar(
                x=macd_hist.index,
                y=macd_hist.values,
                name="MACD Hist",
                marker_color=np.where(macd_hist.values >= 0, "red", "green"),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=macd.index,
                y=macd.values,
                name="MACD",
                line=dict(color="black", width=1.0),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=macd_signal.index,
                y=macd_signal.values,
                name="Signal",
                line=dict(color="magenta", width=1.0),
            ),
            row=2,
            col=1,
        )

    # Row3: RSI
    if "rsi_14" in X_test.columns:
        rsi = X_test["rsi_14"]
        fig.add_trace(
            go.Scatter(
                x=rsi.index,
                y=rsi.values,
                name="RSI(14)",
                line=dict(color="purple", width=1.0),
            ),
            row=3,
            col=1,
        )
        # 超买 / 超卖线
        fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(
            y=30, line=dict(color="green", width=1, dash="dash"), row=3, col=1
        )

    fig.update_layout(
        template="plotly_white",
        title=f"A股 {cfg.stock_code} 未来{cfg.predict_horizon_days}日收盘价预测与技术指标（测试集）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=700,
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    return fig


def make_candlestick_fig(cfg: Config, df_feat, index):
    """
    使用 Plotly 绘制 K 线 + 均线 + 布林带。
    """
    if df_feat.empty:
        return None

    if len(index) == 0:
        df_plot = df_feat
    else:
        start, end = index.min(), index.max()
        df_plot = df_feat.loc[start:end]

    if df_plot.empty:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="K线",
            increasing_line_color="red",
            decreasing_line_color="green",
        ),
        row=1,
        col=1,
    )

    # 均线
    for col, color, name in [
        ("ma_5", "blue", "MA5"),
        ("ma_20", "orange", "MA20"),
        ("ma_60", "green", "MA60"),
    ]:
        if col in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot[col],
                    name=name,
                    line=dict(color=color, width=1.0),
                ),
                row=1,
                col=1,
            )

    # 布林带
    if {"bb_upper", "bb_lower"}.issubset(df_plot.columns):
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot["bb_upper"],
                name="Bollinger Upper",
                line=dict(color="rgba(200,0,0,0.4)", width=1.0),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot["bb_lower"],
                name="Bollinger Lower",
                line=dict(color="rgba(200,0,0,0.4)", width=1.0),
                fill="tonexty",
                fillcolor="rgba(200,0,0,0.05)",
            ),
            row=1,
            col=1,
        )

    # 成交量
    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot["volume"],
            name="Volume",
            marker_color="rgba(128,128,128,0.6)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        title=f"A股 {cfg.stock_code} 测试区间 K 线 + 均线 + 布林带",
        margin=dict(l=40, r=20, t=60, b=40),
        height=700,
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    return fig


def make_feature_importance_fig(model, feature_names):
    """
    使用 Plotly 绘制特征重要性条形图，只展示 Top-N。
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = np.array(feature_names)[indices]
    sorted_importances = importances[indices]

    top_n = min(12, len(sorted_names))
    sorted_names = sorted_names[:top_n]
    sorted_importances = sorted_importances[:top_n]

    fig = go.Figure(
        data=[
            go.Bar(
                x=sorted_importances[::-1],
                y=sorted_names[::-1],
                orientation="h",
                marker=dict(
                    color=sorted_importances[::-1],
                    colorscale="Blues",
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title="随机森林特征重要性（Top 特征）",
        xaxis_title="重要性",
        margin=dict(l=80, r=20, t=60, b=40),
        height=500,
    )
    return fig


with st.sidebar:
    st.header("参数配置")

    default_start = dt.date(2015, 1, 1)
    default_end = dt.date.today()

    stock_code = st.text_input("股票代码（6 位数字）", value="000001")
    start_date = st.date_input("起始日期", value=default_start)
    end_date = st.date_input("结束日期", value=default_end)

    horizon = st.number_input("预测未来几日收盘价", min_value=1, max_value=10, value=1, step=1)
    test_ratio = st.slider("测试集占比", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    n_estimators = st.slider(
        "随机森林树的数量（n_estimators）", min_value=100, max_value=800, value=300, step=50
    )

    run_button = st.button("运行预测", type="primary")


if run_button:
    cfg = Config(
        stock_code=stock_code.strip(),
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        predict_horizon_days=horizon,
        test_size_ratio=float(test_ratio),
    )

    try:
        with st.spinner("正在拉取数据并构造特征..."):
            df_raw, X, y, df_feat = load_data(cfg)
    except Exception as e:
        st.error(f"数据加载失败：{e}")
        st.stop()

    st.success(
        f"数据加载完成：共 {len(df_raw)} 条原始记录，"
        f"可用样本 {len(X)} 条，特征维度 {X.shape[1]}。"
    )

    with st.expander("查看原始日线数据（前 50 行）", expanded=False):
        st.dataframe(df_raw.head(50))

    # 划分训练 / 测试集
    X_train, X_test, y_train, y_test = train_test_split_time_series(cfg, X, y)

    st.write(
        f"训练集样本数：**{len(X_train)}**，测试集样本数：**{len(X_test)}**。"
    )

    # 训练模型
    with st.spinner("正在训练随机森林模型..."):
        model = train_model(cfg, X_train, y_train, n_estimators=n_estimators)

    # 计算评估指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    tab_pred, tab_kline, tab_fi = st.tabs(
        ["预测表现与技术指标", "K 线与价格结构", "特征重要性"]
    )

    with tab_pred:
        st.subheader("预测表现与关键技术指标")
        cols = st.columns(4)
        cols[0].metric("MSE", f"{metrics['mse']:.4f}")
        cols[1].metric("RMSE", f"{metrics['rmse']:.4f}")
        cols[2].metric("MAE", f"{metrics['mae']:.4f}")
        cols[3].metric("R²", f"{metrics['r2']:.4f}")

        fig_pred = make_prediction_fig(cfg, X_test, y_test, model)
        st.plotly_chart(fig_pred, width="stretch")

    with tab_kline:
        st.subheader("测试区间 K 线 + 均线 + 布林带 + 成交量")
        fig_k = make_candlestick_fig(cfg, df_feat, X_test.index)
        if fig_k is not None:
            st.plotly_chart(fig_k, width="stretch")
        else:
            st.info("暂无可用的 K 线数据。")

    with tab_fi:
        st.subheader("特征重要性（Feature Importance）")
        fig_fi = make_feature_importance_fig(model, X_train.columns)
        if fig_fi is not None:
            st.plotly_chart(fig_fi, width="stretch")
        else:
            st.info("当前模型不支持特征重要性或训练失败。")

