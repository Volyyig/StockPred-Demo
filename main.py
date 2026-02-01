from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import akshare as ak
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class Config:
    """
    全局配置：可以根据需要自行修改。
    """

    # 股票代码（A 股），例如：000001（平安银行）、600000（浦发银行）
    stock_code: str = "000001"

    # 起止日期（字符串格式：YYYYMMDD），注意要在交易所上市之后
    start_date: str = "20150101"
    end_date: str = "20251231"

    # 预测未来多少天后的收盘价，这里默认预测下一交易日
    predict_horizon_days: int = 1

    # 测试集占比（按时间顺序切分）
    test_size_ratio: float = 0.2

    # 随机种子，保证结果可复现
    random_state: int = 42


def fetch_ashare_daily(cfg: Config) -> pd.DataFrame:
    """
    使用 akshare 拉取 A 股日线数据（前复权）。

    返回包含日期索引和常用字段（open/close/high/low/volume）的 DataFrame。
    """
    df = ak.stock_zh_a_hist(
        symbol=cfg.stock_code,
        period="daily",
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        adjust="qfq",  # 前复权
    )

    if df is None or df.empty:
        raise ValueError(f"没有获取到 {cfg.stock_code} 在 {cfg.start_date}-{cfg.end_date} 的数据")

    # 兼容不同版本 akshare 的字段名（常见为中文列名）
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["date", "open", "close", "high", "low", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少必要字段: {missing}")

    # 转换日期并设置为索引
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # 转换数值类型（部分字段可能是字符串）
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    return df


def build_features(cfg: Config, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    基于历史价格构造特征，并生成预测目标（未来 N 日收盘价）。
    """
    df = df.copy()

    # 日收益率
    df["return"] = df["close"].pct_change()

    # 滞后收盘价特征
    for lag in (1, 3, 5):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    # 移动平均和波动率
    for window in (5, 10, 20, 60):
        df[f"ma_{window}"] = df["close"].rolling(window).mean()
        df[f"volatility_{window}"] = df["return"].rolling(window).std()

    # ==== MACD 指标 ====
    ema_short = df["close"].ewm(span=12, adjust=False).mean()
    ema_long = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_short - ema_long
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ==== RSI 指标（14 日） ====
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ==== 布林带（Bollinger Bands, 20 日） ====
    bb_window = 20
    bb_ma = df["close"].rolling(bb_window).mean()
    bb_std = df["close"].rolling(bb_window).std()
    upper_band = bb_ma + 2 * bb_std
    lower_band = bb_ma - 2 * bb_std
    df["bb_upper"] = upper_band
    df["bb_lower"] = lower_band
    # 布林带宽度 & %b
    band_width = (upper_band - lower_band) / (bb_ma + 1e-9)
    df["bb_width"] = band_width
    df["bb_percent_b"] = (df["close"] - lower_band) / ((upper_band - lower_band) + 1e-9)

    # ==== KDJ 指标（9, 3, 3） ====
    low_list = df["low"].rolling(9, min_periods=9).min()
    high_list = df["high"].rolling(9, min_periods=9).max()
    rsv = (df["close"] - low_list) / ((high_list - low_list) + 1e-9) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()  # 相当于 3 日平滑
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # ==== ATR 波动率（14 日） ====
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # 预测目标：未来 N 日收盘价
    horizon = cfg.predict_horizon_days
    df["target"] = df["close"].shift(-horizon)

    # 丢弃包含 NaN 的行（由于 rolling 和 shift 产生）
    df = df.dropna()

    feature_cols = [
        # 价格 & 成交量基础特征
        "close_lag_1",
        "close_lag_3",
        "close_lag_5",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_60",
        "volatility_5",
        "volatility_10",
        "volatility_20",
        "volatility_60",
        "volume",
        # MACD 系列
        "macd",
        "macd_signal",
        "macd_hist",
        # RSI
        "rsi_14",
        # 布林带
        "bb_width",
        "bb_percent_b",
        # KDJ
        "kdj_k",
        "kdj_d",
        "kdj_j",
        # ATR
        "atr_14",
    ]

    X = df[feature_cols]
    y = df["target"]

    # 返回对齐后的特征 DataFrame，方便后续画技术指标图
    return X, y, df


def train_test_split_time_series(
    cfg: Config, X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    按时间顺序切分训练集和测试集（不打乱）。
    """
    n_samples = len(X)
    if n_samples < 50:
        warnings.warn("样本数量较少，模型效果可能不稳定。")

    split_idx = int(n_samples * (1 - cfg.test_size_ratio))
    split_idx = max(1, min(split_idx, n_samples - 1))

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_model(
    cfg: Config,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int | None = None,
) -> RandomForestRegressor:
    """
    使用随机森林回归模型进行训练。

    参数 n_estimators 允许在外部自定义树的数量，默认使用 300。
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators or 300,
        max_depth=None,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_and_plot(
    cfg: Config,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: RandomForestRegressor,
) -> tuple[plt.Figure, dict]:
    """
    对模型进行评估，并画出真实值与预测值的对比图。

    返回 matplotlib Figure 和评估指标字典，方便在 CLI 或 Web 中复用。
    """
    y_pred = model.predict(X_test)
    y_pred_series = pd.Series(y_pred, index=y_test.index, name="pred")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": r2,
    }

    print("=== 回归指标 ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    # 绘制多子图：价格 + MACD + RSI
    fig, (ax_price, ax_macd, ax_rsi) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1]}
    )

    # 价格 & 预测
    ax_price.plot(
        y_test.index, y_test.values, label="真实收盘价", color="blue", linewidth=1.2
    )
    ax_price.plot(
        y_pred_series.index,
        y_pred_series.values,
        label="预测收盘价",
        color="orange",
        alpha=0.8,
        linewidth=1.0,
    )
    ax_price.set_ylabel("价格", fontproperties="SimHei")
    ax_price.legend(prop={"family": "SimHei"})
    ax_price.grid(True, linestyle="--", alpha=0.5)

    # MACD
    if {"macd", "macd_signal", "macd_hist"}.issubset(X_test.columns):
        macd = X_test["macd"]
        macd_signal = X_test["macd_signal"]
        macd_hist = X_test["macd_hist"]
        # 柱状图
        ax_macd.bar(
            macd_hist.index,
            macd_hist.values,
            label="MACD Hist",
            color=np.where(macd_hist >= 0, "red", "green"),
            alpha=0.6,
        )
        # 线
        ax_macd.plot(macd.index, macd.values, label="MACD", color="black", linewidth=0.8)
        ax_macd.plot(
            macd_signal.index,
            macd_signal.values,
            label="Signal",
            color="magenta",
            linewidth=0.8,
        )
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left", fontsize=8)
        ax_macd.grid(True, linestyle="--", alpha=0.5)

    # RSI
    if "rsi_14" in X_test.columns:
        rsi = X_test["rsi_14"]
        ax_rsi.plot(rsi.index, rsi.values, label="RSI(14)", color="purple", linewidth=0.8)
        ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
        ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_xlabel("日期", fontproperties="SimHei")
        ax_rsi.legend(loc="upper left", fontsize=8)
        ax_rsi.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"A股 {cfg.stock_code} 未来{cfg.predict_horizon_days}日收盘价预测 + 技术指标（测试集）",
        fontproperties="SimHei",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig, metrics


def plot_candlestick_with_indicators(
    cfg: Config, df_feat: pd.DataFrame, test_index: pd.DatetimeIndex
) -> plt.Figure | None:
    """
    使用 mplfinance 绘制带有均线和布林带的 K 线图（聚焦测试集时间段）。
    """
    if df_feat.empty:
        return None

    # 只取测试集时间段的数据以便观察模型评估区间
    df_plot = df_feat.loc[df_feat.index.min() : df_feat.index.max()]
    if not test_index.empty:
        start, end = test_index.min(), test_index.max()
        df_plot = df_feat.loc[start:end]

    # 转换为 mplfinance 需要的列名
    mpf_df = df_plot[["open", "high", "low", "close", "volume"]].copy()
    mpf_df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # 添加均线和布林带
    add_plots = []
    if {"ma_5", "ma_20", "ma_60"}.issubset(df_plot.columns):
        add_plots.extend(
            [
                mpf.make_addplot(df_plot["ma_5"], color="blue", width=0.8),
                mpf.make_addplot(df_plot["ma_20"], color="orange", width=0.8),
                mpf.make_addplot(df_plot["ma_60"], color="green", width=0.8),
            ]
        )
    if {"bb_upper", "bb_lower"}.issubset(df_plot.columns):
        add_plots.extend(
            [
                mpf.make_addplot(df_plot["bb_upper"], color="red", width=0.8),
                mpf.make_addplot(df_plot["bb_lower"], color="red", width=0.8),
            ]
        )

    fig, _axes = mpf.plot(
        mpf_df,
        type="candle",
        volume=True,
        addplot=add_plots if add_plots else None,
        style="yahoo",
        title=f"A股 {cfg.stock_code} K线 + 均线 + 布林带（测试集区间）",
        ylabel="价格",
        ylabel_lower="成交量",
        tight_layout=True,
        mav=(),
        returnfig=True,
    )
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model: RandomForestRegressor, feature_names: pd.Index
) -> plt.Figure | None:
    """
    绘制随机森林的特征重要性条形图（按重要性排序）。
    """
    if not hasattr(model, "feature_importances_"):
        print("当前模型不支持特征重要性属性。")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_names = feature_names[indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importances[::-1], align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names[::-1], fontsize=8)
    ax.set_xlabel("重要性", fontproperties="SimHei")
    ax.set_title("随机森林特征重要性", fontproperties="SimHei")
    fig.tight_layout()
    return fig


def main() -> None:
    """
    主入口：从拉取数据到训练评估的一站式流程。
    """
    cfg = Config()

    print(f"正在拉取 A 股 {cfg.stock_code} 的历史日线数据...")
    df = fetch_ashare_daily(cfg)
    print(f"获取到 {len(df)} 条记录，日期范围：{df.index.min().date()} - {df.index.max().date()}")

    print("正在构造特征和预测目标（含多种技术指标）...")
    X, y, df_feat = build_features(cfg, df)
    print(f"特征样本数：{len(X)}，特征维度：{X.shape[1]}")

    print("按时间切分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split_time_series(cfg, X, y)
    print(f"训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")

    print("开始训练随机森林模型（集成多种技术指标特征）...")
    model = train_model(cfg, X_train, y_train)

    print("在测试集上评估模型表现并绘制技术指标图...")
    fig_eval, _metrics = evaluate_and_plot(cfg, X_test, y_test, model)

    print("绘制测试区间的 K 线 + 均线 + 布林带图表...")
    fig_candle = plot_candlestick_with_indicators(cfg, df_feat, X_test.index)

    print("绘制特征重要性（Feature Importance）图表...")
    fig_fi = plot_feature_importance(model, X_train.columns)


if __name__ == "__main__":
    main()
