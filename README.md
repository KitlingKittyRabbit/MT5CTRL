# MT5CTRL

## 概览
MT5CTRL 是一个基于 Python 的 MetaTrader 5 (MT5) 自动化交易与回测控制库。它封装了 MT5 的底层 API，提供了便捷的登录、行情获取、订单管理以及统计分析功能，旨在帮助开发者快速构建和验证量化交易策略。

**目前已实现的策略模块：**
- **配对交易 (Pair Trading)**：基于统计套利（协整性检验 ADF + 稳健回归 Beta + Z-Score）的配对交易策略，包含完整的实盘/模拟盘监控脚本与历史回测框架。
*(未来将持续扩展更多类型的交易策略与分析工具)*

## 环境要求与安装
- 已安装并能正常运行的 MetaTrader 5 客户端（需与 Python 运行在同一台机器上）。
- Python 3.9+
- 核心依赖：`MetaTrader5`, `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `pytz`, `python-dotenv`

**安装依赖：**
```bash
pip install MetaTrader5 pandas numpy statsmodels scikit-learn pytz python-dotenv
```

## 目录结构与文件说明
- `mt5ctrl.py`：核心控制库。
  - 封装了 MT5 的初始化与账户登录。
  - 提供了行情数据的获取与处理。
  - 封装了开平仓等订单操作。
  - 包含用于配对交易的统计指标计算（如 ADF 检验、Huber 稳健回归计算 Beta 等）。
- `trade/`：实盘/模拟盘交易脚本目录。
  - `pair_trade/pair_trade.py`：配对交易实盘监控脚本，根据实时 Z-Score 和 ADF 检验结果自动执行开平仓。
- `backtest/`：历史数据回测脚本目录。
  - `pair_trade/backtest.py`：配对交易回测框架，支持多时间框架混合（如 M15 计算信号，M1 执行交易），并生成详细的交易日志、买卖点记录和权益曲线。

## 配置说明 (.env)
项目使用 `.env` 文件进行参数配置。在 `trade/pair_trade/` 和 `backtest/pair_trade/` 目录下需要分别配置对应的 `.env` 文件。

### 基础账户配置 (通用)
```ini
MT5_ACCOUNT=你的MT5账号
MT5_PASSWORD=你的MT5密码
MT5_SERVER=你的MT5服务器名称
```

### 配对交易实盘配置 (`trade/pair_trade/.env`)
- `PAIR_TRADE_CATEGORY_A` / `PAIR_TRADE_CATEGORY_B`：配对交易的两个品种（如 EURUSD 和 GBPUSD）。
- `PAIR_TRADE_N_DAYS`：统计窗口天数。
- `PAIR_TRADE_LOT_B`：品种 B 的基础下单手数（品种 A 的手数将根据 Beta 值自动计算，并受限于最小手数限制）。
- `PAIR_TRADE_TIMEFRAME`：K线时间框架（支持 M1, M5, M15, M30, H1, H4, D1 等）。
- `PAIR_TRADE_ADF_THRESHOLD`：ADF 检验的 p-value 阈值（通常设为 0.05，小于该值视为序列平稳）。
- `PAIR_TRADE_MIN_ENTRY_ZSCORE` / `PAIR_TRADE_MAX_ENTRY_ZSCORE`：开仓的 Z-Score 触发区间（例如 1.6 到 2.1）。
- `PAIR_TRADE_TAKE_PROFIT_ZSCORE`：止盈的 Z-Score 阈值（例如 0.8）。
- `PAIR_TRADE_STOP_LOSS_ZSCORE`：止损的 Z-Score 阈值（例如 2.2）。
- `PAIR_TRADE_SKIP_WEEKEND`：是否跳过周末数据（外汇等传统金融市场设为 true，加密货币设为 false）。

### 配对交易回测配置 (`backtest/pair_trade/.env`)
- `BACKTEST_START_TIME` / `BACKTEST_END_TIME`：回测时间区间（UTC 时间，格式 `YYYY-MM-DD HH:MM`）。
- `BACKTEST_SIGNAL_TIMEFRAME`：计算统计信号的时间框架。
- `BACKTEST_TRADING_TIMEFRAME`：模拟交易执行的时间框架（可比信号框架更精细，以提高回测精度）。
- `BACKTEST_PRINCIPAL`：初始回测资金。
- `BACKTEST_TOTAL_LOT`：每笔交易的总手数（将按比例分配给 A 和 B 两个品种）。
- `BACKTEST_TRADING_COST_A` / `BACKTEST_TRADING_COST_B`：单手交易成本（点差+手续费）。
- `BACKTEST_CONTRACT_SIZE_A` / `BACKTEST_CONTRACT_SIZE_B`：合约大小。
- `BACKTEST_OUTPUT_DIR`：回测结果输出目录（默认输出买卖点、交易日志和权益曲线 CSV 文件）。
- 参数优化（`backtest/pair_trade/optimize.py` 使用）：
  - `BACKTEST_OPT_START_TIME` / `BACKTEST_OPT_END_TIME`：优化回测时间区间。
  - `BACKTEST_OPT_N_DAYS_LIST`：统计窗口候选列表（逗号分隔）。
  - `BACKTEST_OPT_MIN_ENTRY_ZSCORE_LIST` / `BACKTEST_OPT_MAX_ENTRY_ZSCORE_LIST`：进场 z-score 候选区间列表。
  - `BACKTEST_OPT_TAKE_PROFIT_ZSCORE_LIST` / `BACKTEST_OPT_STOP_LOSS_ZSCORE_LIST`：止盈/止损 z-score 候选列表。

## 快速开始

### 1. 运行配对交易回测
1. 在 `backtest/pair_trade/` 目录下创建并配置 `.env` 文件。
2. 确保 MT5 客户端已启动并登录。
3. 运行回测脚本：
   ```bash
   python backtest/pair_trade/backtest.py
   ```
4. 查看 `BACKTEST_OUTPUT_DIR` 目录下的回测报告（包含 `bs_point.csv`, `trade_log.csv`, `equity_curve.csv`）。

### 2. 运行实盘/模拟盘监控
1. 在 `trade/pair_trade/` 目录下创建并配置 `.env` 文件。
2. 确保 MT5 客户端已启动并登录，且开启了“允许算法交易”。
3. 运行监控脚本：
   ```bash
   python trade/pair_trade/pair_trade.py
   ```
4. 脚本将持续拉取最新行情，计算统计指标，并在满足条件时自动下单或平仓。

## 核心参数提示
- 进场阈值需大于 1 (`min_entry_zscore > 1`)；`take_profit_zscore` 通常更小以避免频繁反转。
- `skip_weekend=true` 适合外汇等休市品种，7x24 品种请设为 false 以包含周末数据。
- `BACKTEST_TOTAL_LOT` 与合约乘数一起决定名义敞口，实盘需结合保证金限制。

## 许可与贡献
本项目为个人量化交易工具库，可根据实际需求自由修改和扩展。欢迎添加新的交易策略模块或优化底层 MT5 交互逻辑。