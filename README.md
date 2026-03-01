# MT5CTRL

## 概览
MT5CTRL 是一个基于 Python 的 MetaTrader 5 (MT5) 自动化交易与回测控制库。它封装了 MT5 底层 API，覆盖登录、行情、下单、统计分析，并内置一套可直接跑实盘/回测的配对交易方案（ADF + Huber 回归 Beta + Z-Score）。

**特性**
- MT5 登录与下单封装：最小手数查询、双腿同步下单/平仓。
- 配对交易信号：滚动窗口的 Beta、价差、ADF p-value、Z-Score，含开/平/止损逻辑与周末跳过控制。
- 回测引擎：支持信号/执行不同时间粒度，滚动统计，合约乘数、交易成本、再采样收益等指标。
- 脚本即用：`trade/pair_trade/pair_trade.py` 持续监控；`backtest/pair_trade/backtest.py` 生成 `bs_point.csv`、`trade_log.csv`、`equity_curve.csv`。

## 环境要求与安装
- 已安装并能正常运行的 MetaTrader 5 客户端（需与 Python 同机）。
- Python 3.8+。
- 依赖：`MetaTrader5`, `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `pytz`, `python-dotenv`。

**从 GitHub 安装（推荐）**
```bash
pip install "git+https://github.com/yourname/MT5CTRL.git"
```

**本地开发安装**
```bash
pip install -e .
```

**仅装核心依赖**
```bash
pip install MetaTrader5 pandas numpy statsmodels scikit-learn pytz python-dotenv
```

## 目录结构
- `mt5ctrl.py`：核心库，提供登录、行情、下单、配对交易信号与回测能力。
- `trade/pair_trade/pair_trade.py`：配对交易实盘/模拟盘监控脚本。
- `backtest/pair_trade/backtest.py`：配对交易回测脚本，支持多时间框架与交易成本设置。
- `backtest/pair_trade/optimize.py`：参数搜索入口（需自备 `.env` 配置）。

## 配置 (.env)
每个脚本目录使用独立的 `.env`。布尔值支持 `1/true/yes/on`。

### 通用账户配置
```ini
MT5_ACCOUNT=你的MT5账号
MT5_PASSWORD=你的MT5密码
MT5_SERVER=你的MT5服务器名称
```

### 实盘/模拟盘监控 (`trade/pair_trade/.env` 示例)
```ini
MT5_ACCOUNT=123456
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
PAIR_TRADE_CATEGORY_A=EURUSD
PAIR_TRADE_CATEGORY_B=GBPUSD
PAIR_TRADE_N_DAYS=30
PAIR_TRADE_LOT_B=0.10              # B腿基础手数，A腿按 Beta 自动调整且不低于最小手数
PAIR_TRADE_TIMEFRAME=M5            # M1/M5/M15/M30/H1/H4/D1
PAIR_TRADE_ADF_THRESHOLD=0.05
PAIR_TRADE_MIN_ENTRY_ZSCORE=1.6
PAIR_TRADE_MAX_ENTRY_ZSCORE=2.1
PAIR_TRADE_TAKE_PROFIT_ZSCORE=0.8
PAIR_TRADE_STOP_LOSS_ZSCORE=2.2
PAIR_TRADE_SKIP_WEEKEND=true
```

### 回测 (`backtest/pair_trade/.env` 示例)
```ini
MT5_ACCOUNT=123456
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
PAIR_TRADE_CATEGORY_A=EURUSD
PAIR_TRADE_CATEGORY_B=GBPUSD
PAIR_TRADE_N_DAYS=30

BACKTEST_SIGNAL_TIMEFRAME=M15      # 统计窗口粒度
BACKTEST_TRADING_TIMEFRAME=M1      # 交易执行粒度，可为 S1
BACKTEST_START_TIME=2024-01-01 00:00
BACKTEST_END_TIME=2024-02-01 00:00
BACKTEST_ADF_THRESHOLD=0.05
BACKTEST_MIN_ENTRY_ZSCORE=1.6
BACKTEST_MAX_ENTRY_ZSCORE=2.1
BACKTEST_TAKE_PROFIT_ZSCORE=0.8
BACKTEST_STOP_LOSS_ZSCORE=2.2
BACKTEST_SKIP_WEEKEND=true

BACKTEST_PRINCIPAL=10000
BACKTEST_TOTAL_LOT=0.20
BACKTEST_RESAMPLE_RULE=D           # 权益曲线再采样频率，见 pandas resample 规则
BACKTEST_TRADING_COST_A=5          # 每手交易成本（含点差+手续费）
BACKTEST_TRADING_COST_B=5
BACKTEST_CONTRACT_SIZE_A=100000
BACKTEST_CONTRACT_SIZE_B=100000
BACKTEST_OUTPUT_DIR=./result
```

## 快速开始
1) 安装依赖并准备 `.env`（按上方示例）。

**回测**
```bash
python backtest/pair_trade/backtest.py
```
生成文件：`bs_point.csv`（买卖点）、`trade_log.csv`（成交日志）、`equity_curve.csv`（权益曲线）。

**实盘/模拟盘监控**
```bash
python trade/pair_trade/pair_trade.py
```
脚本会持续轮询行情，满足 ADF 与 z-score 条件即下单，达到止盈/止损则平仓。

## 参数提示
- `PAIR_TRADE_MIN_ENTRY_ZSCORE` 必须大于 1；止盈阈值通常小于进场阈值以避免频繁反向。
- `PAIR_TRADE_SKIP_WEEKEND=true` 适合外汇等休市品种，7x24 品种请设为 false。
- 回测时 `BACKTEST_TOTAL_LOT` 结合合约乘数决定名义敞口，注意保证金约束。

## 许可与贡献
本项目为个人量化交易工具库，可按需修改与扩展，欢迎新增策略模块或优化 MT5 交互逻辑。