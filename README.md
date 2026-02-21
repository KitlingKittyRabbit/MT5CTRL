MT5CTRL 配对交易工具
===================

概览
----
- 封装 MetaTrader5 登录、下单、行情获取与统计分析的库，核心实现见 [mt5ctrl.py](mt5ctrl.py)。
- 提供配对交易信号监控示例 [trade/pair_trade/pair_trade.py](trade/pair_trade/pair_trade.py)，按 z-score + ADF 逻辑自动下单与平仓。
- 提供配对交易回测脚本 [backtest/pair_trade/backtest.py](backtest/pair_trade/backtest.py)，可生成买卖点、交易日志、权益曲线。

环境要求与安装
--------------
- 已安装并能正常登录的 MetaTrader 5 终端（与 Python 运行在同一机器）。
- Python 3.9+，依赖：MetaTrader5、pandas、numpy、statsmodels、scikit-learn、pytz、python-dotenv。
- 安装示例：`pip install MetaTrader5 pandas numpy statsmodels scikit-learn pytz python-dotenv`。

文件说明
--------
- [mt5ctrl.py](mt5ctrl.py)：
  - `MetaTrader5Control` 负责初始化与登录。
  - `Order` 封装开平仓。
  - `Signal` 计算配对交易信号（ADF + 回归 beta + z-score）。
  - `Backtest` 生成买卖点并评估收益、波动、回撤、夏普、胜率。
- [trade/pair_trade/pair_trade.py](trade/pair_trade/pair_trade.py)：实时监控配对信号并自动下单，适合模拟或实盘。
- [backtest/pair_trade/backtest.py](backtest/pair_trade/backtest.py)：使用历史数据回测，输出 CSV 报表。

.env 配置
---------
示例文件已放在对应目录，可复制后按需修改：

实盘/模拟监控（trade/pair_trade/.env）
- `MT5_ACCOUNT` / `MT5_SERVER` / `MT5_PASSWORD`：MT5 账号、服务器、密码。
- `PAIR_TRADE_CATEGORY_A` / `PAIR_TRADE_CATEGORY_B`：成对品种，如 EURUSD / GBPUSD。
- `PAIR_TRADE_N_DAYS`：统计窗口（工作日天数，周末可选择跳过）。
- `PAIR_TRADE_LOT_B`：B 腿下单手数，A 腿按 beta 自动匹配并不低于最小手数。
- `PAIR_TRADE_TIMEFRAME`：信号时间框架，支持 M1/M5/M15/M30/H1/H4/D1/S1。
- `PAIR_TRADE_ADF_THRESHOLD`：ADF p 值阈值，<= 该值视为平稳。
- `PAIR_TRADE_ENTRY_ZSCORE` / `PAIR_TRADE_EXIT_ZSCORE`：进出场 z-score。
- `PAIR_TRADE_SKIP_WEEKEND`：是否跳过周末（加密等 7x24 可设为 false）。

回测（backtest/pair_trade/.env）
- 账户信息同上，用于连接 MT5 取历史数据。
- `BACKTEST_START_TIME` / `BACKTEST_END_TIME`：UTC 时间区间，格式 `YYYY-MM-DD HH:MM`。
- `BACKTEST_SIGNAL_TIMEFRAME` / `BACKTEST_TRADING_TIMEFRAME`：信号统计粒度与交易执行粒度，可分离（如统计用 M1，执行用 S1）。
- `BACKTEST_ADF_THRESHOLD` / `BACKTEST_ENTRY_ZSCORE` / `BACKTEST_EXIT_ZSCORE` / `BACKTEST_SKIP_WEEKEND`：信号参数同上。
- `BACKTEST_PRINCIPAL`：初始资金。
- `BACKTEST_TOTAL_LOT`：每笔交易总手数，按比率分配到各腿并不低于最小手数。
- `BACKTEST_TRADING_COST_A/B`：每手交易成本。
- `BACKTEST_CONTRACT_SIZE_A/B`：合约大小。
- `BACKTEST_RESAMPLE_RULE`：权益曲线重采样频率（用于波动与夏普），如 D/H/15T。
- `BACKTEST_OUTPUT_DIR`：回测输出目录，默认 `./result`。

快速开始
--------
1) 实盘/模拟配对监控
- 配置 [trade/pair_trade/.env](trade/pair_trade/.env)。
- 运行：`python trade/pair_trade/pair_trade.py`。
- 脚本会：
  - 读取 MT5 账号登录，获取 A/B 最小手数；
  - 实时计算 ADF + z-score；
  - 信号满足且市场开盘时自动开仓，落地 retcode=10009 视为成功；
  - 命中止盈/止损信号后平仓并继续监控。

2) 配对交易回测
- 配置 [backtest/pair_trade/.env](backtest/pair_trade/.env)。
- 运行：`python backtest/pair_trade/backtest.py`。
- 脚本会：
  - 从 MT5 取历史价格，先按 `BACKTEST_SIGNAL_TIMEFRAME` 预计算 rolling beta/均值/标准差/ADF；
  - 按 `BACKTEST_TRADING_TIMEFRAME` 生成买卖点并配比仓位；
  - 计算盈亏、成本、权益曲线及指标；
  - 输出 CSV 到 `BACKTEST_OUTPUT_DIR`：
    - `bs_point.csv`：逐点信号、买卖点、z-score、仓位比例。
    - `equity_curve.csv`：权益、峰值、回撤序列。
    - `trade_log.csv`：逐笔开平及净利润、成本、实际手数。

核心参数提示
------------
- 进场阈值需大于 1 (`entry_zscore > 1`)；`exit_zscore` 通常更小以避免频繁反转。
- `skip_weekend=true` 适合外汇等休市品种，7x24 品种请设为 false 以包含周末数据。
- `BACKTEST_TOTAL_LOT` 与合约乘数一起决定名义敞口，实盘需结合保证金限制。

常见问题
--------
- 无法连接 MT5：确认已安装 64 位 MT5 终端且已登录相同账号，Python 与终端位数一致。
- retcode 非 10009：表示下单未成交，脚本会自动平仓已成交腿并继续监控；可检查手数、可交易性或交易时间。
- CSV 时间戳异常为 1970：回测脚本已在写出前调用 UTC 时间索引，若自定义处理请确保索引为 datetime 类型。

许可与贡献
----------
- 自用脚本，可按需修改。欢迎根据实际需求调整信号逻辑或回测成本模型。