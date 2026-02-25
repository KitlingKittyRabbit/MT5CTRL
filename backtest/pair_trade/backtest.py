import os
from datetime import datetime

import dotenv
import mt5ctrl


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"环境变量 {name} 未设置")
    return value


def parse_bool(value: str) -> bool:
    return value.strip().lower() in ["1", "true", "yes", "y", "on"]


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


dotenv.load_dotenv()

# MT5账户配置
account = int(get_env("MT5_ACCOUNT"))
server = get_env("MT5_SERVER")
password = get_env("MT5_PASSWORD")

# 回测参数
category_a = get_env("PAIR_TRADE_CATEGORY_A")
category_b = get_env("PAIR_TRADE_CATEGORY_B")
n_days = int(get_env("PAIR_TRADE_N_DAYS"))

signal_timeframe = get_env("BACKTEST_SIGNAL_TIMEFRAME")
trading_timeframe = get_env("BACKTEST_TRADING_TIMEFRAME")
start_time = parse_dt(get_env("BACKTEST_START_TIME"))
end_time = parse_dt(get_env("BACKTEST_END_TIME"))

adf_threshold = float(get_env("BACKTEST_ADF_THRESHOLD"))
min_entry_zscore = float(get_env("BACKTEST_MIN_ENTRY_ZSCORE"))
take_profit_zscore = float(get_env("BACKTEST_TAKE_PROFIT_ZSCORE"))
max_entry_zscore = float(get_env("BACKTEST_MAX_ENTRY_ZSCORE"))
stop_loss_zscore = float(get_env("BACKTEST_STOP_LOSS_ZSCORE"))
skip_weekend = parse_bool(get_env("BACKTEST_SKIP_WEEKEND"))

principal = float(get_env("BACKTEST_PRINCIPAL"))
total_lot = float(get_env("BACKTEST_TOTAL_LOT"))
resample_rule = get_env("BACKTEST_RESAMPLE_RULE")

# 注意：当前mt5ctrl.backtest内部按 price_a/price_b 解析为 a/b 键
trading_cost = {
    category_a: float(get_env("BACKTEST_TRADING_COST_A")),
    category_b: float(get_env("BACKTEST_TRADING_COST_B")),
}
contract_size = {
    category_a: float(get_env("BACKTEST_CONTRACT_SIZE_A")),
    category_b: float(get_env("BACKTEST_CONTRACT_SIZE_B")),
}

output_dir = os.getenv("BACKTEST_OUTPUT_DIR", "./result")
os.makedirs(output_dir, exist_ok=True)

print("开始生成开平点数据...")
backtester = mt5ctrl.Backtest(
    account=account, password=password, server=server)

bs_point = backtester.genarate_bs_point4pair_trade(
    category_a=category_a,
    category_b=category_b,
    signal_timeframe=signal_timeframe,  # type: ignore[arg-type]
    n_days=n_days,
    start_time=start_time,
    end_time=end_time,
    adf_threshold=adf_threshold,
    min_entry_zscore=min_entry_zscore,
    take_profit_zscore=take_profit_zscore,
    skip_weekend=skip_weekend,
    trading_timeframe=trading_timeframe,  # type: ignore[arg-type]
    max_entry_zscore=max_entry_zscore,
    stop_loss_zscore=stop_loss_zscore,
)

print("开始执行回测...")
result = backtester.backtest(
    bs_point=bs_point,
    principal=principal,
    trading_cost=trading_cost,
    total_lot=total_lot,
    contract_size=contract_size,
    resample_rule=resample_rule,
)

print("回测完成：")
print(f"total_return={result['total_return']:.6f}")
print(f"annualized_return={result['annualized_return']}")
print(f"annualized_volatility={result['annualized_volatility']}")
print(f"max_drawdown={result['max_drawdown']}")
print(f"sharpe_ratio={result['sharpe_ratio']}")
print(f"win_rate={result['win_rate']}")

bs_point.to_csv(os.path.join(output_dir, "bs_point.csv"))
result["equity_curve"].to_csv(os.path.join(output_dir, "equity_curve.csv"))
result["trade_log"].to_csv(os.path.join(output_dir, "trade_log.csv"))

print(f"结果已保存到: {output_dir}")
