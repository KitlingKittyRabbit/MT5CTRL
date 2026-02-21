import mt5ctrl
import time
import os
import dotenv


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"环境变量 {name} 未设置")
    return value


def parse_bool(value: str) -> bool:
    return value.strip().lower() in ["1", "true", "yes", "y", "on"]


# 从.env配置文件中导入账户信息、品种信息、监控天数等
dotenv.load_dotenv()

# 执行配对交易，监控交易信号
account = int(get_env("MT5_ACCOUNT"))
server = get_env("MT5_SERVER")
password = get_env("MT5_PASSWORD")
category_a = get_env("PAIR_TRADE_CATEGORY_A")
category_b = get_env("PAIR_TRADE_CATEGORY_B")
n_days = int(get_env("PAIR_TRADE_N_DAYS"))
lot_b = float(get_env("PAIR_TRADE_LOT_B"))

timeframe = get_env("PAIR_TRADE_TIMEFRAME")
adf_threshold = float(get_env("PAIR_TRADE_ADF_THRESHOLD"))
entry_zscore = float(get_env("PAIR_TRADE_ENTRY_ZSCORE"))
exit_zscore = float(get_env("PAIR_TRADE_EXIT_ZSCORE"))
skip_weekend = parse_bool(get_env("PAIR_TRADE_SKIP_WEEKEND"))

signal = mt5ctrl.Signal(account=account, password=password,  # type: ignore
                        server=server)  # type: ignore

# 获取品种的最小手数
min_lot_a = signal.get_min_lot(category_a)
min_lot_b = signal.get_min_lot(category_b)
print(f"品种 {category_a} 最小手数: {min_lot_a}")
print(f"品种 {category_b} 最小手数: {min_lot_b}")
print("开始配对交易监控...")

while True:
    pair_trade_signal = signal.pair_trade_signal(
        category_a,
        category_b,
        timeframe,  # type: ignore[arg-type]
        n_days=n_days,
        adf_threshold=adf_threshold,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        skip_weekend=skip_weekend,
    )
    print(pair_trade_signal['adf_result'], pair_trade_signal['z_score'],
          pair_trade_signal['market_all_open'])
    if pair_trade_signal['adf_result'] and pair_trade_signal['market_all_open'] and (pair_trade_signal['signal'] == 'long_a,short_b' or pair_trade_signal['signal'] == 'short_a,long_b'):
        print(f"检测到信号:{pair_trade_signal['signal']}，开始下单...")
        a_direction, b_direction = (
            'long', 'short') if pair_trade_signal['signal'] == 'long_a,short_b' else ('short', 'long')
        lot_a_calc = round(lot_b * pair_trade_signal['beta'], 2)
        lot_a = max(min_lot_a, lot_a_calc)  # 确保手数不低于最小值
        if lot_a != lot_a_calc:
            print(f"警告：计算手数 {lot_a_calc} 低于最小值 {min_lot_a}，已调整为 {lot_a}")
        order_a = mt5ctrl.Order(category_a, a_direction, lot=lot_a,  # type: ignore
                                account=account, server=server, password=password)  # type: ignore
        order_b = mt5ctrl.Order(
            category_b, b_direction, lot=lot_b, account=account, server=server, password=password)  # type: ignore

        if order_a.retcode != 10009 or order_b.retcode != 10009:
            print("下单失败，重新监控信号...")
            if order_a.retcode == 10009:
                order_a.close_order()
            if order_b.retcode == 10009:
                order_b.close_order()
            time.sleep(1)
            continue

        while True:
            pair_trade_signal = signal.pair_trade_signal(
                category_a,
                category_b,
                timeframe,  # type: ignore[arg-type]
                n_days=n_days,
                adf_threshold=adf_threshold,
                entry_zscore=entry_zscore,
                exit_zscore=exit_zscore,
                skip_weekend=skip_weekend,
            )
            if pair_trade_signal['signal'] == 'take_profit' and pair_trade_signal['market_all_open']:
                # 平仓
                order_a.close_order()
                order_b.close_order()
                break
            elif pair_trade_signal['signal'] == 'stop_loss' and pair_trade_signal['market_all_open']:
                # 平仓
                order_a.close_order()
                order_b.close_order()
                time.sleep(60)  # 避免频繁交易，停60秒后继续监控信号
                break
            time.sleep(1)
    time.sleep(1)
