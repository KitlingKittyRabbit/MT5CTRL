import mt5ctrl
import time
import os
import dotenv
# 从.env配置文件中导入账户信息、品种信息、监控天数等
dotenv.load_dotenv()

# 执行配对交易，监控交易信号
account = int(os.getenv("MT5_ACCOUNT"))  # type: ignore
server = os.getenv("MT5_SERVER")
password = os.getenv("MT5_PASSWORD")
categary_a = os.getenv("PAIR_TRADE_CATEGARY_A")
categary_b = os.getenv("PAIR_TRADE_CATEGARY_B")
n_days = int(os.getenv("PAIR_TRADE_N_DAYS"))  # type: ignore
lot_b = float(os.getenv("PAIR_TRADE_LOT_B"))  # type: ignore

signal = mt5ctrl.Signal(account=account, password=password,  # type: ignore
                        server=server)  # type: ignore
print("开始配对交易监控...")

while True:
    pair_trade_signal = signal.pair_trade_signal(
        categary_a, categary_b, 'M1', n_days=n_days)  # type: ignore
    print(pair_trade_signal['adf_result'], pair_trade_signal['z_score'],
          pair_trade_signal['market_all_open'])
    if pair_trade_signal['adf_result'] and pair_trade_signal['market_all_open'] and (pair_trade_signal['signal'] == 'long_a,short_b' or pair_trade_signal['signal'] == 'short_a,long_b'):
        print(f"检测到信号:{pair_trade_signal['signal']}，开始下单...")
        a_direction, b_direction = (
            'long', 'short') if pair_trade_signal['signal'] == 'long_a,short_b' else ('short', 'long')
        order_a = mt5ctrl.Order(categary_a, a_direction, lot=round(lot_b*pair_trade_signal['beta'], 2),  # type: ignore
                                account=account, server=server, password=password)  # type: ignore
        order_b = mt5ctrl.Order(
            categary_b, b_direction, lot=lot_b, account=account, server=server, password=password)  # type: ignore

        if order_a.order_number != 10009 or order_b.order_number != 10009:
            print("下单失败，重新监控信号...")
            if order_a.order_number == 10009:
                order_a.close_order()
            if order_b.order_number == 10009:
                order_b.close_order()
            time.sleep(1)
            continue

        while True:
            pair_trade_signal = signal.pair_trade_signal(
                categary_a, categary_b, 'M1', n_days=n_days)  # type: ignore
            if pair_trade_signal['signal'] == 'close_positions' and pair_trade_signal['market_all_open']:
                # 平仓
                order_a.close_order()
                order_b.close_order()
                break
            time.sleep(1)
    time.sleep(1)
