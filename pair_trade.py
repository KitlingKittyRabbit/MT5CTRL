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
n_days = os.getenv("PAIR_TRADE_N_DAYS")

signal = mt5ctrl.Signal(account=account, password=password,  # type: ignore
                        server=server)  # type: ignore
print("开始配对交易监控...")

while True:
    pair_trade_signal = signal.pair_trade_signal(
        categary_a, categary_b, 'M1', n_days=n_days)  # type: ignore
    print(pair_trade_signal['adf_result'], pair_trade_signal['zscore'])
    if pair_trade_signal['adf_result']:
        if pair_trade_signal['signal'] == 'long_a,short_b':
            print(f"检测到做多{categary_a}，做空{categary_b}信号，开始下单...")
            order_a = mt5ctrl.Order(categary_a, 'long', lot=round(0.1*pair_trade_signal['beta'], 2),  # type: ignore
                                    account=account, server=server, password=password)  # type: ignore
            order_b = mt5ctrl.Order(
                categary_b, 'short', lot=0.1, account=account, server=server, password=password)  # type: ignore
            while True:
                pair_trade_signal, beta = signal.pair_trade_signal(
                    categary_a, categary_b, 'M1', n_days=n_days)  # type: ignore
                if pair_trade_signal == 'close_positions':
                    # 平仓
                    order_a.close_order()
                    order_b.close_order()
                    break
                time.sleep(6)

        elif pair_trade_signal['signal'] == 'short_a,long_b':
            print(f"检测到做空{categary_a}，做多{categary_b}信号，开始下单...")
            order_a = mt5ctrl.Order(
                categary_a, 'short', lot=round(0.1*pair_trade_signal['beta'], 2), account=account, server=server, password=password)  # type: ignore
            order_b = mt5ctrl.Order(categary_b, 'long', lot=0.1,  # type: ignore
                                    account=account, server=server, password=password)  # type: ignore
            while True:
                pair_trade_signal, beta = signal.pair_trade_signal(
                    categary_a, categary_b, 'M1', n_days=n_days)  # type: ignore
                if pair_trade_signal == 'close_positions':
                    # 平仓
                    order_a.close_order()
                    order_b.close_order()
                    break
                time.sleep(6)
    time.sleep(10)
