import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression  # type: ignore
from typing import Any, Union
from typing import List
from typing import Tuple
from typing import Literal
from typing import Dict


class MetaTrader5Control:
    '''
    MetaTrader5 控制类，封装了 MetaTrader5 的初始化和登录功能

    '''

    def __init__(self, account: int, password: str, server: str) -> None:
        '''
        初始化 MetaTrader5 控制类

        arguments:
            account: 交易账户
            password: 交易密码
            server: 交易服务器

        return:
            None
        '''
        self.account = account
        self.password = password
        self.server = server

        self.login()

    def login(self) -> bool:
        '''
        登录 MetaTrader5 交易账户

        return:
            result: 登录结果，True 表示登录成功，False 表示登录失败
        '''

        result = mt5.initialize(login=self.account,  # type: ignore
                                server=self.server,
                                password=self.password)

        return result


class Order(MetaTrader5Control):
    '''
    下单类，封装了 MetaTrader5 的下单功能

    '''

    def __init__(self, categary: str, direction: Literal['long', 'short'], lot: float, account: int, password: str, server: str, stop_loss: Union[float, None] = None, take_profit: Union[float, None] = None) -> None:
        '''
        初始化下单类

        arguments:
            categary: 交易品种，例如 'EURUSD'
            direction: 交易方向，'long' 或 'short'
            lot: 交易手数
            stop_loss: 止损价格
            take_profit: 止盈价格
            account: 交易账户
            password: 交易密码
            server: 交易服务器

        return:
            None
        '''
        super().__init__(account, password, server)
        self.categary = categary
        self.direction = direction
        self.lot = lot
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        if self.login():
            mt5.symbol_select(self.categary, True)  # type: ignore
            request = {
                "action": mt5.TRADE_ACTION_DEAL,  # 无关紧要
                "symbol": self.categary,  # 交易品种
                "volume": self.lot,  # 交易量
                "type": mt5.ORDER_TYPE_SELL if self.direction == 'short' else mt5.ORDER_TYPE_BUY,  # 卖单或买单
                # 开单价格
                "price": mt5.symbol_info_tick(self.categary).bid if self.direction == 'short' else mt5.symbol_info_tick(  # type: ignore
                    self.categary).ask,
            }
            if self.stop_loss is not None:
                request["sl"] = self.stop_loss  # 止损
            if self.take_profit is not None:
                request["tp"] = self.take_profit  # 止盈
            result = mt5.order_send(request)  # type: ignore
            print(result)
            self.order_number = result.order
        else:
            print("登录失败")

    def close_order(self):
        '''
        平仓函数

        return:
            result: 平仓结果
        '''
        mt5.symbol_select(self.categary, True)  # type: ignore
        request = {
            "action": mt5.TRADE_ACTION_DEAL,  # 无关紧要
            "symbol": self.categary,  # 交易品种
            "volume": self.lot,  # 交易量
            "type": mt5.ORDER_TYPE_BUY if self.direction == 'long' else mt5.ORDER_TYPE_SELL,  # 买单或卖单
            "position": self.order_number,
        }

        # 发送交易请求
        result = mt5.order_send(request)  # type: ignore

        return result


class Signal(MetaTrader5Control):
    '''
    信号类，封装了各种信号的生成函数
    '''

    def __init__(self, account: int, password: str, server: str) -> None:
        '''
        初始化信号类

        arguments:
            account: 交易账户
            password: 交易密码
            server: 交易服务器

        return:
            None
        '''
        super().__init__(account, password, server)

    def _time_convert(self, time_tuple: Tuple) -> datetime:
        '''
        时间转换函数，将时间元组转换为 datetime 对象

        arguments:
            time_tuple: 时间元组，格式为 (年,月,日,时,分)

        return:
            dt: datetime 对象
        '''
        year, month, day, hour, minute = time_tuple
        timezone = pytz.timezone("Etc/UTC")
        dt = datetime(year, month, day, hour, minute, tzinfo=timezone)
        return dt

    def _get_start_time_without_weekend(self, n_days: int) -> datetime:
        '''
        获取据现在n天的起始时间，不包含周末

        arguments:
            n_days: 距离现在的天数

        return:
            start_time: 起始时间
        '''
        now = datetime.now(pytz.timezone("Etc/UTC"))
        start_time = now - timedelta(days=n_days)
        # 如果起始时间是周末，则调整到最近的工作日
        while start_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            start_time -= timedelta(days=1)
        return start_time

    def _get_price_series(self, categary: str, time_frame: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'], start: Union[Tuple, datetime], end: Union[Tuple, datetime]) -> pd.Series:
        '''
        获取交易品种的价格序列

        arguments:
            categary: 交易品种
            start: 起始时间，可以是时间元组或 datetime 对象
            end: 结束时间，可以是时间元组或 datetime 对象
            time_frame: 时间周期，'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'

        return:
            price_series: 价格序列,pd.Series形式返回
        '''
        if type(start) == tuple:
            start_dt = self._time_convert(start)
        if type(end) == tuple:
            end_dt = self._time_convert(end)

        if type(start) == datetime:
            timezone = pytz.timezone("Etc/UTC")
            start_dt = start.replace(tzinfo=timezone)
        if type(end) == datetime:
            timezone = pytz.timezone("Etc/UTC")
            end_dt = end.replace(tzinfo=timezone)

        timeframe_dict = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                          'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
        timeframe = timeframe_dict[time_frame]

        rates = mt5.copy_rates_range(  # type: ignore
            categary, timeframe, start_dt, end_dt)
        df = pd.DataFrame(rates)
        price_series = df['close']

        return price_series

    def _adf_test(self, series: List[float]) -> float:
        '''
        ADF检验函数,计算序列的ADF检验p值

        arguments:
            series: 价格序列

        return:
            pvalue: ADF检验的p值
        '''

        result = adfuller(series)
        pvalue = float(result[1])

        return pvalue

    def _regression_coeff(self, series_a: List[float], series_b: List[float]) -> float:
        '''
        回归系数计算函数,计算品种A对品种B的回归系数

        arguments:
            series_a: 品种A的价格序列
            series_b: 品种B的价格序列

        return:
            beta: 回归系数
        '''

        X = np.array(series_a).reshape(-1, 1)
        y = np.array(series_b)

        model = LinearRegression()
        model.fit(X, y)
        beta = model.coef_[0]

        return beta

    def prepare_for_pair_trade(self, categary_a: str, categary_b: str, time_frame: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'], start: Union[Tuple, datetime], end: Union[Tuple, datetime]) -> Tuple[float, float, float, float]:
        '''
        配对交易准备函数,计算价差均值、标准差、回归系数和ADF检验的p值

        arguments:
            categary_a: 交易品种A
            categary_b: 交易品种B
            time_frame: 时间周期，'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
            start: 起始时间，可以是时间元组或 datetime 对象
            end: 结束时间，可以是时间元组或 datetime 对象

        return:
            spread_mean: 价差均值
            spread_std: 价差标准差
            beta: 回归系数
            adf_pvalue: ADF检验的p值
        '''
        price_series_a = self._get_price_series(
            categary_a, time_frame, start, end)
        price_series_b = self._get_price_series(
            categary_b, time_frame, start, end)

        # 保持两个序列长度一致，将长序列reindex到短序列长度
        if len(price_series_a) > len(price_series_b):
            price_series_a = price_series_a.reindex(
                price_series_b.index)
        elif len(price_series_b) > len(price_series_a):
            price_series_b = price_series_b.reindex(
                price_series_a.index)

        # 转换为列表形式
        price_series_a = price_series_a.dropna().tolist()
        price_series_b = price_series_b.dropna().tolist()

        # 计算回归系数
        beta = self._regression_coeff(price_series_a, price_series_b)

        # 计算原始价差序列
        spread = [b - beta * a for a, b in zip(price_series_a, price_series_b)]

        # 去除极端值
        q1, q3 = np.percentile(spread, [25, 75])
        iqr = q3 - q1
        spread = [x for x in spread if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]

        # 计算价差均值和标准差
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))

        if spread_std < 0.000001:
            adf_pvalue = 1.0
        else:
            # 计算ADF检验p值
            adf_pvalue = self._adf_test(spread)

        return spread_mean, spread_std, beta, adf_pvalue

    def pair_trade_signal(self, categary_a: str, categary_b: str, time_frame: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'], n_days: int, adf_threshold: float = 0.1, std_threshold: float = 1.5) -> Dict[str, Any]:
        '''
        配对交易信号函数

        arguments:
            categary_a: 交易品种A
            categary_b: 交易品种B
            time_frame: 时间周期，'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
            adf_threshold: ADF检验阈值,默认为0.1
            n_days: 用于计算价差均值和标准差的天数

        return:
            signal: 交易信号，'long_a,short_b', 'short_a,long_b', 'close_positions'
            beta: 回归系数
            adf_result: 平稳性检验结果，True表示平稳，False表示不平稳
            z_score: 当前z-score值

        '''

        # std_threshold必须大于1
        if std_threshold <= 1:
            raise ValueError("std_threshold must be greater than 1")

        start_time = self._get_start_time_without_weekend(n_days)
        end_time = datetime.now(pytz.timezone("Etc/UTC"))

        spread_mean, spread_std, beta, adf_pvalue = self.prepare_for_pair_trade(
            categary_a, categary_b, time_frame, start_time, end_time)

        # 计算当前价差
        price_a = mt5.symbol_info_tick(categary_a).bid  # type: ignore
        price_b = mt5.symbol_info_tick(categary_b).bid  # type: ignore
        current_spread = price_b - beta * price_a

        # 判断ADF检验结果
        if adf_pvalue > adf_threshold:
            adf_result = False
        elif adf_pvalue <= adf_threshold:
            adf_result = True

        # 计算z-score
        z_score = (current_spread - spread_mean) / \
            spread_std if spread_std > 0 else 0

        # 生成交易信号
        if std_threshold < z_score < std_threshold+0.5:
            signal = 'long_a,short_b'
        elif -std_threshold-0.5 < z_score < -std_threshold:
            signal = 'short_a,long_b'
        else:
            signal = 'close_positions'

        return {'signal': signal, 'beta': beta, 'adf_result': adf_result, 'z_score': z_score}
