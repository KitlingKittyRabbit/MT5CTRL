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
    MetaTrader5 控制类,封装了 MetaTrader5 的初始化和登录功能

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
            result: 登录结果,True 表示登录成功,False 表示登录失败
        '''

        result = mt5.initialize(login=self.account,  # type: ignore
                                server=self.server,
                                password=self.password)

        return result


class Order(MetaTrader5Control):
    '''
    下单类,封装了 MetaTrader5 的下单功能

    '''

    def __init__(self, category: str, direction: Literal['long', 'short'], lot: float, account: int, password: str, server: str, stop_loss: Union[float, None] = None, take_profit: Union[float, None] = None) -> None:
        '''
        初始化下单类

        arguments:
            category: 交易品种,例如 'EURUSD'
            direction: 交易方向,'long' 或 'short'
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
        self.category = category
        self.direction = direction
        self.lot = lot
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        if self.login():
            mt5.symbol_select(self.category, True)  # type: ignore
            request = {
                "action": mt5.TRADE_ACTION_DEAL,  # 无关紧要
                "symbol": self.category,  # 交易品种
                "volume": self.lot,  # 交易量
                "type": mt5.ORDER_TYPE_SELL if self.direction == 'short' else mt5.ORDER_TYPE_BUY,  # 卖单或买单
                # 开单价格
                "price": mt5.symbol_info_tick(self.category).bid if self.direction == 'short' else mt5.symbol_info_tick(  # type: ignore
                    self.category).ask,
            }
            if self.stop_loss is not None:
                request["sl"] = self.stop_loss  # 止损
            if self.take_profit is not None:
                request["tp"] = self.take_profit  # 止盈
            result = mt5.order_send(request)  # type: ignore
            print(result)
            self.order_number = result.order
            self.retcode = result.retcode
        else:
            print("登录失败")

    def close_order(self):
        '''
        平仓函数

        return:
            result: 平仓结果
        '''
        mt5.symbol_select(self.category, True)  # type: ignore
        request = {
            "action": mt5.TRADE_ACTION_DEAL,  # 无关紧要
            "symbol": self.category,  # 交易品种
            "volume": self.lot,  # 交易量
            "type": mt5.ORDER_TYPE_BUY if self.direction == 'short' else mt5.ORDER_TYPE_SELL,  # 买单或卖单
            "position": self.order_number,
        }

        # 发送交易请求
        result = mt5.order_send(request)  # type: ignore

        return result


class Signal(MetaTrader5Control):
    '''
    信号类,封装了各种信号的生成函数
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
        时间转换函数,将时间元组转换为 datetime 对象

        arguments:
            time_tuple: 时间元组,格式为 (年,月,日,时,分)

        return:
            dt: datetime 对象
        '''
        year, month, day, hour, minute = time_tuple
        timezone = pytz.timezone("Etc/UTC")
        dt = datetime(year, month, day, hour, minute, tzinfo=timezone)
        return dt

    def get_min_lot(self, category: str) -> float:
        '''
        获取品种的最小手数

        arguments:
            category: 交易品种

        return:
            min_lot: 最小手数
        '''
        mt5.symbol_select(category, True)  # type: ignore
        symbol_info = mt5.symbol_info(category)  # type: ignore
        if symbol_info is None:
            return 0.01  # 默认最小手数
        return float(symbol_info.volume_min)

    def _get_start_time(self, n_days: int, current_time: Union[datetime, None] = None, skip_weekend: bool = True) -> datetime:
        '''
        获取距离指定时间n天的起始时间

        arguments:
            n_days: 距离现在的天数(如果skip_weekend=True，则为工作日天数)
            current_time: 当前时间，如果为None则使用现在时间
            skip_weekend: 是否跳过周末，True表示计算n个工作日，False表示计算n个自然日

        return:
            start_time: 起始时间
        '''
        if current_time is None:
            current_time = datetime.now(pytz.timezone("Etc/UTC"))

        if not skip_weekend:
            # 不跳过周末，直接减去n天
            return current_time - timedelta(days=n_days)

        # 跳过周末，计算n个工作日之前的日期
        start_time = current_time
        days_count = 0

        while days_count < n_days:
            start_time -= timedelta(days=1)
            # 只有在工作日时才计数
            if start_time.weekday() < 5:  # 0-4 是周一到周五
                days_count += 1

        return start_time

    def _interpolate_timestamps(self, timestamps: List[int]) -> List[int]:
        '''
        对时间戳进行插值处理,如果两个时间戳之间的间隔大于1秒且小于60秒,则插入缺失的时间戳，否则不进行插值

        arguments:
            timestamps: 原始时间戳列表

        return:
            interpolated_timestamps: 插值后的时间戳列表
        '''
        timestamps = sorted(timestamps)
        interpolated_timestamps = []
        for i in range(len(timestamps) - 1):
            current_ts = timestamps[i]
            next_ts = timestamps[i + 1]
            interpolated_timestamps.append(current_ts)

            time_diff = next_ts - current_ts
            if 1 < time_diff < 60:
                # 插入缺失的时间戳
                for ts in range(current_ts + 1, next_ts):
                    interpolated_timestamps.append(ts)
        interpolated_timestamps.append(timestamps[-1])
        # 去重并排序
        interpolated_timestamps = sorted(list(set(interpolated_timestamps)))
        return interpolated_timestamps

    def _get_price_series(self, category: str, time_frame: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'S1'], start: Union[Tuple, datetime], end: Union[Tuple, datetime]) -> pd.Series:
        '''
        获取交易品种的价格序列

        arguments:
            category: 交易品种
            start: 起始时间,可以是时间元组或 datetime 对象,例如 (2026,1,30,0,0) 或 datetime(2026,1,30,0,0)
            end: 结束时间,可以是时间元组或 datetime 对象,例如 (2026,2,8,0,0) 或 datetime(2026,2,8,0,0)
            time_frame: 时间周期,'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1','S1'

        return:
            price_series: 价格序列,pd.Series形式返回,索引为时间戳
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

        if time_frame in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']:
            timeframe_dict = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                              'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
            timeframe = timeframe_dict[time_frame]

            mt5.symbol_select(category, True)  # type: ignore

            rates = mt5.copy_rates_range(  # type: ignore
                category, timeframe, start_dt, end_dt)
            df = pd.DataFrame(rates)
            df.set_index('time', inplace=True)
            price_series = df['close']
        elif time_frame == 'S1':
            ticks = mt5.copy_ticks_range(  # type: ignore
                category, start_dt, end_dt, mt5.COPY_TICKS_ALL)
            ticks_df = pd.DataFrame(ticks)
            ticks_df.set_index(ticks_df['time'], inplace=True)
            ticks_df = ticks_df[~ticks_df.index.duplicated(keep='first')]
            ticks_df['price'] = [
                (a+b)/2 for a, b in zip(ticks_df['bid'].to_list(), ticks_df['ask'].to_list())]
            timestamp = ticks_df.index.to_list()
            all_timestamps = self._interpolate_timestamps(timestamp)
            ticks_df = ticks_df.reindex(all_timestamps)
            price_series = ticks_df['price'].interpolate(method='linear')

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

    def prepare_for_pair_trade(self, category_a: str, category_b: str, time_frame: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'S1'], start: Union[Tuple, datetime], end: Union[Tuple, datetime]) -> Tuple[float, float, float, float]:
        '''
        配对交易准备函数,计算价差均值、标准差、回归系数和ADF检验的p值

        arguments:
            category_a: 交易品种A
            category_b: 交易品种B
            time_frame: 时间周期,'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
            start: 起始时间,可以是时间元组或 datetime 对象
            end: 结束时间,可以是时间元组或 datetime 对象

        return:
            spread_mean: 价差均值
            spread_std: 价差标准差
            beta: 回归系数
            adf_pvalue: ADF检验的p值
        '''
        price_series_a = self._get_price_series(
            category_a, time_frame, start, end)
        price_series_b = self._get_price_series(
            category_b, time_frame, start, end)

        # 保持两个序列长度一致,将长序列reindex到短序列长度
        if len(price_series_a) > len(price_series_b):
            price_series_a = price_series_a.reindex(
                price_series_b.index)
        elif len(price_series_b) > len(price_series_a):
            price_series_b = price_series_b.reindex(
                price_series_a.index)
        aligned = pd.concat([price_series_a, price_series_b],
                            axis=1, keys=["a", "b"]).dropna()

        # 转换为列表形式
        price_series_a = aligned["a"].tolist()
        price_series_b = aligned["b"].tolist()

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

    def _pair_trade_signal_at_time(self, category_a: str, category_b: str, time_frame: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'S1'],
                                   n_days: int, current_time: datetime, price_a: float, price_b: float,
                                   adf_threshold: float = 0.1, entry_zscore: float = 1.6, exit_zscore: float = 0.8, skip_weekend: bool = True) -> Dict[str, Any]:
        '''
        在指定时间点计算配对交易信号(用于回测)

        arguments:
            category_a: 交易品种A
            category_b: 交易品种B
            time_frame: 时间周期,'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'S1'
            n_days: 用于计算价差均值和标准差的天数(如果skip_weekend=True则为工作日天数)
            current_time: 当前时间点
            price_a: 品种A的当前价格
            price_b: 品种B的当前价格
            adf_threshold: ADF检验的p值阈值,默认为0.1
            entry_zscore: 进场z-score阈值,默认为1.6
            exit_zscore: 平仓z-score阈值,默认为0.8
            skip_weekend: 是否跳过周末,True表示计算工作日,False表示计算自然日(适用于周末不休盘的品种如加密货币)

        return:
            signal: 交易信号,'long_a,short_b', 'short_a,long_b', 'take_profit', 'stop_loss' 或 'no_action'
            beta: 回归系数
            adf_result: 平稳性检验结果,True表示平稳,False表示不平稳
            z_score: 当前z-score值
        '''
        # entry_zscore必须大于1
        if entry_zscore <= 1:
            raise ValueError("entry_zscore must be greater than 1")

        # 计算历史数据的起始时间
        start_time = self._get_start_time(n_days, current_time, skip_weekend)

        spread_mean, spread_std, beta, adf_pvalue = self.prepare_for_pair_trade(
            category_a, category_b, time_frame, start_time, current_time)

        # 计算当前价差
        current_spread = price_b - beta * price_a

        # 判断ADF检验结果
        adf_result = adf_pvalue <= adf_threshold

        # 计算z-score
        z_score = (current_spread - spread_mean) / \
            spread_std if spread_std > 0 else 0

        # 生成交易信号(开仓与平仓间隔0.1个标准差,防止频繁交易)
        if entry_zscore < z_score < entry_zscore+0.5:
            signal = 'long_a,short_b'
        elif -entry_zscore-0.5 < z_score < -entry_zscore:
            signal = 'short_a,long_b'
        elif -exit_zscore < z_score < exit_zscore:
            signal = 'take_profit'
        elif z_score > entry_zscore + 0.6 or z_score < -entry_zscore - 0.6:
            signal = 'stop_loss'
        else:
            signal = 'no_action'

        return {'signal': signal, 'beta': beta, 'adf_result': adf_result, 'z_score': z_score}

    def pair_trade_signal(self, category_a: str, category_b: str, time_frame: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'S1'], n_days: int, adf_threshold: float = 0.1, entry_zscore: float = 1.6, exit_zscore: float = 0.8, skip_weekend: bool = True) -> Dict[str, Any]:
        '''
        配对交易信号函数

        arguments:
            category_a: 交易品种A
            category_b: 交易品种B
            time_frame: 时间周期,'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'S1'
            n_days: 用于计算价差均值和标准差的天数(如果skip_weekend=True则为工作日天数)
            adf_threshold: ADF检验的p值阈值,默认为0.1
            entry_zscore: 进场z-score阈值,默认为1.6
            exit_zscore: 平仓z-score阈值,默认为0.8
            skip_weekend: 是否跳过周末,True表示计算工作日,False表示计算自然日(适用于周末不休盘的品种如加密货币)

        return:
            signal: 交易信号,'long_a,short_b', 'short_a,long_b', 'take_profit', 'stop_loss' 或 'no_action'
            beta: 回归系数
            adf_result: 平稳性检验结果,True表示平稳,False表示不平稳
            z_score: 当前z-score值
            market_all_open: 市场是否开盘

        '''
        # 比对报价时间与当前时间,判断市场是否开盘
        market_all_open = True
        for category in [category_a, category_b]:
            tick = mt5.symbol_info_tick(category)  # type: ignore
            if tick is None:
                market_all_open = False
                break
            tick_time = datetime.fromtimestamp(
                tick.time, pytz.timezone("Etc/UTC"))
            now_time = datetime.now(pytz.timezone("Etc/UTC"))
            time_diff = now_time - tick_time
            if time_diff > timedelta(minutes=5):
                market_all_open = False
                break

        # 获取当前价格
        price_a = mt5.symbol_info_tick(category_a).bid  # type: ignore
        price_b = mt5.symbol_info_tick(category_b).bid  # type: ignore
        current_time = datetime.now(pytz.timezone("Etc/UTC"))

        # 调用内部函数计算信号
        result = self._pair_trade_signal_at_time(
            category_a, category_b, time_frame, n_days, current_time, price_a, price_b,
            adf_threshold, entry_zscore, exit_zscore, skip_weekend)
        result['market_all_open'] = market_all_open

        return result


class Backtest(MetaTrader5Control):
    '''
    回测类,封装了各种回测功能
    '''

    def __init__(self, account: int, password: str, server: str) -> None:
        '''
        初始化回测类

        arguments:
            account: 交易账户
            password: 交易密码
            server: 交易服务器

        return:
            None
        '''
        super().__init__(account, password, server)
        self.S = Signal(account, password, server)

    def genarate_bs_point4pair_trade(self, category_a: str, category_b: str, signal_timeframe: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'], n_days: int,
                                     start_time: Union[Tuple, datetime], end_time: Union[Tuple, datetime], adf_threshold: float = 0.1, entry_zscore: float = 1.6,
                                     exit_zscore: float = 0.8, skip_weekend: bool = True, trading_timeframe: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'S1'] = 'S1') -> pd.DataFrame:
        '''
        生成配对交易的开平点数据

        arguments:
            category_a: 交易品种A
            category_b: 交易品种B
            signal_timeframe: 信号计算时间周期(用于统计分析),'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
            n_days: 用于计算价差均值和标准差的天数(如果skip_weekend=True则为工作日天数)
            start_time: 回测的起始时间,可以是时间元组或 datetime 对象
            end_time: 回测的结束时间,可以是时间元组或 datetime 对象
            adf_threshold: ADF检验的p值阈值,默认为0.1
            entry_zscore: 进场z-score阈值,默认为1.6
            exit_zscore: 平仓z-score阈值,默认为0.8
            skip_weekend: 是否跳过周末,True表示计算工作日,False表示计算自然日(适用于周末不休盘的品种如加密货币)
            trading_timeframe: 交易执行时间周期(生成买卖点的时间粒度),默认'S1'

        return:
            bs_point: 开平点数据,包含以下列:
                price_a: 品种A的价格序列
                price_b: 品种B的价格序列
                signal: 交易信号
                bs_point_a: 品种A的开平点, 1表示开多, -1表示开空, 0表示平仓, NaN表示无动作
                bs_point_b: 品种B的开平点, 1表示开多, -1表示开空, 0表示平仓, NaN表示无动作
                z_score: z-score值
                ratio_a: 交易中A的仓位比例
                ratio_b: 交易中B的仓位比例(ratio_a + ratio_b = 1)
        '''
        print("正在获取交易执行价格数据...")
        price_col_a = f'price_{category_a}'
        price_col_b = f'price_{category_b}'
        bs_col_a = f'bs_point_{category_a}'
        bs_col_b = f'bs_point_{category_b}'
        ratio_col_a = f'ratio_{category_a}'
        ratio_col_b = f'ratio_{category_b}'

        # 获取交易执行时间粒度的价格序列(用于生成买卖点)
        price_series_a = self.S._get_price_series(
            category_a, trading_timeframe, start_time, end_time)
        price_series_b = self.S._get_price_series(
            category_b, trading_timeframe, start_time, end_time)

        # 对齐两个序列
        df = pd.concat([price_series_a, price_series_b], axis=1,
                       keys=[price_col_a, price_col_b]).dropna()

        print(f"交易数据点数: {len(df)}")

        # 如果trading_timeframe和signal_timeframe相同，直接使用df
        # 否则需要获取signal_timeframe的数据用于统计计算
        if trading_timeframe == signal_timeframe:
            signal_df = df.copy()
        else:
            print("正在获取信号计算价格数据...")
            # 扩展时间范围以确保有足够历史数据
            if isinstance(start_time, tuple):
                start_dt = self.S._time_convert(start_time)
            else:
                start_dt = start_time
            extended_start = self.S._get_start_time(
                n_days + 5, start_dt, skip_weekend)

            signal_price_a = self.S._get_price_series(
                category_a, signal_timeframe, extended_start, end_time)
            signal_price_b = self.S._get_price_series(
                category_b, signal_timeframe, extended_start, end_time)
            signal_df = pd.concat([signal_price_a, signal_price_b], axis=1,
                                  keys=[price_col_a, price_col_b]).dropna()

        print(f"信号数据点数: {len(signal_df)}")
        print("开始计算滚动统计量...")

        # 计算需要的窗口大小(转换为数据点数)
        # 这里使用近似：假设M1每天约1440个点(24小时)，根据signal_timeframe调整
        timeframe_minutes = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                             'H1': 60, 'H4': 240, 'D1': 1440}
        points_per_day = 1440 / timeframe_minutes[signal_timeframe]

        if skip_weekend:
            # 工作日：5/7
            window_size = int(n_days * points_per_day * 5 / 7)
        else:
            window_size = int(n_days * points_per_day)

        # 确保窗口大小合理
        window_size = max(window_size, 100)  # 至少100个点
        window_size = min(window_size, len(signal_df) // 2)  # 不超过数据一半

        print(f"滚动窗口大小: {window_size}")

        # 使用滚动窗口计算beta和价差统计量
        betas = []
        spread_means = []
        spread_stds = []
        adf_pvalues = []

        for i in range(len(signal_df)):
            if i < window_size:
                betas.append(np.nan)
                spread_means.append(np.nan)
                spread_stds.append(np.nan)
                adf_pvalues.append(1.0)
            else:
                window_a = signal_df[price_col_a].iloc[i-window_size:i].values
                window_b = signal_df[price_col_b].iloc[i-window_size:i].values

                # 计算beta
                beta = self.S._regression_coeff(
                    window_a.tolist(), window_b.tolist())

                # 计算价差
                spread = window_b - beta * window_a

                # 去除极端值
                q1, q3 = np.percentile(spread, [25, 75])
                iqr = q3 - q1
                spread_clean = spread[(spread >= q1 - 1.5*iqr)
                                      & (spread <= q3 + 1.5*iqr)]

                # 计算统计量
                spread_mean = float(np.mean(spread_clean))
                spread_std = float(np.std(spread_clean))

                # ADF检验
                if spread_std > 0.000001 and len(spread_clean) > 10:
                    adf_pvalue = self.S._adf_test(spread_clean.tolist())
                else:
                    adf_pvalue = 1.0

                betas.append(beta)
                spread_means.append(spread_mean)
                spread_stds.append(spread_std)
                adf_pvalues.append(adf_pvalue)

            # 进度提示
            if (i + 1) % 1000 == 0 or i == len(signal_df) - 1:
                print(f"  预计算进度: {i+1}/{len(signal_df)}")

        # 将统计量添加到signal_df
        signal_df['beta'] = betas
        signal_df['spread_mean'] = spread_means
        signal_df['spread_std'] = spread_stds
        signal_df['adf_pvalue'] = adf_pvalues

        print("统计量计算完成，开始生成交易信号...")

        # 初始化结果列
        df['signal'] = 'no_action'
        df[bs_col_a] = np.nan
        df[bs_col_b] = np.nan
        df['z_score'] = np.nan
        df[ratio_col_a] = np.nan
        df[ratio_col_b] = np.nan
        df['adf_result'] = False

        # 遍历交易时间点，映射到最近的信号时间点
        position_open = False

        for idx, (timestamp, row) in enumerate(df.iterrows()):
            # 找到最近的signal时间点
            signal_idx = signal_df.index.searchsorted(
                timestamp, side='right') - 1

            if signal_idx < 0 or signal_idx >= len(signal_df):
                continue

            signal_row = signal_df.iloc[signal_idx]

            # 跳过统计量未准备好的点
            if pd.isna(signal_row['beta']):
                continue

            # 计算当前z-score
            current_spread = row[price_col_b] - \
                signal_row['beta'] * row[price_col_a]
            z_score = (current_spread - signal_row['spread_mean']) / \
                signal_row['spread_std'] if signal_row['spread_std'] > 0 else 0
            adf_result = signal_row['adf_pvalue'] <= adf_threshold

            # 根据回归关系 b = beta * a，令 ratio_b = beta * ratio_a 且 ratio_a + ratio_b = 1
            beta = float(signal_row['beta'])
            beta_sign = 1.0 if beta >= 0 else -1.0
            beta_abs = abs(beta)
            denominator = 1 + beta_abs
            if abs(denominator) < 1e-12:
                ratio_a = np.nan
                ratio_b = np.nan
            else:
                ratio_a = 1 / denominator
                ratio_b = beta_abs / denominator

            # 生成交易信号(开仓与平仓间隔0.1个标准差,防止频繁交易)
            if entry_zscore < z_score < entry_zscore+0.5:
                signal = 'long_a,short_b'
            elif -entry_zscore-0.5 < z_score < -entry_zscore:
                signal = 'short_a,long_b'
            elif -exit_zscore < z_score < exit_zscore:
                signal = 'take_profit'
            elif z_score > entry_zscore + 0.6 or z_score < -entry_zscore - 0.6:
                signal = 'stop_loss'
            else:
                signal = 'no_action'

            # 记录结果
            df.loc[timestamp, 'signal'] = signal
            df.loc[timestamp, 'z_score'] = z_score
            df.loc[timestamp, ratio_col_a] = ratio_a
            df.loc[timestamp, ratio_col_b] = ratio_b
            df.loc[timestamp, 'adf_result'] = adf_result

            # 根据信号设置开平点
            if not position_open:
                if signal == 'long_a,short_b' and adf_result:
                    df.loc[timestamp, bs_col_a] = 1  # 开多A
                    df.loc[timestamp, bs_col_b] = - \
                        1 * beta_sign  # B腿方向随beta符号调整
                    position_open = True
                elif signal == 'short_a,long_b' and adf_result:
                    df.loc[timestamp, bs_col_a] = -1  # 开空A
                    df.loc[timestamp, bs_col_b] = 1 * \
                        beta_sign  # B腿方向随beta符号调整
                    position_open = True
            else:
                # 有持仓时检查平仓信号
                if signal in ['take_profit', 'stop_loss']:
                    df.loc[timestamp, bs_col_a] = 0  # 平仓
                    df.loc[timestamp, bs_col_b] = 0  # 平仓
                    position_open = False

            # 进度提示
            if (idx + 1) % 10000 == 0:
                print(f"  已处理 {idx + 1}/{len(df)} 个交易点...")

        print("完成！")
        return df

    def backtest(self, bs_point: pd.DataFrame, principal: float, trading_cost: Dict[str, float], total_lot: float, contract_size: Dict[str, float], resample_rule: str = 'D') -> Dict[str, Any]:
        '''
        回测函数

        arguments:
            bs_point: 回测的开平点数据,如果是单一品种,需要包含以下列:
                price_x: 品种x的价格序列
                bs_point_x:品种x的开平点, 1表示开多, -1表示开空, 0表示平仓, NaN表示无动作
                如果是多品种交易,在上述基础上,需要包含以下列:
                ratio_x: 交易中品种x的仓位比例(ratio_a + ratio_b + ... = 1)
                index: 时间戳索引,表示交易点的时间
            principal: 初始资金
            trading_cost: 每手交易成本,例如 {'EURUSD': 5, 'USDJPY': 3}表示每手交易EURUSD的成本为5美元,每手交易USDJPY的成本为3美元
            total_lot: 每次交易的总手数
            contract_size: 合约数量,例如 {'EURUSD': 100000, 'USDJPY': 1000}
            resample_rule: 权益曲线重采样频率,用于计算固定间隔收益率,默认'D'(日频)
        return:
            backtest_result: 回测结果,包含以下内容:
                total_return: 总收益率
                annualized_return: 年化收益率
                max_drawdown: 最大回撤
                sharpe_ratio: 夏普比率
                equity_curve: 权益曲线,包含时间戳和对应的资金余额
                win_rate: 胜率
        '''
        # 提取价格列、开平点列和仓位比例列名称
        price_columns = [
            col for col in bs_point.columns if col.startswith('price_')]
        bs_point_columns = [
            col for col in bs_point.columns if col.startswith('bs_point_')]
        ratio_columns = [
            col for col in bs_point.columns if col.startswith('ratio_')]
        # 提取品种名称
        category_names = [col[len('price_'):] for col in price_columns]
        if len(price_columns) == 0 or len(bs_point_columns) == 0:
            raise ValueError("bs_point缺少price_或bs_point_列")

        # 如果没有仓位比例列，默认等权分配
        if len(ratio_columns) == 0:
            equal_ratio = 1.0 / len(category_names)
            for category in category_names:
                bs_point[f'ratio_{category}'] = equal_ratio

        # 提取交易点数据,去除无动作的点
        trade_points = bs_point[bs_point[bs_point_columns[0]].notna()].copy()
        if len(trade_points) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': np.nan,
                'annualized_volatility': np.nan,
                'max_drawdown': np.nan,
                'sharpe_ratio': np.nan,
                'win_rate': np.nan,
                'equity_curve': pd.DataFrame(columns=['equity', 'equity_peak', 'drawdown']),
                'trade_log': pd.DataFrame()
            }

        # 确保索引为datetime类型,如果不是则尝试转换
        if not pd.api.types.is_datetime64_any_dtype(trade_points.index):
            if pd.api.types.is_integer_dtype(trade_points.index) or pd.api.types.is_float_dtype(trade_points.index):
                trade_points.index = pd.to_datetime(
                    trade_points.index, unit='s', utc=True)
            else:
                trade_points.index = pd.to_datetime(
                    trade_points.index, errors='coerce', utc=True)

        # 获取品种最小手数
        min_lots = {}
        for category in category_names:
            min_lots[category] = self.S.get_min_lot(category)

        # 交易金额=总手数*仓位比例*价格*合约数量
        for category in category_names:
            # 逐行取 max，确保每笔手数不低于最小手数
            raw_lot = total_lot * trade_points[f'ratio_{category}']
            actual_lot = np.maximum(min_lots[category], raw_lot)
            trade_points[f'actual_lot_{category}'] = actual_lot
            trade_points[f'amount_{category}'] = actual_lot * \
                trade_points[f'price_{category}'] * contract_size[category]
        # 若开仓点交易金额=a，平仓点的交易金额=b,则做多时盈亏为b-a,做空时盈亏为-(b-a)
        open_points = trade_points[[True if x in [
            1, -1] else False for x in trade_points[bs_point_columns[0]]]].copy()
        close_points = trade_points[[
            True if x == 0 else False for x in trade_points[bs_point_columns[0]]]].copy()

        if len(open_points) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': np.nan,
                'annualized_volatility': np.nan,
                'max_drawdown': np.nan,
                'sharpe_ratio': np.nan,
                'win_rate': np.nan,
                'equity_curve': pd.DataFrame(columns=['equity', 'equity_peak', 'drawdown']),
                'trade_log': pd.DataFrame()
            }

        if len(open_points) != len(close_points):
            raise ValueError(
                f"开仓点({len(open_points)})与平仓点({len(close_points)})数量不一致，无法逐笔配对计算盈亏")

        for category in category_names:
            open_price = open_points[f'price_{category}'].to_numpy()
            close_price = close_points[f'price_{category}'].to_numpy()
            # 1=多，-1=空
            direction = open_points[f'bs_point_{category}'].to_numpy()
            lot = open_points[f'actual_lot_{category}'].to_numpy()
            # 按开仓方向计算点差收益：多=(平-开)，空=(开-平)
            price_diff = close_price - open_price
            open_points[f'profit_{category}'] = direction * \
                price_diff * lot * contract_size[category]
        # 计算总盈亏金额
        open_points['total_profit'] = open_points[[
            f'profit_{category}' for category in category_names]].sum(axis=1)
        # 计算交易成本（使用实际手数）
        open_points['total_cost'] = 0.0
        for category in category_names:
            open_points['total_cost'] += trading_cost[category] * \
                open_points[f'actual_lot_{category}']
        # 计算净盈亏金额
        open_points['net_profit'] = open_points['total_profit'] - \
            open_points['total_cost']
        # .cumsum()函数计算盈亏金额的累积和,加上初始资金即为资金曲线
        open_points['equity'] = principal + open_points['net_profit'].cumsum()
        # 计算总收益率
        total_return = (open_points['equity'].iloc[-1] - principal) / principal
        # 计算年化收益率
        num_years = (open_points.index[-1] -
                     open_points.index[0]).total_seconds() / (365.25 * 24 * 3600)
        if num_years > 0 and (1 + total_return) > 0:
            annualized_return = (1 + total_return) ** (1 / num_years) - 1
        else:
            annualized_return = np.nan
        # 计算最大回撤,首先计算资金曲线的历史最高点,然后计算回撤=（历史最高点-当前资金）/历史最高点,最后取最大值
        open_points['equity_peak'] = open_points['equity'].cummax()
        open_points['drawdown'] = (
            open_points['equity_peak'] - open_points['equity']) / open_points['equity_peak']
        max_drawdown = open_points['drawdown'].max()
        # 计算夏普比率（传统法）：先将权益对齐到固定频率，再计算收益率波动
        annualized_volatility = np.nan
        sharpe_ratio = np.nan

        equity_series = open_points['equity'].dropna()
        if len(equity_series) >= 2 and (equity_series > 0).all():
            aligned_equity = equity_series.resample(
                resample_rule).last().ffill().dropna()
            aligned_returns = aligned_equity.pct_change().dropna()

            if len(aligned_returns) > 1:
                step_seconds = aligned_equity.index.to_series(
                ).diff().dropna().dt.total_seconds().median()
                if step_seconds and step_seconds > 0:
                    periods_per_year = (365.25 * 24 * 3600) / step_seconds
                    annualized_volatility = aligned_returns.std(
                        ddof=1) * np.sqrt(periods_per_year)

                    risk_free_rate = 0.0
                    if pd.notna(annualized_volatility) and annualized_volatility > 0 and pd.notna(annualized_return):
                        sharpe_ratio = (annualized_return -
                                        risk_free_rate) / annualized_volatility

        # 计算胜率
        win_rate = (open_points['net_profit'] > 0).mean() if len(
            open_points) > 0 else np.nan

        backtest_result = {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return) if pd.notna(annualized_return) else np.nan,
            'annualized_volatility': float(annualized_volatility) if pd.notna(annualized_volatility) else np.nan,
            'max_drawdown': float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
            'sharpe_ratio': float(sharpe_ratio) if pd.notna(sharpe_ratio) else np.nan,
            'win_rate': float(win_rate) if pd.notna(win_rate) else np.nan,
            'equity_curve': open_points[['equity', 'equity_peak', 'drawdown']].copy(),
            'trade_log': open_points.copy()
        }

        return backtest_result
