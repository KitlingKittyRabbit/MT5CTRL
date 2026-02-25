from mt5ctrl import Backtest  # type: ignore
import os
import sys
import itertools
from datetime import datetime
from typing import List

import pandas as pd
from dotenv import load_dotenv

# 将项目根目录加入搜索路径，便于导入 mt5ctrl
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../..")))


def _get_env(name: str, default: str = None) -> str:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        raise ValueError(f"环境变量 {name} 未设置")
    return value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ["1", "true", "yes", "y", "on"]


def _parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def _parse_list_float(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip() != ""]


def _parse_list_int(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def run_optimization():
    load_dotenv()

    # 账户 & 品种
    account = int(_get_env("MT5_ACCOUNT"))
    password = _get_env("MT5_PASSWORD")
    server = _get_env("MT5_SERVER")
    category_a = _get_env("PAIR_TRADE_CATEGORY_A")
    category_b = _get_env("PAIR_TRADE_CATEGORY_B")

    # 固定回测配置
    signal_timeframe = _get_env("BACKTEST_SIGNAL_TIMEFRAME")
    trading_timeframe = _get_env("BACKTEST_TRADING_TIMEFRAME")
    start_time = _parse_dt(_get_env("BACKTEST_OPT_START_TIME"))
    end_time = _parse_dt(_get_env("BACKTEST_OPT_END_TIME"))
    skip_weekend = _parse_bool(_get_env("BACKTEST_SKIP_WEEKEND", "true"))

    principal = float(_get_env("BACKTEST_PRINCIPAL"))
    total_lot = float(_get_env("BACKTEST_TOTAL_LOT"))
    resample_rule = _get_env("BACKTEST_RESAMPLE_RULE", "D")

    trading_cost = {
        category_a: float(_get_env("BACKTEST_TRADING_COST_A")),
        category_b: float(_get_env("BACKTEST_TRADING_COST_B")),
    }
    contract_size = {
        category_a: float(_get_env("BACKTEST_CONTRACT_SIZE_A")),
        category_b: float(_get_env("BACKTEST_CONTRACT_SIZE_B")),
    }

    # 参数网格
    n_days_list = _parse_list_int(_get_env("BACKTEST_OPT_N_DAYS_LIST"))
    min_entry_list = _parse_list_float(
        _get_env("BACKTEST_OPT_MIN_ENTRY_ZSCORE_LIST"))
    max_entry_list = _parse_list_float(
        _get_env("BACKTEST_OPT_MAX_ENTRY_ZSCORE_LIST"))
    take_profit_list = _parse_list_float(
        _get_env("BACKTEST_OPT_TAKE_PROFIT_ZSCORE_LIST"))
    stop_loss_list = _parse_list_float(
        _get_env("BACKTEST_OPT_STOP_LOSS_ZSCORE_LIST"))

    param_grid = {
        "n_days": n_days_list,
        "min_entry_zscore": min_entry_list,
        "max_entry_zscore": max_entry_list,
        "take_profit_zscore": take_profit_list,
        "stop_loss_zscore": stop_loss_list,
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_runs = len(combinations)

    print(f"总共需要测试 {total_runs} 种参数组合...")

    bt = Backtest(account=account, password=password, server=server)
    results = []

    for i, params in enumerate(combinations, 1):
        # 合理性过滤
        if not (params["min_entry_zscore"] < params["max_entry_zscore"] < params["stop_loss_zscore"]):
            continue

        print(f"\n[{i}/{total_runs}] 正在测试参数: {params}")
        try:
            bs_point = bt.genarate_bs_point4pair_trade(
                category_a=category_a,
                category_b=category_b,
                signal_timeframe=signal_timeframe,  # type: ignore[arg-type]
                n_days=params["n_days"],
                start_time=start_time,
                end_time=end_time,
                adf_threshold=float(_get_env("BACKTEST_ADF_THRESHOLD")),
                min_entry_zscore=params["min_entry_zscore"],
                take_profit_zscore=params["take_profit_zscore"],
                max_entry_zscore=params["max_entry_zscore"],
                stop_loss_zscore=params["stop_loss_zscore"],
                skip_weekend=skip_weekend,
                trading_timeframe=trading_timeframe,  # type: ignore[arg-type]
            )

            res = bt.backtest(
                bs_point=bs_point,
                principal=principal,
                trading_cost=trading_cost,
                total_lot=total_lot,
                contract_size=contract_size,
                resample_rule=resample_rule,
            )

            results.append({
                **params,
                "total_return": res["total_return"],
                "annualized_return": res["annualized_return"],
                "max_drawdown": res["max_drawdown"],
                "sharpe_ratio": res["sharpe_ratio"],
                "win_rate": res["win_rate"],
                "trade_count": len(res["trade_log"]),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"测试失败: {exc}")
            continue

    if not results:
        print("没有产生有效的回测结果。")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="sharpe_ratio", ascending=False)

    print("\n=== 优化完成，排名前 5 的参数组合 ===")
    print(df.head(5).to_string(index=False))

    output_file = os.path.join(os.getcwd(), "optimization_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\n完整结果已保存至 {output_file}")


if __name__ == "__main__":
    run_optimization()
