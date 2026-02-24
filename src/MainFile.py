import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import logging
import os
from typing import Dict, Optional
import warnings
import pytz
import pandas_ta as ta
import calendar
import time
import threading
import json
import sys
import os

# Add src directory to Python path for Render
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger_config import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")

# --------------------------- LOGGING ---------------------------

logger = get_logger()

# --------------------------- EXPIRY HELPERS ---------------------------

def get_last_thursday(year: int, month: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    while dt.weekday() != 3:
        dt -= timedelta(days=1)
    return dt


def get_current_or_next_monthly_expiry(now_ist: Optional[datetime] = None) -> datetime:
    now_ist = now_ist or datetime.now(IST)
    y, m = now_ist.year, now_ist.month
    exp_dt = get_last_thursday(y, m)
    if exp_dt.date() < now_ist.date():
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
        exp_dt = get_last_thursday(y, m)
    return exp_dt

def get_next_tuesday(now_ist: Optional[datetime] = None) -> datetime:
    """
    Weekly expiry helper: next Tuesday.
    If today is Tuesday, go to next week's Tuesday.
    """
    now_ist = now_ist or datetime.now(IST)
    if now_ist.weekday() < 1:
        days_ahead = 1 - now_ist.weekday()
    else:
        days_ahead = 7 - (now_ist.weekday() - 1)
    return (now_ist + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

def format_expiry_code(dt: datetime) -> str:
    yy = dt.year % 100
    mon = dt.strftime("%b").upper()
    return f"{yy:02d}{mon}"


# --------------------------- CONFIG ---------------------------

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME = "2"
    HIST_DAYS = 90

    TAKE_PROFIT_PCT   = 15
    STOP_LOSS_PCT     = 5
    DOJI_MAX_BODY_PCT = 0.9
    OPEN_TOLERANCE    = 0.1

    LIMIT_BUFFER = 0.5

    USE_EMA_FILTER       = False
    EMA_LEN              = 21
    USE_EMA_SLOPE        = False
    EMA_SLOPE_BARS       = 10
    USE_SLOPE_STRENGTH   = False
    SLOPE_LOOKBACK       = 15
    MIN_SLOPE_PERCENT    = 0.1

    USE_RSI_FILTER   = False
    RSI_LEN          = 14
    RSI_OVERSOLD     = 30
    RSI_OVERBOUGHT   = 70

    STRIKE_STEP     = 100
    # Updated to NSE Jan 2026 lot size: 65. [web:1][web:26]
    NIFTY_LOT_SIZE  = 65

    EMA9_LEN              = 9
    EMA9_ANGLE_LOOKBACK   = 3
    EMA9_MIN_ANGLE_DEG    = 30.0

    HTF_RULE    = "60T"
    HTF_EMA_LEN = 21

    SESSION_START = dtime(9, 20)
    SESSION_END   = dtime(15, 30)

    BO_QTY = 65

    # Tick size for index options (NIFTY) is 0.05. [web:47]
    OPTION_TICK_SIZE = 0.05

   # NEW: choose expiry style
    EXPIRY_TYPE = "WEEKLY"   # "MONTHLY" or "WEEKLY"

    def __init__(self):
        now = datetime.now(IST)
        if self.EXPIRY_TYPE == "WEEKLY":
            exp_dt = get_next_tuesday(now)
        else:
            exp_dt = get_current_or_next_monthly_expiry(now)
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logging.getLogger(__name__).info(
            f"Using {self.EXPIRY_TYPE} expiry {exp_dt.date()} with code {self.EXPIRY_CODE}"
        )


# --------------------------- ORDER MANAGER (BO) ---------------------------

class OrderManager:
    def __init__(self, fyers_client: fyersModel.FyersModel, config: TradingConfig):
        self.fyers = fyers_client
        self.config = config

    @staticmethod
    def _round_to_tick(value: float, tick: float) -> float:
        """
        Round a numeric value to the nearest multiple of tick size. [web:47]
        """
        return round(round(value / tick) * tick, 2)

    def _compute_bo_sl_tp_from_pct(self, opt_entry: float) -> tuple[float, float]:
        """
        Convert STOP_LOSS_PCT / TAKE_PROFIT_PCT into rupee points for BO legs,
        which must be distance-from-entry, not absolute prices. [web:50]
        """
        sl_pct = self.config.STOP_LOSS_PCT
        tp_pct = self.config.TAKE_PROFIT_PCT

        sl_points = opt_entry * (sl_pct / 100.0)
        tp_points = opt_entry * (tp_pct / 100.0)

        return sl_points, tp_points

    def place_bo_for_signal(
        self,
        signal_row: pd.Series,
        sl_points: float = None,
        tp_points: float = None,
        qty: int = None,
    ) -> Optional[str]:
        """
        For both BUY/SELL index signals we BUY the option (CE or PE) via BO.
        Entry: stop‑limit using option LTP as reference:
            stopPrice  = option_close
            limitPrice = option_close + 2
        SL/TP: computed as % of option entry price, passed as points and tick-aligned.
        """
        symbol = signal_row.get("option_symbol")
        if not symbol:
            logger.warning(f"Signal has no option_symbol, skipping BO: {signal_row.to_dict()}")
            return None

        side = 1
        qty = qty or self.config.BO_QTY

        opt_close = signal_row.get("option_entry_price")
        if opt_close is None or (isinstance(opt_close, float) and np.isnan(opt_close)):
            logger.warning(
                "No option_entry_price in signal, cannot build stop‑limit entry: "
                f"{signal_row.to_dict()}"
            )
            return None

        opt_close = float(opt_close)

        if sl_points is None or tp_points is None:
            sl_calc, tp_calc = self._compute_bo_sl_tp_from_pct(opt_close)
            sl_points = sl_points or sl_calc
            tp_points = tp_points or tp_calc

        tick = self.config.OPTION_TICK_SIZE

        sl_points   = self._round_to_tick(sl_points, tick)
        tp_points   = self._round_to_tick(tp_points, tick)
        stop_price  = self._round_to_tick(opt_close, tick)
        limit_price = self._round_to_tick(opt_close + 2.0, tick)

        data = {
            "symbol": symbol,
            "qty": int(qty),
            "type": 4,
            "side": side,
            "productType": "BO",
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": float(sl_points),
            "takeProfit": float(tp_points),
            "isSliceOrder": False,
        }

        try:
            logger.info(
                f"🚀 Placing BO (SL‑L entry) for signal: {signal_row.to_dict()} | "
                f"qty={qty}, SL_pts={sl_points}, TP_pts={tp_points}, "
                f"stopPrice={stop_price}, limitPrice={limit_price}"
            )
            resp = self.fyers.place_order(data=data)
            logger.info(f"Place BO response: {json.dumps(resp, indent=2)}")

            if isinstance(resp, dict) and resp.get("s") == "ok":
                order_id = resp.get("id")
                logger.info(f"BO submitted, order_id={order_id}")
                return order_id
            else:
                logger.error(f"BO rejected/failed: {resp}")
                return None
        except Exception as e:
            logger.error(f"BO placement error: {e}")
            return None


# --------------------------- TRADING BOT ---------------------------

class TradingBot:
    def __init__(self):
        self.config = TradingConfig()
        self.fyers = self._init_fyers()
        self.position_size: int = 0
        self.last_trade: Optional[str] = None

        self.app_id: Optional[str] = None
        self.access_token: Optional[str] = None
        with open("fyers_appid.txt", "r") as f:
            self.app_id = f.read().strip()
        with open("fyers_token.txt", "r") as f:
            self.access_token = f.read().strip()

        logger.info("TradingBot initialized (backtest + live).")

    def _init_fyers(self):
        if not all(os.path.exists(f) for f in ["fyers_appid.txt", "fyers_token.txt"]):
            raise FileNotFoundError("Missing fyers_appid.txt or fyers_token.txt")

        app_id = open("fyers_appid.txt", "r").read().strip()
        token = open("fyers_token.txt", "r").read().strip()

        if not app_id or not token:
            raise ValueError("Empty app_id or token in credential files")

        # v3 client with pre-generated access token. [web:47][web:50]
        fyers_v3 = fyersModel.FyersModel(
            client_id=app_id,
            token=token,
            is_async=False,
            log_path="logs"
        )
        logger.info("Fyers V3 FyersModel initialized")
        return fyers_v3

    # ---------- HISTORY ----------

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end_dt_ist = datetime.now(IST)
        start_dt_ist = end_dt_ist - timedelta(days=days)

        range_from = start_dt_ist.strftime("%Y-%m-%d")
        range_to   = end_dt_ist.strftime("%Y-%m-%d")

        data = {
            "symbol": symbol,
            "resolution": self.config.TIMEFRAME,
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1",
        }
        resp = self.fyers.history(data=data)

        if isinstance(resp, dict):
            status = resp.get("s")
            candles = resp.get("candles", [])
            if status != "ok" or not candles:
                logger.warning(f"No history for {symbol} {range_from}->{range_to}, starting empty.")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            candles = resp.get("candles", []) if hasattr(resp, "get") else []
            if not candles:
                logger.warning(f"No history for {symbol} (non-dict resp), starting empty.")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_index_data(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.HIST_DAYS
        df = self._fetch_history(self.config.UNDERLYING_SYMBOL, days)
        df.rename(
            columns={
                "open": "und_open",
                "high": "und_high",
                "low": "und_low",
                "close": "und_close",
                "volume": "und_volume",
            },
            inplace=True,
        )
        df.set_index("timestamp", inplace=True)
        df = df.between_time(
            self.config.SESSION_START,
            self.config.SESSION_END,
            inclusive="both",
        )
        logger.info(f"Warmup candles loaded in-session: {len(df)}")
        return df

    def _fetch_option_history(self, symbol: str, days: int) -> pd.DataFrame:
        df = self._fetch_history(symbol, days)
        df.set_index("timestamp", inplace=True)
        return df

    # ---------- OPTION PICKER ----------

    def _pick_itm_option_symbol(self, underlying_price: float, direction: str) -> str:
        c = self.config
        spot = float(underlying_price)
        step = c.STRIKE_STEP

        if direction == "BUY":
            atm_strike = (int(spot) // step) * step
            strike = atm_strike - step
            opt_type = "CE"
        else:
            atm_strike = ((int(spot) + step - 1) // step) * step
            strike = atm_strike + step
            opt_type = "PE"

        strike = max(step, (int(strike) // step) * step)
        strike_int = int(strike)

        symbol = f"NSE:NIFTY{c.EXPIRY_CODE}{strike_int}{opt_type}"
        return symbol

    # ---------- INDICATORS / HTF / REVERSAL ----------

    def calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[["und_open", "und_high", "und_low", "und_close"]].rename(
            columns={
                "und_open": "open",
                "und_high": "high",
                "und_low": "low",
                "und_close": "close",
            }
        )
        ha = ta.ha(
            open_=base["open"],
            high=base["high"],
            low=base["low"],
            close=base["close"],
        )
        df_ha = df.copy()
        df_ha["ha_open"]  = ha["HA_open"]
        df_ha["ha_high"]  = ha["HA_high"]
        df_ha["ha_low"]   = ha["HA_low"]
        df_ha["ha_close"] = ha["HA_close"]
        return df_ha

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df_ha = self.calculate_heikin_ashi(df)

        df_ha["ema"] = df_ha["und_close"].ewm(span=c.EMA_LEN, adjust=False).mean()

        bars = c.EMA_SLOPE_BARS
        df_ha["ema_slope_up"]   = df_ha["ema"] > df_ha["ema"].shift(bars)
        df_ha["ema_slope_down"] = df_ha["ema"] < df_ha["ema"].shift(bars)

        lookback = c.SLOPE_LOOKBACK
        ema_prev = df_ha["ema"].shift(lookback)
        df_ha["ema_change_pct"] = (df_ha["ema"] - ema_prev).abs() / ema_prev.abs() * 100
        df_ha["strong_trend"]   = df_ha["ema_change_pct"] >= c.MIN_SLOPE_PERCENT

        delta = df_ha["und_close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(c.RSI_LEN, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(c.RSI_LEN, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df_ha["rsi"] = 100 - (100 / (1 + rs))

        df_ha["ema9"] = df_ha["und_close"].ewm(span=c.EMA9_LEN, adjust=False).mean()
        n = c.EMA9_ANGLE_LOOKBACK
        ema9_prev = df_ha["ema9"].shift(n)
        slope = (df_ha["ema9"] - ema9_prev) / n
        slope_norm = slope / df_ha["ema9"].replace(0, np.nan)
        df_ha["ema9_angle_deg"] = np.degrees(np.arctan(slope_norm))

        return df_ha

    def build_htf(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df_15 = df_idx.resample(c.HTF_RULE).agg({
            "und_open": "first",
            "und_high": "max",
            "und_low": "min",
            "und_close": "last",
            "und_volume": "sum",
        }).dropna()

        df_15["htf_ema"] = df_15["und_close"].ewm(span=c.HTF_EMA_LEN, adjust=False).mean()
        df_15["htf_trend"] = np.where(
            df_15["und_close"] > df_15["htf_ema"], "up",
            np.where(df_15["und_close"] < df_15["htf_ema"], "down", "flat")
        )

        df_15 = df_15.reset_index()
        return df_15

    def _is_bullish_reversal_actual(self, df: pd.DataFrame, i: int) -> bool:
        cur = df.iloc[i]
        prev = df.iloc[i - 1]

        cond_dir = (prev["und_close"] < prev["und_open"]) and (cur["und_close"] > cur["und_open"])

        rng = cur["und_high"] - cur["und_low"]
        if rng <= 0:
            return False

        body = abs(cur["und_close"] - cur["und_open"])
        upper_wick = cur["und_high"] - max(cur["und_close"], cur["und_open"])

        cond_structure = (
            cur["und_low"] <= prev["und_low"] and
            upper_wick <= 0.5 * rng and
            body       >= 0.2 * rng
        )
        return cond_dir and cond_structure

    def _is_bearish_reversal_actual(self, df: pd.DataFrame, i: int) -> bool:
        cur = df.iloc[i]
        prev = df.iloc[i - 1]

        cond_dir = (prev["und_close"] > prev["und_open"]) and (cur["und_close"] < cur["und_open"])

        rng = cur["und_high"] - cur["und_low"]
        if rng <= 0:
            return False

        body = abs(cur["und_close"] - cur["und_open"])
        lower_wick = min(cur["und_close"], cur["und_open"]) - cur["und_low"]

        cond_structure = (
            cur["und_high"] >= prev["und_high"] and
            lower_wick <= 0.5 * rng and
            body       >= 0.2 * rng
        )
        return cond_dir and cond_structure

    # ---------- SIGNAL GENERATION ----------

    def generate_signals(self, df_index: pd.DataFrame) -> pd.DataFrame:
        if df_index.empty:
            return pd.DataFrame()

        df_idx_reset = df_index.reset_index()
        df_ha = self.calculate_indicators(df_idx_reset)
        df_15 = self.build_htf(df_index)

        df_ha = pd.merge_asof(
            df_ha.sort_values("timestamp"),
            df_15[["timestamp", "htf_trend"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        signals = []
        warmup = max(
            self.config.EMA_LEN,
            self.config.SLOPE_LOOKBACK + 1,
            self.config.RSI_LEN + 1,
            self.config.EMA9_ANGLE_LOOKBACK + 1,
            2,
        )

        for i in range(warmup, len(df_ha)):
            sig = self._check_signal(df_ha, i)
            if sig:
                signals.append(sig)

        return pd.DataFrame(signals) if signals else pd.DataFrame()

    def _create_signal(self, row: pd.Series, direction: str) -> Dict:
        und_price = row["und_close"]
        opt_symbol = self._pick_itm_option_symbol(underlying_price=und_price, direction=direction)

        signal = {
            "timestamp": row["timestamp"],
            "index_symbol": self.config.UNDERLYING_SYMBOL,
            "index_price": float(und_price),
            "direction": direction,
            "option_symbol": opt_symbol,
        }

        if direction == "BUY":
            self.position_size = 1
            self.last_trade = "long"
        else:
            self.position_size = -1
            self.last_trade = "short"

        return signal

    def _check_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        c = self.config
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        prev_body  = abs(prev["ha_close"] - prev["ha_open"])
        prev_range = prev["ha_high"] - prev["ha_low"]
        prev_is_doji = prev_range > 0 and (prev_body / prev_range) < c.DOJI_MAX_BODY_PCT

        fuzzy_match = abs(row["ha_open"] - prev["ha_close"]) / prev["ha_close"] < c.OPEN_TOLERANCE

        raw_long  = prev_is_doji and fuzzy_match and (row["ha_close"] > prev["ha_high"])
        raw_short = prev_is_doji and fuzzy_match and (row["ha_close"] < prev["ha_low"])

        ema = row["ema"]
        close = row["und_close"]
        ema_slope_up   = bool(row["ema_slope_up"])
        ema_slope_down = bool(row["ema_slope_down"])
        strong_trend   = bool(row["strong_trend"])

        ema_ok_long = (
            (not c.USE_EMA_FILTER or close > ema) and
            (not c.USE_EMA_SLOPE or ema_slope_up) and
            (not c.USE_SLOPE_STRENGTH or strong_trend)
        )
        ema_ok_short = (
            (not c.USE_EMA_FILTER or close < ema) and
            (not c.USE_EMA_SLOPE or ema_slope_down) and
            (not c.USE_SLOPE_STRENGTH or strong_trend)
        )

        rsi = row["rsi"]
        rsi_ok_long  = (not c.USE_RSI_FILTER) or (rsi < c.RSI_OVERSOLD)
        rsi_ok_short = (not c.USE_RSI_FILTER) or (rsi > c.RSI_OVERBOUGHT)

        long_allowed  = (self.position_size <= 0) and (self.last_trade != "long")
        short_allowed = (self.position_size >= 0) and (self.last_trade != "short")

        htf_trend = row.get("htf_trend", "flat")
        long_allowed  = long_allowed and (htf_trend == "up")
        short_allowed = short_allowed and (htf_trend == "down")

        long_cond  = raw_long and ema_ok_long and rsi_ok_long and long_allowed
        short_cond = raw_short and ema_ok_short and rsi_ok_short and short_allowed

        if long_cond:
            sig = self._create_signal(row, "BUY")
            sig["strategy_type"] = "doji_breakout"
            return sig

        if short_cond:
            sig = self._create_signal(row, "SELL")
            sig["strategy_type"] = "doji_breakout"
            return sig

        angle9 = row.get("ema9_angle_deg", np.nan)
        bullish_rev = self._is_bullish_reversal_actual(df, i)
        bearish_rev = self._is_bearish_reversal_actual(df, i)

        ema9_long_trend  = angle9 >= c.EMA9_MIN_ANGLE_DEG
        ema9_short_trend = angle9 <= -c.EMA9_MIN_ANGLE_DEG

        long2_allowed  = (self.position_size <= 0) and (self.last_trade != "long") and (htf_trend == "up")
        short2_allowed = (self.position_size >= 0) and (self.last_trade != "short") and (htf_trend == "down")

        long2_cond  = ema9_long_trend  and bullish_rev  and long2_allowed
        short2_cond = ema9_short_trend and bearish_rev and short2_allowed

        if long2_cond:
            sig = self._create_signal(row, "BUY")
            sig["strategy_type"] = "ema9_angle_reversal_actual"
            return sig

        if short2_cond:
            sig = self._create_signal(row, "SELL")
            sig["strategy_type"] = "ema9_angle_reversal_actual"
            return sig

        return None


# --------------------------- LIVE ENGINE ---------------------------

class LiveEngine:
    def __init__(self, bot: TradingBot, access_token: str):
        self.bot = bot
        self.access_token = access_token
        self.symbol = bot.config.UNDERLYING_SYMBOL

        self.candles = bot.get_index_data(days=bot.config.HIST_DAYS).reset_index()
        self.current_bar = None
        self.last_tick_time: Optional[datetime] = None
        self.logged_raw_once = False

        self.timeframe_mins = int(self.bot.config.TIMEFRAME)
        self.heartbeat_secs = int(self.timeframe_mins * 60 * 2)

        logger.info(
            f"Timeframe = {self.timeframe_mins}m, heartbeat threshold = {self.heartbeat_secs} sec"
        )

        self.ws: Optional[data_ws.FyersDataSocket] = None
        self.order_manager = OrderManager(self.bot.fyers, self.bot.config)

        # 1-bar confirmation state
        self.pending_signal: Optional[pd.Series] = None
        self.pending_bar_ts: Optional[datetime] = None

    def _bar_key(self, ts: datetime):
        minute_bucket = (ts.minute // self.timeframe_mins) * self.timeframe_mins
        return ts.replace(second=0, microsecond=0, minute=minute_bucket)

    def _on_open(self):
        logger.info("WS connected - subscribing to symbol")
        if self.ws is not None:
            self.ws.subscribe(symbols=[self.symbol], data_type="SymbolUpdate")

    def _on_close(self, msg):
        logger.warning(f"WS closed: {msg}")

    def _on_error(self, msg):
        logger.error(f"WS error: {msg}")

    def _on_message(self, msg):
        try:
            if not self.logged_raw_once:
                logger.info(f"RAW WS MSG (once): {msg}")
                self.logged_raw_once = True

            if isinstance(msg, list) and msg:
                data = msg[0]
            elif isinstance(msg, dict):
                data = msg.get("symbolData") or msg
            else:
                return

            if not isinstance(data, dict):
                return

            v_dict = data.get("v", {}) or {}
            ltp = (
                v_dict.get("lp")
                or data.get("ltp")
                or data.get("price")
                or data.get("close")
            )
            if ltp is None:
                logger.debug(f"No ltp/price/close in data: {data}")
                return

            now = datetime.now(IST)
            self.last_tick_time = now
            ts = now

            bar_time = self._bar_key(ts)
            ltp = float(ltp)
            vol = float(v_dict.get("volume") or data.get("volume", 0) or 0)

            # NEW BAR HANDLING WITH CONFIRMATION ENTRY
            if self.current_bar is None or self.current_bar["timestamp"] != bar_time:
                # finalize previous bar (signal detection / pending signal)
                if self.current_bar is not None:
                    self._finalize_bar(self.current_bar)

                # on first tick of new bar, execute pending signal (if any)
                if (
                    self.pending_signal is not None
                    and self.pending_bar_ts is not None
                    and bar_time > self.pending_bar_ts
                ):
                    sig = self.pending_signal.copy()
                    opt_symbol = sig["option_symbol"]
                    try:
                        quote_req = {"symbols": opt_symbol}
                        quote_resp = self.bot.fyers.quotes(data=quote_req)
                        opt_ltp = None
                        if isinstance(quote_resp, dict) and quote_resp.get("d"):
                            d0 = quote_resp["d"][0]
                            if isinstance(d0, dict):
                                v0 = d0.get("v", {})
                                opt_ltp = v0.get("lp")
                        if opt_ltp is None:
                            logger.warning(f"[CONFIRMATION] Option LTP None for {opt_symbol}, skip BO.")
                        else:
                            sig["option_entry_price"] = float(opt_ltp)
                            logger.info(
                                f"[CONFIRMATION] Executing pending signal on next bar open "
                                f"{bar_time}: {sig.to_dict()}"
                            )
                            order_id = self.order_manager.place_bo_for_signal(sig)
                            if order_id:
                                logger.info(
                                    f"[CONFIRMATION] BO order placed at next bar open, "
                                    f"order_id={order_id}"
                                )
                            else:
                                logger.warning("[CONFIRMATION] Failed to place BO for pending signal.")
                    except Exception as e:
                        logger.warning(f"[CONFIRMATION] Failed to execute pending signal: {e}")

                    # clear pending
                    self.pending_signal = None
                    self.pending_bar_ts = None

                # start new bar
                self.current_bar = {
                    "timestamp": bar_time,
                    "und_open": ltp,
                    "und_high": ltp,
                    "und_low":  ltp,
                    "und_close": ltp,
                    "und_volume": vol,
                }

            else:
                # same bar, update OHLCV
                self.current_bar["und_high"] = max(self.current_bar["und_high"], ltp)
                self.current_bar["und_low"]  = min(self.current_bar["und_low"],  ltp)
                self.current_bar["und_close"] = ltp
                self.current_bar["und_volume"] += vol

        except Exception as e:
            logger.error(f"WS on_message error: {e} | raw: {msg}")
            return

    def _finalize_bar(self, bar: dict):
        bar_ts = bar["timestamp"]
        today = datetime.now(IST).date()

        if bar_ts.date() != today:
            logger.info(f"Skipping bar from previous day: {bar_ts}")
            return

        if not (self.bot.config.SESSION_START <= bar_ts.time() <= self.bot.config.SESSION_END):
            logger.info(f"Skipping bar outside session: {bar_ts}")
            return

        logger.info(f"Finalized bar (completed candle): {bar}")
        self.candles = (
            pd.concat([self.candles, pd.DataFrame([bar])], ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        df_for_signals = self.candles.copy()
        df_for_signals.set_index("timestamp", inplace=True)

        signals = self.bot.generate_signals(df_for_signals)
        if signals.empty:
            logger.info("No signals from generate_signals on this completed bar.")
            return

        if isinstance(signals["timestamp"].iloc[0], pd.Timestamp):
            if signals["timestamp"].dt.tz is None:
                signals["timestamp"] = signals["timestamp"].dt.tz_localize(IST)
            else:
                signals["timestamp"] = signals["timestamp"].dt.tz_convert(IST)

        sig_on_bar = signals[signals["timestamp"] == bar_ts]
        if sig_on_bar.empty:
            logger.info(f"No signal with timestamp == completed bar ts {bar_ts}.")
            return

        last_sig = sig_on_bar.iloc[-1]

        # store for next bar open
        self.pending_signal = last_sig
        self.pending_bar_ts = bar_ts
        logger.info(
            f"CONFIRMATION MODE: stored pending signal from bar {bar_ts} "
            f"to execute at next bar open: {last_sig.to_dict()}"
        )

    def ist_time(self):
        """Get current IST time"""
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.now(ist)

    def _heartbeat_loop(self):
        while True:
            try:
                if self.last_tick_time is not None:
                    delta = (datetime.now(IST) - self.last_tick_time).total_seconds()
                    if delta > self.heartbeat_secs:
                        logger.warning(
                            f"No ticks for {delta:.0f}s (> {self.heartbeat_secs}s). Triggering WS reconnect."
                        )
                        try:
                            if self.ws is not None:
                                try:
                                    self.ws.close()
                                except Exception:
                                    pass
                            self._create_and_connect_ws()
                        except Exception as e:
                            logger.error(f"Reconnect failed: {e}")
                        self.last_tick_time = datetime.now(IST)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(5)

    def _create_and_connect_ws(self):
        logger.info("Creating new FyersDataSocket and connecting...")
        self.ws = data_ws.FyersDataSocket(
            access_token=self.access_token,
            log_path="",
            litemode=True,
            write_to_file=False,
            reconnect=True,
            on_connect=self._on_open,
            on_close=self._on_close,
            on_error=self._on_error,
            on_message=self._on_message,
        )
        self.ws.connect()

    def start(self):
        logger.info("Starting WebSocket live engine with heartbeat watchdog.")
        try:
            self._create_and_connect_ws()
            hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            hb_thread.start()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping engine.")
        finally:
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass

    
# # --------------------------- MAIN ---------------------------

# if __name__ == "__main__":
#     bot = TradingBot()
#     app_id = bot.app_id
#     raw_token = bot.access_token
#     if not app_id or not raw_token:
#         raise RuntimeError("app_id or token missing; check fyers_appid.txt / fyers_token.txt")

#     # v3 websocket token format: "client_id:access_token". [web:50]
#     ws_access_token = f"{app_id}:{raw_token}"
#     engine = LiveEngine(bot, access_token=ws_access_token)
#     engine.start()
