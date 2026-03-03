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
import calendar
import time
import threading
import json
import sys

# Optional: if you use logger_config like in your live file
try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")


# --------------------------- EXPIRY HELPERS (WEEKLY & MONTHLY) ---------------------------

def get_next_tuesday(now_ist: Optional[datetime] = None) -> datetime:
    """
    NIFTY weekly expiry: next Tuesday.
    If today is Tuesday, use next week's Tuesday.
    """
    now_ist = now_ist or datetime.now(IST)
    if now_ist.weekday() < 1:  # Mon=0, Tue=1
        days_ahead = 1 - now_ist.weekday()
    else:
        days_ahead = 7 - (now_ist.weekday() - 1)
    return (now_ist + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def get_nifty_monthly_expiry(year: int, month: int) -> datetime:
    """
    NIFTY monthly expiry: last Tuesday of the month.
    """
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    while dt.weekday() != 1:  # Tuesday = 1
        dt -= timedelta(days=1)
    return dt


def get_current_or_next_monthly_expiry(now_ist: Optional[datetime] = None) -> datetime:
    now_ist = now_ist or datetime.now(IST)
    y, m = now_ist.year, now_ist.month
    exp_dt = get_nifty_monthly_expiry(y, m)
    if exp_dt.date() < now_ist.date():
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
        exp_dt = get_nifty_monthly_expiry(y, m)
    return exp_dt


def format_expiry_code(dt: datetime) -> str:
    yy = dt.year % 100
    mon = dt.strftime("%b").upper()
    return f"{yy:02d}{mon}"


# --------------------------- CONFIG ---------------------------

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME = "1"        # 1‑minute (same as backtest)
    HIST_DAYS = 60

    # Same as backtest
    STRIKE_STEP = 100
    NIFTY_LOT_SIZE = 65

    # Expiry control: "WEEKLY" (next Tue) or "MONTHLY" (last Tue)
    EXPIRY_TYPE = "WEEKLY"

    # SwiftTrend params (same as backtest)
    ST_PERIOD    = 100
    ST_BAND_MULT = 0.5
    ST_TOL_MULT  = 0.5

    # Higher timeframe trend filter (same as backtest)
    HTF_RULE    = "120min"
    HTF_EMA_LEN = 5

    SESSION_START = dtime(9, 20)
    SESSION_END   = dtime(15, 30)

    # For live trading window; you can tighten if needed
    TRADE_START = dtime(9, 30)
    TRADE_END   = dtime(15, 0)

    # SL/TP same as backtest
    STOP_LOSS_PCT = -7.0
    TAKE_PROFIT_PCT = 20.0

    # Tick size for index options
    OPTION_TICK_SIZE = 0.05

    # BO quantity
    BO_QTY = 65

    def __init__(self):
        now = datetime.now(IST)
        exp_type = self.EXPIRY_TYPE.upper()
        if exp_type == "WEEKLY":
            exp_dt = get_next_tuesday(now)
            kind = "WEEKLY"
        else:
            exp_dt = get_current_or_next_monthly_expiry(now)
            kind = "MONTHLY"
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logging.getLogger(__name__).info(
            f"Using NIFTY {kind} expiry {exp_dt.date()} with code {self.EXPIRY_CODE}"
        )


# --------------------------- SWIFTTREND CORE ---------------------------

class SwiftTrendParams:
    def __init__(self, period=100, band_multiplier=2.5, tolerance_multiplier=2.5):
        self.period = period
        self.band_multiplier = band_multiplier
        self.tolerance_multiplier = tolerance_multiplier


def compute_swift_trend(df: pd.DataFrame,
                        params: SwiftTrendParams,
                        open_col="open",
                        high_col="high",
                        low_col="low",
                        close_col="close") -> pd.DataFrame:
    o = df[open_col].values
    h = df[high_col].values
    l = df[low_col].values
    c = df[close_col].values
    n = len(df)

    period = params.period
    band_multiplier = params.band_multiplier
    tolerance_multiplier = params.tolerance_multiplier

    body = np.abs(o - c)
    avg_body = pd.Series(body).rolling(window=period, min_periods=period).mean().values
    body_mid = (o + c) / 2.0

    basic_upper = np.full(n, np.nan)
    basic_lower = np.full(n, np.nan)
    tolerance = np.full(n, np.nan)
    avg_line = np.full(n, np.nan)
    margin_line_base = np.full(n, np.nan)
    is_uptrend_arr = np.full(n, True, dtype=bool)
    trend_changed = np.full(n, False, dtype=bool)
    signal = [None] * n

    for i in range(n):
        if i > 0 and not np.isnan(avg_body[i - 1]):
            basic_upper[i] = body_mid[i - 1] + band_multiplier * avg_body[i - 1]
            basic_lower[i] = body_mid[i - 1] - band_multiplier * avg_body[i - 1]

        if not np.isnan(avg_body[i]):
            tolerance[i] = tolerance_multiplier * avg_body[i]

        prev_avg_line = avg_line[i - 1] if i > 0 else np.nan
        prev_is_uptrend = is_uptrend_arr[i - 1] if i > 0 else True

        if np.isnan(prev_avg_line):
            avg_line[i] = basic_lower[i]
            is_uptrend_arr[i] = True
        else:
            if prev_is_uptrend:
                avg_line[i] = np.nanmax([basic_lower[i], prev_avg_line]) if not np.isnan(basic_lower[i]) else prev_avg_line
            else:
                avg_line[i] = np.nanmin([basic_upper[i], prev_avg_line]) if not np.isnan(basic_upper[i]) else prev_avg_line
            is_uptrend_arr[i] = prev_is_uptrend

        if not np.isnan(avg_line[i]) and not np.isnan(tolerance[i]):
            margin_line_base[i] = avg_line[i] - tolerance[i] if is_uptrend_arr[i] else avg_line[i] + tolerance[i]

        trend_changed[i] = False
        if not np.isnan(margin_line_base[i]):
            if is_uptrend_arr[i] and c[i] < margin_line_base[i]:
                is_uptrend_arr[i] = False
                trend_changed[i] = True
                signal[i] = "Sell"
            elif (not is_uptrend_arr[i]) and c[i] > margin_line_base[i]:
                is_uptrend_arr[i] = True
                trend_changed[i] = True
                signal[i] = "Buy"

        if not trend_changed[i]:
            signal[i] = None

    out = df.copy()
    out["st_avg_line"] = avg_line
    out["st_margin"] = margin_line_base
    out["st_is_uptrend"] = is_uptrend_arr
    out["st_trend_changed"] = trend_changed
    out["st_signal"] = signal
    return out


# --------------------------- SWIFTTREND BOT (SIGNAL ENGINE) ---------------------------

class SwiftTrendBot:
    def __init__(self):
        self.config = TradingConfig()
        self.fyers = self._init_fyers()
        self.position_size: int = 0
        self.last_trade: Optional[str] = None

        # For WS token reuse
        self.app_id: Optional[str] = None
        self.access_token: Optional[str] = None
        if os.path.exists("fyers_appid.txt"):
            with open("fyers_appid.txt", "r") as f:
                self.app_id = f.read().strip()
        if os.path.exists("fyers_token.txt"):
            with open("fyers_token.txt", "r") as f:
                self.access_token = f.read().strip()

        logger.info("SwiftTrendBot initialized (signals for backtest + live).")

    def _init_fyers(self):
        if not all(os.path.exists(f) for f in ["fyers_appid.txt", "fyers_token.txt"]):
            raise FileNotFoundError("Missing fyers_appid.txt or fyers_token.txt")

        app_id = open("fyers_appid.txt", "r").read().strip()
        token = open("fyers_token.txt", "r").read().strip()
        if not app_id or not token:
            raise ValueError("Empty app_id or token in credential files")

        fyers_v3 = fyersModel.FyersModel(
            client_id=app_id,
            token=token,
            is_async=False,
            log_path="logs"
        )
        logger.info("Fyers V3 API initialized for SwiftTrendBot")
        return fyers_v3

    # -------- history helpers --------

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end_dt_ist = datetime.now(IST)
        start_dt_ist = end_dt_ist - timedelta(days=days)

        data = {
            "symbol": symbol,
            "resolution": self.config.TIMEFRAME,
            "date_format": "1",
            "range_from": start_dt_ist.strftime("%Y-%m-%d"),
            "range_to": end_dt_ist.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }
        resp = self.fyers.history(data=data)

        if isinstance(resp, dict):
            if resp.get("s") != "ok":
                logger.warning(f"API error for {symbol}: {resp}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            candles = resp.get("candles", [])
        else:
            candles = resp.get("candles", []) if hasattr(resp, "get") else []

        if not candles:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_index_data(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.HIST_DAYS
        df = self._fetch_history(self.config.UNDERLYING_SYMBOL, days)
        if df.empty:
            return df
        df.set_index("timestamp", inplace=True)
        session_mask = df.index.indexer_between_time(
            self.config.SESSION_START, self.config.SESSION_END,
            include_start=True, include_end=True
        )
        df = df.iloc[session_mask]
        logger.info(f"Loaded NIFTY50 index data with {len(df)} rows [IST] in session")
        return df

    # -------- higher timeframe trend filter --------

    def _build_htf_trend(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        """
        Build higher timeframe trend on index using HTF_RULE & HTF_EMA_LEN.
        """
        c = self.config
        df_htf = df_idx.resample(c.HTF_RULE).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        df_htf["htf_ema"] = df_htf["close"].ewm(span=c.HTF_EMA_LEN, adjust=False).mean()
        df_htf["htf_trend"] = np.where(
            df_htf["close"] > df_htf["htf_ema"], "up",
            np.where(df_htf["close"] < df_htf["htf_ema"], "down", "flat")
        )
        df_htf = df_htf.reset_index()
        return df_htf

    # -------- option helper --------

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

        # 5-digit formatting for symbol
        symbol = f"NSE:NIFTY{c.EXPIRY_CODE}{strike_int:05d}{opt_type}"
        logger.info(f"Option symbol: {symbol} (spot={spot:.2f}, dir={direction})")
        return symbol

    # -------- SwiftTrend signals on index + HTF trend filter --------

    def generate_swifttrend_signals(self, df_index: pd.DataFrame) -> pd.DataFrame:
        if df_index.empty:
            return pd.DataFrame()

        df = df_index.copy().reset_index()

        # Compute SwiftTrend on index
        st_params = SwiftTrendParams(
            period=self.config.ST_PERIOD,
            band_multiplier=self.config.ST_BAND_MULT,
            tolerance_multiplier=self.config.ST_TOL_MULT
        )
        df_st = compute_swift_trend(
            df,
            st_params,
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close"
        )

        # Build HTF trend (on original index series) and merge
        df_idx_for_htf = df_index.copy()
        df_htf = self._build_htf_trend(df_idx_for_htf)

        df_merged = pd.merge_asof(
            df_st.sort_values("timestamp"),
            df_htf[["timestamp", "htf_trend"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        signals = []
        warmup = self.config.ST_PERIOD  # just for stability

        for i in range(warmup, len(df_merged)):
            row = df_merged.iloc[i]
            prev_pos_ok_long = (self.position_size <= 0) and (self.last_trade != "long")
            prev_pos_ok_short = (self.position_size >= 0) and (self.last_trade != "short")

            ts: datetime = row["timestamp"]
            ts_time = ts.time()
            in_trade_window = (ts_time >= self.config.TRADE_START) and (ts_time <= self.config.TRADE_END)

            htf_trend = row.get("htf_trend", "flat")

            if bool(row["st_trend_changed"]) and in_trade_window:
                # Longs only when HTF trend is up; shorts only when HTF trend is down
                if row["st_is_uptrend"] and prev_pos_ok_long and (htf_trend == "up"):
                    sig = self._create_signal(row, "BUY")
                    sig["strategy_type"] = "swifttrend_flip_htf"
                    sig["htf_trend"] = htf_trend
                    signals.append(sig)
                elif (not row["st_is_uptrend"]) and prev_pos_ok_short and (htf_trend == "down"):
                    sig = self._create_signal(row, "SELL")
                    sig["strategy_type"] = "swifttrend_flip_htf"
                    sig["htf_trend"] = htf_trend
                    signals.append(sig)

        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
        logger.info(
            f"Generated {len(signals_df)} SwiftTrend index signals (with HTF trend filter)"
        )
        return signals_df

    def _create_signal(self, row: pd.Series, direction: str) -> Dict:
        und_price = row["close"]
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


# --------------------------- ORDER MANAGER (BO, SL/TP from % like backtest) ---------------------------

class OrderManager:
    def __init__(self, fyers_client: fyersModel.FyersModel, config: TradingConfig):
        self.fyers = fyers_client
        self.config = config

    @staticmethod
    def _round_to_tick(value: float, tick: float) -> float:
        return round(round(value / tick) * tick, 2)

    def _compute_bo_sl_tp_from_pct(self, opt_entry: float) -> tuple[float, float]:
        """
        Convert STOP_LOSS_PCT / TAKE_PROFIT_PCT into rupee points for BO legs.
        Note: STOP_LOSS_PCT is negative in config; we use absolute value for distance.
        """
        sl_pct = abs(self.config.STOP_LOSS_PCT)
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
            stopPrice  = option_entry_price
            limitPrice = option_entry_price + 2
        SL/TP: computed as % of option entry price, passed as points and tick-aligned.
        """
        symbol = signal_row.get("option_symbol")
        if not symbol:
            logger.warning(f"Signal has no option_symbol, skipping BO: {dict(signal_row)}")
            return None

        side = 1  # always buy option
        qty = qty or self.config.BO_QTY

        opt_entry = signal_row.get("option_entry_price")
        if opt_entry is None or (isinstance(opt_entry, float) and np.isnan(opt_entry)):
            logger.warning(
                "No option_entry_price in signal, cannot build stop‑limit entry: "
                f"{dict(signal_row)}"
            )
            return None

        opt_entry = float(opt_entry)

        if sl_points is None or tp_points is None:
            sl_calc, tp_calc = self._compute_bo_sl_tp_from_pct(opt_entry)
            sl_points = sl_points or sl_calc
            tp_points = tp_points or tp_calc

        tick = self.config.OPTION_TICK_SIZE

        sl_points   = self._round_to_tick(sl_points, tick)
        tp_points   = self._round_to_tick(tp_points, tick)
        stop_price  = self._round_to_tick(opt_entry, tick)
        limit_price = self._round_to_tick(opt_entry + 2.0, tick)

        data = {
            "symbol": symbol,
            "qty": int(qty),
            "type": 4,  # SL-L
            "side": side,
            "productType": "BO",
            "limitPrice": float(limit_price),
            "stopPrice": float(stop_price),
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": float(sl_points),
            "takeProfit": float(tp_points),
            "isSliceOrder": False,
        }

        try:
            logger.info(
                f"🚀 Placing SwiftTrend BO (SL‑L entry) for signal: {dict(signal_row)} | "
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


# --------------------------- LIVE ENGINE (SWIFTTREND) ---------------------------

class LiveEngine:
    def __init__(self, bot: SwiftTrendBot, access_token: str):
        self.bot = bot
        self.access_token = access_token
        self.symbol = bot.config.UNDERLYING_SYMBOL

        # Warmup history
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
                                f"[CONFIRMATION] Executing pending SwiftTrend signal on next bar open "
                                f"{bar_time}: {dict(sig)}"
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
                    "open": ltp,
                    "high": ltp,
                    "low":  ltp,
                    "close": ltp,
                    "volume": vol,
                }

            else:
                # same bar, update OHLCV
                self.current_bar["high"] = max(self.current_bar["high"], ltp)
                self.current_bar["low"]  = min(self.current_bar["low"],  ltp)
                self.current_bar["close"] = ltp
                self.current_bar["volume"] += vol

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

        logger.info(f"Finalized SwiftTrend bar (completed candle): {bar}")
        self.candles = (
            pd.concat([self.candles, pd.DataFrame([bar])], ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        df_for_signals = self.candles.copy()
        df_for_signals.set_index("timestamp", inplace=True)

        signals = self.bot.generate_swifttrend_signals(df_for_signals)
        
        if signals.empty:
            logger.info("No SwiftTrend signals on this completed bar.")
            return

        # Ensure timestamps are timezone-aware IST
        if isinstance(signals["timestamp"].iloc[0], pd.Timestamp):
            if signals["timestamp"].dt.tz is None:
                signals["timestamp"] = signals["timestamp"].dt.tz_localize(IST)
            else:
                signals["timestamp"] = signals["timestamp"].dt.tz_convert(IST)

        sig_on_bar = signals[signals["timestamp"] == bar_ts]
        if sig_on_bar.empty:
            logger.info(f"No SwiftTrend signal with timestamp == completed bar ts {bar_ts}.")
            return

        last_sig = sig_on_bar.iloc[-1]

        # store for next bar open
        self.pending_signal = last_sig
        self.pending_bar_ts = bar_ts
        logger.info(
            f"CONFIRMATION MODE: stored SwiftTrend pending signal from bar {bar_ts} "
            f"to execute at next bar open: {dict(last_sig)}"
        )

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
        logger.info("Creating new FyersDataSocket and connecting (SwiftTrend)...")
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
        logger.info("Starting SwiftTrend WebSocket live engine with heartbeat watchdog.")
        try:
            self._create_and_connect_ws()
            hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            hb_thread.start()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping SwiftTrend engine.")
        finally:
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot = SwiftTrendBot()
    app_id = bot.app_id
    raw_token = bot.access_token
    if not app_id or not raw_token:
        raise RuntimeError("app_id or token missing; check fyers_appid.txt / fyers_token.txt")

    # v3 websocket token format: "client_id:access_token"
    ws_access_token = f"{app_id}:{raw_token}"
    engine = LiveEngine(bot, access_token=ws_access_token)
    engine.start()
