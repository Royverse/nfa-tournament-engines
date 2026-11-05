import os
import pickle
import warnings
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from crunch_synth import (
    TrackerBase,
    SubTracker,
    FORECAST_PROFILES,
    SUPPORTED_ASSETS,
    pricedb,
    PriceStore,
    Asset,
    PriceData
)

# ============================================================
# Configurations & Constants
# ============================================================

@dataclass
class TrainingConfig:
    name: str
    assets: list                     # which assets are modeled
    horizons: list                   # which horizons are modeled
    exogenous_series: list           # which external signals are used

    # Data / feature parameters
    window_days: int
    resolution: int   # in seconds
    window_size: int  # in number of points

    # CV parameters
    n_splits: int
    early_stopping_rounds: int

    # LightGBM parameters
    model_params: dict

    def required_history_days(self) -> int:
        return int(self.window_size * self.resolution / 86400) + 1


class Constants:
    CRYPTO_CONFIG = TrainingConfig(
        name="crypto",
        assets=["BTC", "SOL", "ETH", "XRP", "HYPE"],
        horizons=[86400, 3600],
        exogenous_series=["BTC"],
        window_days=50,
        resolution=300,
        window_size=288,
        n_splits=5,
        early_stopping_rounds=100,
        model_params={
            "objective": "regression",
            "learning_rate": 0.04,
            "n_estimators": 350,
            "max_depth": 6,
            "num_leaves": 24,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.15,
            "reg_lambda": 0.5,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1
        }
    )

    STOCK_CONFIG = TrainingConfig(
        name="stock",
        assets=["SP500", "NVDAX", "TSLAX", "AAPLX", "GOOGLX", "SPCX"],
        horizons=[86400, 3600],
        exogenous_series=["SP500"],
        window_days=50,
        resolution=300,
        window_size=288,
        n_splits=5,
        early_stopping_rounds=100,
        model_params={
            "objective": "regression",
            "learning_rate": 0.04,
            "n_estimators": 350,
            "max_depth": 6,
            "num_leaves": 24,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.15,
            "reg_lambda": 0.5,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1
        }
    )

    METAL_CONFIG = TrainingConfig(
        name="metal",
        assets=["XAUT"],
        horizons=[86400, 3600],
        exogenous_series=["BTC", "SP500"],
        window_days=50,
        resolution=300,
        window_size=288,
        n_splits=5,
        early_stopping_rounds=100,
        model_params={
            "objective": "regression",
            "learning_rate": 0.04,
            "n_estimators": 500,
            "max_depth": 4,
            "num_leaves": 16,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.15,
            "reg_lambda": 0.5,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1
        }
    )

    COMMODITIES_CONFIG = TrainingConfig(
        name="commodities",
        assets=["WTIOIL"],
        horizons=[86400, 3600],
        exogenous_series=["SP500"],
        window_days=50,
        resolution=300,
        window_size=288,
        n_splits=5,
        early_stopping_rounds=100,
        model_params={
            "objective": "regression",
            "learning_rate": 0.04,
            "n_estimators": 500,
            "max_depth": 4,
            "num_leaves": 16,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.15,
            "reg_lambda": 0.5,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1
        }
    )


EQUITY_ASSETS = {"SP500", "NVDAX", "TSLAX", "AAPLX", "GOOGLX", "SPCX"}
COMMODITY_METAL_ASSETS = {"XAUT", "WTIOIL"}


def compute_seasonality_multiplier(asset: str, ts: pd.Timestamp, empirical_profile: dict = None) -> float:
    """Compute time-of-week volatility multiplier for future step timestamp `ts` (UTC).

    Captures the structural 3x-8x volatility difference between active US trading
    sessions (13:30-20:00 UTC Mon-Fri), overnight sessions, and weekends for equities
    and commodities, blended with empirical asset-specific seasonality when available.
    """
    weekday = int(ts.weekday())
    hour_float = float(ts.hour) + float(ts.minute) / 60.0
    is_weekend = weekday >= 5

    if asset in EQUITY_ASSETS:
        if is_weekend:
            prior = 0.22
        elif 13.5 <= hour_float < 20.0:
            # US regular market hours (9:30 - 16:00 ET)
            prior = 1.62
        elif 12.0 <= hour_float < 13.5 or 20.0 <= hour_float < 21.0:
            # Pre-open & post-close active auction windows
            prior = 0.92
        elif 8.0 <= hour_float < 12.0:
            # European overlap / pre-market
            prior = 0.58
        else:
            # Overnight session
            prior = 0.38
    elif asset in COMMODITY_METAL_ASSETS:
        if is_weekend:
            prior = 0.28
        elif 12.5 <= hour_float < 20.0:
            # London / NY commodity session overlap
            prior = 1.32
        elif 7.0 <= hour_float < 12.5:
            prior = 0.96
        else:
            prior = 0.72
    else:
        # 24/7 Crypto assets
        if is_weekend:
            prior = 0.86
        elif 13.0 <= hour_float < 21.0:
            prior = 1.10
        else:
            prior = 0.96

    if empirical_profile:
        bucket_key = ("weekend" if is_weekend else "weekday", int(ts.hour))
        emp_mult = empirical_profile.get(bucket_key)
        if emp_mult is not None and np.isfinite(emp_mult) and emp_mult > 0:
            return float(np.clip(0.55 * emp_mult + 0.45 * prior, 0.15, 2.50))

    return float(prior)


def learn_empirical_seasonality(df: pd.DataFrame) -> dict:
    """Learn empirical (weekday/weekend, hour_utc) volatility ratios from historical 5m bars."""
    if df is None or len(df) < 60:
        return {}
    prices = df["price"].values
    returns = np.abs(np.diff(prices))
    if len(returns) == 0:
        return {}
    global_mean = float(np.mean(returns))
    if global_mean <= 1e-9:
        return {}

    ts_series = pd.to_datetime(df["timestamp"].values[1:])
    profile = {}
    for is_wknd in (False, True):
        label = "weekend" if is_wknd else "weekday"
        wk_mask = (ts_series.weekday >= 5) if is_wknd else (ts_series.weekday < 5)
        for hr in range(24):
            mask = wk_mask & (ts_series.hour == hr)
            if np.sum(mask) >= 8:
                bucket_vol = float(np.mean(returns[mask]))
                profile[(label, hr)] = float(np.clip(bucket_vol / global_mean, 0.15, 2.50))
    return profile


def build_step_mixture_components(step_scale: float, step_drift: float = 0.0) -> list:
    """Construct a 2-component variance-preserving leptokurtic Gaussian mixture.

    Weights (0.78, 0.22) and scale multipliers (0.62, 1.84) satisfy:
      0.78 * (0.62^2) + 0.22 * (1.84^2) = 0.2998 + 0.7448 = 1.0446 ~ 1.0
    This captures both tight diffusive cores and fat jump tails under CRPS while
    staying strictly below MAX_DISTRIBUTION_COMPONENTS = 3.
    """
    safe_scale = max(1e-6, float(step_scale))
    core_scale = max(1e-6, 0.62 * safe_scale)
    tail_scale = max(1e-6, 1.84 * safe_scale)
    loc_val = float(step_drift) if np.isfinite(step_drift) else 0.0
    return [
        {
            "density": {
                "type": "builtin",
                "name": "norm",
                "params": {
                    "loc": loc_val,
                    "scale": core_scale,
                },
            },
            "weight": 0.78,
        },
        {
            "density": {
                "type": "builtin",
                "name": "norm",
                "params": {
                    "loc": loc_val,
                    "scale": tail_scale,
                },
            },
            "weight": 0.22,
        },
    ]


# ============================================================
# Feature Extraction & Model Training
# ============================================================

def compute_feat(past, timestamp, compute_time_feat=True):
    def summary_features(x, is_constant, suffix=""):
        mean_ = np.mean(x)
        std_ = np.std(x)
        skew = ((x - mean_)**3).mean() / (std_**3 + 1e-9) if not is_constant else 0.0
        kurtosis = ((x - mean_)**4).mean() / (std_**4 + 1e-9) if not is_constant else 0.0
        values = [
            mean_, std_, np.median(x), np.min(x), np.max(x),
            skew, kurtosis, x[-1], x[0],
        ]
        names = [
            f"mean{suffix}", f"std{suffix}", f"median{suffix}", f"min{suffix}", f"max{suffix}",
            f"skew{suffix}", f"kurt{suffix}", f"last{suffix}", f"first{suffix}",
        ]
        return values, names

    def trend_features(x, is_constant, suffix=""):
        t = np.arange(len(x))
        slope = np.polyfit(t, x, 1)[0] if not is_constant else 0.0
        rv_6 = np.std(x[-6:]) if len(x) >= 6 else np.std(x)
        rv_12 = np.std(x[-12:]) if len(x) >= 12 else np.std(x)
        short_std = np.std(x[-25:])
        med_std = np.std(x[-72:]) if len(x) >= 72 else np.std(x)
        long_std = np.std(x[-100:])
        rv_ratio_short = rv_12 / (short_std + 1e-9) if not is_constant else 1.0
        rv_ratio_long = short_std / (long_std + 1e-9) if not is_constant else 1.0
        jump_ratio = np.max(np.abs(x[-25:])) / (short_std + 1e-9) if not is_constant else 0.0
        downside_std = np.std(np.minimum(x[-72:], 0.0)) if not is_constant else 0.0
        ac1 = np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 and not is_constant else 0.0
        if not np.isfinite(ac1):
            ac1 = 0.0
        energy = np.sum(x**2)
        abs_sum = np.sum(np.abs(x))
        iqr = np.percentile(x, 90) - np.percentile(x, 10)
        values = [
            slope, rv_6, rv_12, short_std, med_std, long_std,
            rv_ratio_short, rv_ratio_long, jump_ratio, downside_std,
            ac1, energy, abs_sum, iqr
        ]
        names = [
            f"slope{suffix}", f"rv6{suffix}", f"rv12{suffix}", f"short_std{suffix}",
            f"med_std{suffix}", f"long_std{suffix}", f"rv_ratio_short{suffix}",
            f"rv_ratio_long{suffix}", f"jump_ratio{suffix}", f"downside_std{suffix}",
            f"ac1{suffix}", f"energy{suffix}", f"abs_sum{suffix}", f"iqr{suffix}",
        ]
        return values, names

    feats = []
    names = []
    is_constant = len(past) == 0 or np.allclose(past, past[0])

    if compute_time_feat:
        hour_val = timestamp.hour + timestamp.minute / 60.0
        weekday_val = timestamp.weekday()
        is_us_open = 1.0 if (weekday_val < 5 and 13.5 <= hour_val < 20.0) else 0.0
        is_weekend = 1.0 if weekday_val >= 5 else 0.0
        feats += [
            hour_val,
            weekday_val,
            np.sin(2.0 * np.pi * hour_val / 24.0),
            np.cos(2.0 * np.pi * hour_val / 24.0),
            is_us_open,
            is_weekend,
        ]
        names += [
            "hour_of_day",
            "day_of_week",
            "hour_sin",
            "hour_cos",
            "is_us_open",
            "is_weekend",
        ]

    v, n = summary_features(past, is_constant)
    feats += v
    names += n

    v, n = trend_features(past, is_constant)
    feats += v
    names += n

    return feats, names


def build_feature_dataframe(pairs_prices, dict_pairs_exogenous_series=None, resolution=300):
    df = pd.DataFrame(pairs_prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    resolution_minute = int(resolution // 60)
    df = (
        df.resample(f"{resolution_minute}min", on="timestamp")
        .first()
        .reset_index()
        .sort_values("timestamp")
    )

    if dict_pairs_exogenous_series is not None:
        for series_name, pairs_exogenous in dict_pairs_exogenous_series.items():
            if not pairs_exogenous or len(pairs_exogenous) == 0:
                continue
            df_exogenous = pd.DataFrame(
                pairs_exogenous, columns=["timestamp", f"value_{series_name}"]
            )
            df_exogenous["timestamp"] = pd.to_datetime(df_exogenous["timestamp"], unit="s")
            df_exogenous = (
                df_exogenous.resample(f"{resolution_minute}min", on="timestamp")
                .first()
                .reset_index()
                .sort_values("timestamp")
            )
            df = pd.merge_asof(
                df,
                df_exogenous,
                on="timestamp",
                direction="backward"
            )

    df = df.dropna()
    if len(df) == 0:
        raise ValueError("Empty dataset after alignment")
    return df


def build_training_data(pairs_prices, dict_pairs_exogenous_series=None, resolution=300, window_size=288, horizon_size=288):
    df = build_feature_dataframe(pairs_prices, dict_pairs_exogenous_series, resolution)
    prices = df["price"].values
    timestamps = df["timestamp"].values
    returns = np.diff(prices)

    exogenous_returns = {
        col: np.diff(df[col].values)
        for col in df.columns if col.startswith("value_")
    }

    X = []
    y = []

    for t in range(window_size, len(df) - horizon_size):
        feats, feature_names = compute_feat(
            returns[t-window_size:t],
            pd.Timestamp(timestamps[t])
        )

        for col in sorted(exogenous_returns.keys()):
            r = exogenous_returns[col]
            feats_exogenous, feature_names_exogenous = compute_feat(
                r[t-window_size:t],
                pd.Timestamp(timestamps[t]),
                compute_time_feat=False
            )
            feats = np.concatenate([feats, feats_exogenous])
            feature_names = feature_names + [
                f"{col}_{n}" for n in feature_names_exogenous
            ]

        future_returns = returns[t:t+horizon_size]
        target = np.std(future_returns)

        X.append(feats)
        y.append(target)

    return np.array(X), np.array(y), feature_names


def train_model(horizon, asset, exogenous_series, histories, config, verbose=False):
    warnings.filterwarnings("ignore", category=FutureWarning)
    pairs_prices = histories[asset]
    dict_pairs_exogenous_series = {
        series_name: histories[series_name]
        for series_name in exogenous_series
        if series_name != asset and series_name in histories and histories[series_name] and len(histories[series_name]) > 0
    }

    horizon_size = horizon // config.resolution
    X, y, feature_names = build_training_data(
        pairs_prices, dict_pairs_exogenous_series,
        resolution=config.resolution, window_size=config.window_size,
        horizon_size=horizon_size
    )

    if len(X) == 0:
        raise ValueError(f"Insufficient samples to train model for {asset}")

    best_iterations = []
    kf = TimeSeriesSplit(n_splits=config.n_splits)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        model = LGBMRegressor(**config.model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="mae",
            callbacks=[
                early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )
        if getattr(model, "best_iteration_", None):
            best_iterations.append(int(model.best_iteration_))

    final_params = dict(config.model_params)
    if best_iterations:
        final_params["n_estimators"] = max(50, int(np.median(best_iterations)))

    final_model = LGBMRegressor(**final_params)
    final_model.fit(X, y)
    return final_model


# ============================================================
# API Price Client for External Feeds
# ============================================================

class APIPriceDbClient:
    """Inert client: external unauthenticated Pyth Hermes feeds were discontinued.
    All asset series are delivered natively through CrunchDAO pricedb."""
    def __init__(self):
        self._history_cache: dict = {}

    def get_price_history(self, *args, **kwargs) -> list:
        return []

    def get_price_history_chunked(self, *args, **kwargs) -> list:
        return []

    def get_last_price(self, *args, **kwargs) -> tuple:
        return (int(datetime.now(tz=timezone.utc).timestamp()), 0.0)

    def clear_cache(self):
        self._history_cache.clear()


# ============================================================
# SubTracker & Volatility Forecaster
# ============================================================

class MySubTracker(SubTracker):
    def __init__(
        self,
        config,
        end_date_training=None,
        filter_assets=None,
        specific_horizon=None,
        exogenous_db=None
    ):
        self.config = config
        self.fitted = False
        self.scales = {}
        self.seasonality_profiles = {}
        if filter_assets is not None:
            self.tracker_assets = [asset for asset in self.config.assets if asset in filter_assets]
        else:
            self.tracker_assets = list(self.config.assets)
        self.end_date_training = end_date_training if end_date_training is not None else datetime.now(tz=timezone.utc)
        self.is_backtest = True if end_date_training is not None else False
        self.specific_horizon = specific_horizon
        self.exogenous_pricedb = exogenous_db or APIPriceDbClient()

        self.data_initialization()
        if not self.is_backtest:
            self._load_existing_models()

    def data_initialization(self):
        self.prices = PriceStore(window_days=self.config.window_days)
        from_ = self.end_date_training - timedelta(days=self.config.window_days)
        to = self.end_date_training

        initial_histories = {}
        assets_to_fetch = list(dict.fromkeys(
            list(self.tracker_assets) + (list(self.config.exogenous_series) if len(self.tracker_assets) > 0 else [])
        ))
        for asset in assets_to_fetch:
            try:
                hist = pricedb.get_price_history(asset=asset, from_=from_, to=to)
                if hist:
                    initial_histories[asset] = hist
            except Exception as e:
                print(f"[Warning] Failed to fetch initial history for {asset}: {e}")

        if len(self.tracker_assets) > 0 and self.config.exogenous_series:
            for series_name in self.config.exogenous_series:
                if series_name in initial_histories and initial_histories[series_name]:
                    continue
                try:
                    ext_hist = self.exogenous_pricedb.get_price_history_chunked(
                        asset=series_name, from_=from_, to=to, resolution="5minute", chunk=timedelta(days=10)
                    )
                    if ext_hist:
                        initial_histories[series_name] = ext_hist
                except Exception as e:
                    print(f"[Warning] Failed to fetch exogenous history for {series_name}: {e}")

        self.prices.add_bulk(initial_histories)

    def tick(self, data: PriceData):
        self.prices.add_bulk(data)
        if self.config.exogenous_series and not self.is_backtest:
            for series_name in self.config.exogenous_series:
                if series_name not in data:
                    try:
                        last_price = self.exogenous_pricedb.get_last_price(asset=series_name)
                        if last_price and last_price[1] > 0:
                            self.prices.add_bulk({series_name: [last_price]})
                    except Exception:
                        pass

    def _model_path(self, horizon: int, asset: str) -> str:
        os.makedirs("resources", exist_ok=True)
        return f"resources/model_{horizon}_{asset}.pkl"

    def _save_model(self, model, horizon: int, asset: str):
        path = self._model_path(horizon, asset)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def _load_model(self, horizon: int, asset: str):
        path = self._model_path(horizon, asset)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[Model Load Warning] Failed to load {path}: {e}")
        return None

    def _load_existing_models(self):
        self.models = {}
        for horizon in self.config.horizons:
            self.models[horizon] = {}
            for asset in self.tracker_assets:
                model = self._load_model(horizon, asset)
                if model is not None:
                    self.models[horizon][asset] = model

    def train(self):
        if len(self.tracker_assets) == 0:
            return

        new_models = {horizon: {} for horizon in self.config.horizons}
        initial_histories = {asset: self.prices.get_prices(asset, days=self.config.window_days) for asset in self.tracker_assets}
        for series_name in self.config.exogenous_series:
            initial_histories[series_name] = self.prices.get_prices(series_name, days=self.config.window_days)

        for asset in self.tracker_assets:
            try:
                if initial_histories.get(asset):
                    df_emp = build_feature_dataframe(initial_histories[asset], None, self.config.resolution)
                    self.seasonality_profiles[asset] = learn_empirical_seasonality(df_emp)
            except Exception:
                pass

        for horizon in new_models.keys():
            if self.specific_horizon and horizon != self.specific_horizon:
                continue

            for asset in self.tracker_assets:
                try:
                    model = train_model(horizon, asset, self.config.exogenous_series, initial_histories, self.config)
                    new_models[horizon][asset] = model
                    self._save_model(model, horizon, asset)
                except Exception as e:
                    print(f"[Train Error] {asset} @ {horizon}: {e}")
                    if hasattr(self, 'models') and horizon in self.models and asset in self.models[horizon]:
                        new_models[horizon][asset] = self.models[horizon][asset]

        self.models = new_models
        self.fitted = True

    def predict(self, asset: str, horizon: int, step: int):
        spot_scale = None
        short_drift = 0.0
        current_time = pd.Timestamp(datetime.now(tz=timezone.utc))

        try:
            pairs = self.prices.get_prices(asset, days=self.config.required_history_days())
            dict_pairs_exogenous_series = {
                series_name: self.prices.get_prices(
                    series_name, days=self.config.required_history_days()
                )
                for series_name in self.config.exogenous_series
                if series_name != asset
            }
            dict_pairs_exogenous_series = {
                k: v for k, v in dict_pairs_exogenous_series.items() if v and len(v) > 0
            }

            if not pairs:
                return []

            df = build_feature_dataframe(pairs, dict_pairs_exogenous_series, self.config.resolution)
            prices = df["price"].values
            timestamps = df["timestamp"].values
            returns = np.diff(prices)
            past = returns[-self.config.window_size:]
            current_time = pd.Timestamp(timestamps[-1])

            if asset not in self.seasonality_profiles and len(df) >= 100:
                self.seasonality_profiles[asset] = learn_empirical_seasonality(df)

            # Spot short-term realized volatility (last 1 hour = 12 x 5m bars)
            recent_bars = past[-12:] if len(past) >= 12 else past
            raw_spot_std = float(np.std(recent_bars)) if len(recent_bars) > 1 else float(np.std(past))
            current_seasonality = compute_seasonality_multiplier(
                asset, current_time, self.seasonality_profiles.get(asset)
            )
            spot_scale = raw_spot_std / max(0.35, current_seasonality)

            # Dampened short-horizon momentum drift (bounded to <= 8% of 1-step vol)
            if len(recent_bars) >= 6:
                raw_mean = float(np.mean(recent_bars[-6:]))
                short_drift = float(np.clip(0.15 * raw_mean, -0.08 * raw_spot_std, 0.08 * raw_spot_std))

            feats, feature_names = compute_feat(past, current_time)

            for col in sorted(c for c in df.columns if c.startswith("value_")):
                r = np.diff(df[col].values)
                feats_exogenous, feature_names_exogenous = compute_feat(
                    r[-self.config.window_size:], current_time, compute_time_feat=False
                )
                feats = np.concatenate([feats, feats_exogenous])
                feature_names = feature_names + [
                    f"{col}_{n}" for n in feature_names_exogenous
                ]

            X = pd.DataFrame([feats], columns=feature_names)

            if not hasattr(self, 'models') or horizon not in self.models or asset not in self.models[horizon]:
                raise ValueError(f"Missing model horizon {horizon}, asset {asset}")

            model_obj = self.models[horizon][asset]
            expected_features = getattr(model_obj, "n_features_in_", len(feature_names))
            if expected_features != X.shape[1]:
                raise ValueError(f"Feature count mismatch ({X.shape[1]} vs {expected_features}); triggering adaptive fallback")

            scale = float(model_obj.predict(X)[0])
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.std(past))
            self.scales[asset] = scale

        except Exception:
            pairs = self.prices.get_prices(asset, days=10, resolution=300)
            if not pairs:
                return []
            ts_list, past_prices = zip(*pairs)
            current_time = pd.Timestamp(datetime.fromtimestamp(ts_list[-1], tz=timezone.utc))
            returns = np.diff(past_prices)
            scale = float(np.std(returns)) if len(returns) > 1 else 0.01
            recent_bars = returns[-12:] if len(returns) >= 12 else returns
            raw_spot_std = float(np.std(recent_bars)) if len(recent_bars) > 1 else scale
            current_seasonality = compute_seasonality_multiplier(
                asset, current_time, self.seasonality_profiles.get(asset)
            )
            spot_scale = raw_spot_std / max(0.35, current_seasonality)

        if spot_scale is None or not np.isfinite(spot_scale) or spot_scale <= 0:
            spot_scale = scale

        num_segments = horizon // step
        distributions = []
        base_step_factor = float(np.sqrt(step / self.config.resolution))
        emp_profile = self.seasonality_profiles.get(asset)

        for k in range(1, num_segments + 1):
            elapsed_seconds = k * step
            step_ts = current_time + pd.Timedelta(seconds=elapsed_seconds)

            # HAR-RV exponential mean-reversion from de-seasonalized spot vol to horizon forecast
            w_spot = float(np.exp(-elapsed_seconds / 10800.0))
            base_5m_vol = w_spot * spot_scale + (1.0 - w_spot) * scale

            # Apply intraday market-hours & weekend seasonality multiplier at future timestamp step_ts
            season_mult = compute_seasonality_multiplier(asset, step_ts, emp_profile)
            # Blend seasonality more strongly on 24h horizon where steps span sessions
            effective_season_mult = (
                0.70 * season_mult + 0.30 * 1.0
                if horizon >= 86400
                else 0.45 * season_mult + 0.55 * 1.0
            )

            step_scale = max(1e-6, base_step_factor * base_5m_vol * effective_season_mult)
            step_drift = short_drift * base_step_factor * float(np.exp(-elapsed_seconds / 3600.0))

            distributions.append({
                "step": elapsed_seconds,
                "type": "mixture",
                "components": build_step_mixture_components(step_scale, step_drift),
            })
        return distributions


# ============================================================
# Main Challenge Submission Tracker
# ============================================================

class LightGBMVolatilityTracker(TrackerBase):
    def __init__(self, end_date_training=None, filter_assets=None, specific_horizon=None, exogenous_db=None):
        async_mode = True if end_date_training is None else False
        data_clock = datetime.now(tz=timezone.utc) if end_date_training is None else end_date_training
        super().__init__(data_clock=data_clock, async_mode=async_mode)

        self.crypto_tracker = MySubTracker(Constants.CRYPTO_CONFIG, end_date_training, filter_assets, specific_horizon, exogenous_db)
        self.stock_tracker = MySubTracker(Constants.STOCK_CONFIG, end_date_training, filter_assets, specific_horizon, exogenous_db)
        self.metal_tracker = MySubTracker(Constants.METAL_CONFIG, end_date_training, filter_assets, specific_horizon, exogenous_db)
        self.commodities_tracker = MySubTracker(Constants.COMMODITIES_CONFIG, end_date_training, filter_assets, specific_horizon, exogenous_db)

        HOUR = 3600
        self.track(24 * HOUR, self.crypto_tracker.tracker_assets, self.crypto_tracker)
        self.track(1 * HOUR, self.crypto_tracker.tracker_assets, self.crypto_tracker)

        self.track(24 * HOUR, self.stock_tracker.tracker_assets, self.stock_tracker)
        self.track(1 * HOUR, self.stock_tracker.tracker_assets, self.stock_tracker)

        self.track(24 * HOUR, self.metal_tracker.tracker_assets, self.metal_tracker)
        self.track(1 * HOUR, self.metal_tracker.tracker_assets, self.metal_tracker)

        self.track(24 * HOUR, self.commodities_tracker.tracker_assets, self.commodities_tracker)
        self.track(1 * HOUR, self.commodities_tracker.tracker_assets, self.commodities_tracker)

        self.schedule('Crypto train', self.crypto_tracker.train, timedelta(days=5), immediate=True)
        self.schedule('Stock train', self.stock_tracker.train, timedelta(days=5.1), immediate=True)
        self.schedule('Metal train', self.metal_tracker.train, timedelta(days=5.2), immediate=True)
        self.schedule('Commodities train', self.commodities_tracker.train, timedelta(days=5.3), immediate=True)

    def _fallback_predict(self, asset: str, horizon: int, step: int):
        now = datetime.now(timezone.utc)
        current_time = pd.Timestamp(now)
        try:
            pairs = pricedb.get_price_history(asset=asset, from_=now - timedelta(days=5), to=now)
            if pairs and len(pairs) > 1:
                _, past_prices = zip(*pairs)
                returns = np.diff(past_prices)
                scale = float(np.std(returns)) if len(returns) > 1 else 0.01
            else:
                scale = 0.01
        except Exception:
            scale = 0.01

        num_segments = horizon // step
        distributions = []
        base_step_factor = float(np.sqrt(step / 300.0))
        for k in range(1, num_segments + 1):
            elapsed_seconds = k * step
            step_ts = current_time + pd.Timedelta(seconds=elapsed_seconds)
            season_mult = compute_seasonality_multiplier(asset, step_ts)
            step_scale = max(1e-6, base_step_factor * scale * (0.65 * season_mult + 0.35))
            distributions.append({
                "step": elapsed_seconds,
                "type": "mixture",
                "components": build_step_mixture_components(step_scale, 0.0),
            })
        return distributions

    def predict(self, asset: str, horizon: int, step: int):
        tracker = self._resolve_tracker(horizon, asset)
        if tracker is not None:
            return tracker.predict(asset, horizon, step)
        return self._fallback_predict(asset, horizon, step)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        print("Testing LightGBMVolatilityTracker fallback predict...")
        tracker = LightGBMVolatilityTracker(filter_assets=[])
        print("Tracker instantiated:", tracker)
        pred_spcx = tracker.predict("SPCX", 86400, 3600)
        print("SPCX fallback prediction verified:", len(pred_spcx))
    else:
        print("Testing LightGBMVolatilityTracker instantiation...")
        tracker = LightGBMVolatilityTracker()
        print("Tracker successfully instantiated:", tracker)

