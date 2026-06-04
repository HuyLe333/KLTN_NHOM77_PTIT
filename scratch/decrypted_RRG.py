import pandas as pd
import numpy as np
from typing import Union, List, Optional
from datetime import datetime, timedelta
from ..core.Aggregates import GetBarData 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RRG:
    """
    Relative Rotation Graph (RRG) implementation.
    Calculates RS-Ratio, RS-Momentum, and plots the results.
    """

    def __init__(
        self,
        access_token: callable,
        tickers: Optional[Union[str, List[str]]] = None,
        benchmark: Optional[str] = None,
        by: str = "1d",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        period: Optional[int] = None,
    ):
        """
        Initialize RRG instance with default parameters for fetching and filtering.
        """
        self.access_token = access_token
        self.tickers = [tickers] if isinstance(tickers, str) else tickers
        self.benchmark = benchmark
        self.by = by
        self.from_date = from_date
        self.to_date = to_date
        self.period = period
        self.data: pd.DataFrame = pd.DataFrame()
        self.timeframe_to_date_config = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "2h": timedelta(hours=2),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1)
        }
        self.timeframe_from_date_config = {
            "1m": timedelta(minutes=33),
            "5m": timedelta(minutes=33*5),
            "15m": timedelta(minutes=33*15),
            "30m": timedelta(minutes=33*30),
            "1h": timedelta(hours=33),
            "2h": timedelta(hours=33*2),
            "4h": timedelta(hours=33*4),
            "1d": timedelta(days=33)
        }
        self.timeframe_strftime = {
            "1m": "%Y-%m-%d %H:%M",
            "5m": "%Y-%m-%d %H:%M",
            "15m": "%Y-%m-%d %H:%M",
            "30m": "%Y-%m-%d %H:%M",
            "1h": "%Y-%m-%d %H:00",
            "2h": "%Y-%m-%d %H:00",
            "4h": "%Y-%m-%d %H:00",
            "1d": "%Y-%m-%d"
        }

    # ==============================================================
    # Core Calculation
    # ==============================================================

    @staticmethod
    def rrg_calculation(close: pd.Series, benchmark_close: pd.Series) -> pd.DataFrame:
        close_to_benchmark = close / benchmark_close
        rs = 100 * close_to_benchmark.rolling(window=12).mean() / close_to_benchmark.rolling(window=26).mean()
        rm = 100 * rs / rs.rolling(window=9).mean()

        df = pd.DataFrame({"rs": rs, "rm": rm}).dropna()
        conds = [
            (df["rs"] > 100) & (df["rm"] > 100),
            (df["rs"] > 100) & (df["rm"] < 100),
            (df["rs"] < 100) & (df["rm"] < 100),
            (df["rs"] < 100) & (df["rm"] > 100),
        ]
        labels = ["Leading", "Weakening", "Lagging", "Improving"]
        df["phase"] = np.select(conds, labels, default="").astype("object")
        return df

    # ==============================================================
    # Main Compute
    # ==============================================================

    def get(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        benchmark: Optional[str] = None,
        by: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        period: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute RRG values for one or multiple tickers vs benchmark and store the result.
        Falls back to default values if parameters are not provided.
        """
        tickers_list = [tickers] if isinstance(tickers, str) else (
            tickers or self.tickers
        )
        benchmark = benchmark or self.benchmark
        by = by or self.by
        from_date = from_date or self.from_date
        to_date = to_date or self.to_date

        if self.period:
            period = self.period + 33
            
        if not tickers_list or not benchmark:
            raise ValueError("tickers and benchmark are required for fetching RRG data.")

        # --- Fetch data ---
        try:
            ticker_data = GetBarData(
                access_token=self.access_token(),
                tickers=tickers_list,
                fields=["close"],
                adjusted=True,
                period=period,
                by=by,
                from_date=from_date,
                to_date=to_date,
            ).get().to_dataFrame()

            benchmark_data = GetBarData(
                access_token=self.access_token(),
                tickers=benchmark,
                fields=["close"],
                adjusted=True,
                period=period,
                by=by,
                from_date=from_date,
                to_date=to_date,
            ).get().to_dataFrame()
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {e}")

        benchmark_data_length = len(benchmark_data)
        def parse_from_date(date_str):
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Invalid date format: {date_str}")
        
        if from_date:
            from_date_benchmark = from_date
            while len(benchmark_data) < benchmark_data_length + 33:
                to_date_benchmark = (parse_from_date(from_date_benchmark) - self.timeframe_to_date_config[self.by]).strftime(self.timeframe_strftime[self.by])
                from_date_benchmark = (parse_from_date(to_date_benchmark) - self.timeframe_from_date_config[self.by]).strftime(self.timeframe_strftime[self.by])
                sub_benchmark_data = GetBarData(
                    access_token=self.access_token(),
                    tickers=benchmark,
                    fields=["close"],
                    adjusted=True,
                    period=period,
                    by=by,
                    from_date=from_date_benchmark,
                    to_date=to_date_benchmark,
                ).get().to_dataFrame()
                benchmark_data = pd.concat([benchmark_data, sub_benchmark_data], ignore_index=False)
        benchmark_data["timestamp_dt"] = pd.to_datetime(benchmark_data["timestamp"])
        benchmark_data = benchmark_data[benchmark_data["ticker"] == benchmark].sort_values(by="timestamp_dt").reset_index(drop=True)
        frames = []
        for ticker in tickers_list:
            df = ticker_data[ticker_data["ticker"] == ticker].reset_index(drop=True)
            ticker_data_length = len(df)
            if from_date:
                from_date_ticker = from_date
                while len(df) < ticker_data_length + 33:
                    to_date_ticker = (parse_from_date(from_date_ticker) - self.timeframe_to_date_config[self.by]).strftime(self.timeframe_strftime[self.by])
                    from_date_ticker = (parse_from_date(to_date_ticker) - self.timeframe_from_date_config[self.by]).strftime(self.timeframe_strftime[self.by])
                    sub_ticker_data = GetBarData(
                        access_token=self.access_token(),
                        tickers=ticker,
                        fields=["close"],
                        adjusted=True,
                        period=period,
                        by=by,
                        from_date=from_date_ticker,
                        to_date=to_date_ticker,
                    ).get().to_dataFrame()
                    df = pd.concat([df, sub_ticker_data], ignore_index=False).reset_index(drop=True)
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by="timestamp_dt").reset_index(drop=True)
            merged = pd.merge(df, benchmark_data, on="timestamp", suffixes=("", "_bm"), how="inner")

            if not merged.empty:
                rrg_df = self.rrg_calculation(merged["close"], merged["close_bm"])
                merged = pd.merge(merged, rrg_df, left_index=True, right_index=True, how="inner")
                if period:
                    merged = merged.tail(period).reset_index(drop=True)
                if from_date:
                    merged = merged.tail(min(ticker_data_length, benchmark_data_length)).reset_index(drop=True)
                frames.append(merged[["ticker", "timestamp", "close", "rs", "rm", "phase"]])

        self.data = pd.concat(frames, ignore_index=True).dropna()
        if not self.data.empty:
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])

        return self.data


    def filter(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        date: Optional[str] = None,
        phase: Optional[str] = None,
        rs_min: Optional[float] = None,
        rs_max: Optional[float] = None,
        rm_min: Optional[float] = None,
        rm_max: Optional[float] = None,
        changed_only: bool = False,
    ) -> pd.DataFrame:
        """
        Filter RRG data by tickers, date, phase, RS/RM thresholds.
        If no data available, automatically calls get() using defaults.
        """
        if self.data.empty:
            self.get()

        df = self.data.copy()
        cond = pd.Series(True, index=df.index)

        # --- filter by tickers ---
        if tickers is not None:
            if isinstance(tickers, str):
                tickers = [tickers]
            cond &= df["ticker"].isin(tickers)

        # --- filter by date ---
        if date:
            target_date = pd.to_datetime(date).normalize()
            df["date_only"] = df["timestamp"].dt.normalize()
            cond &= df["date_only"] == target_date
            df = df.loc[cond].drop(columns=["date_only"])
            cond = pd.Series(True, index=df.index)

        # --- filter changed only ---
        if changed_only and not df.empty:
            df_sorted = df.sort_values(["ticker", "timestamp"])
            df_sorted["prev_phase"] = df_sorted.groupby("ticker")["phase"].shift(1)
            change_cond = df_sorted["phase"] != df_sorted["prev_phase"]
            df = df_sorted.loc[change_cond].drop(columns=["prev_phase"])
            cond = pd.Series(True, index=df.index)

        # --- phase / rs / rm filters ---
        if phase:
            cond &= df["phase"].str.lower() == phase.lower()
        if rs_min is not None:
            cond &= df["rs"] >= rs_min
        if rs_max is not None:
            cond &= df["rs"] <= rs_max
        if rm_min is not None:
            cond &= df["rm"] >= rm_min
        if rm_max is not None:
            cond &= df["rm"] <= rm_max

        return df.loc[cond].reset_index(drop=True)


    # ==========================
    # Optional: Plot

    def plot(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        latest_only: bool = False,
        figsize: tuple = (12, 8),
        trail_length: int = 100,
    ) -> None:
        """
        Plot Relative Rotation Graph (RRG) for specified tickers.
        Automatically fetches data if not already available.
        
        Args:
            tickers: Ticker(s) to plot (str or list of str). Uses default if None.
            latest_only: If True, plots only the latest point for each ticker.
            figsize: Figure size as (width, height).
            trail_length: Number of points to include in the trail if latest_only=False.
        """
        # Automatically fetch data if self.data is empty
        if self.data.empty:
            self.get(
                tickers=self.tickers,
                benchmark=self.benchmark,
                by=self.by,
                from_date=self.from_date,
                to_date=self.to_date,
                period=self.period
            )

        df = self.data.copy()
        if df.empty:
            print("No data to plot after fetching. Check tickers, benchmark, or data availability.")
            return

        if tickers is not None:
            if isinstance(tickers, str):
                tickers = [tickers]
            df = df[df["ticker"].isin(tickers)]
            if df.empty:
                print(f"No data found for tickers: {tickers}")
                return

        rs_min, rs_max = df["rs"].min(), df["rs"].max()
        rm_min, rm_max = df["rm"].min(), df["rm"].max()

        pad = max(rs_max - rs_min, rm_max - rm_min) * 0.05
        x_min, x_max = rs_min - pad, rs_max + pad
        y_min, y_max = rm_min - pad, rm_max + pad

        center = 100
        min_range = 10.0
        x_min = min(x_min, center - min_range)
        x_max = max(x_max, center + min_range)
        y_min = min(y_min, center - min_range)
        y_max = max(y_max, center + min_range)

        span = max(abs(x_max - center), abs(center - x_min),
                abs(y_max - center), abs(center - y_min))
        x_min, x_max = center - span, center + span
        y_min, y_max = center - span, center + span

        _, ax = plt.subplots(figsize=figsize)

        ax.add_patch(patches.Rectangle((x_min, 100), 100 - x_min, y_max - 100, color="#004c6d", alpha=0.35))
        ax.add_patch(patches.Rectangle((100, 100), x_max - 100, y_max - 100, color="#006d3c", alpha=0.35))
        ax.add_patch(patches.Rectangle((x_min, y_min), 100 - x_min, 100 - y_min, color="#6a040f", alpha=0.35))
        ax.add_patch(patches.Rectangle((100, y_min), x_max - 100, 100 - y_min, color="#996515", alpha=0.35))

        ax.text(center + span / 2, center + span / 2, "TĂNG GIÁ\n(Leading)", color="lime", fontsize=10,
                ha="center", va="center", weight="bold")
        ax.text(center - span / 2, center + span / 2, "TÍCH LŨY\n(Improving)", color="cyan", fontsize=10,
                ha="center", va="center", weight="bold")
        ax.text(center - span / 2, center - span / 2, "GIẢM GIÁ\n(Lagging)", color="red", fontsize=10,
                ha="center", va="center", weight="bold")
        ax.text(center + span / 2, center - span / 2, "SUY YẾU\n(Weakening)", color="yellow", fontsize=10,
                ha="center", va="center", weight="bold")

        ax.axhline(100, color="white", lw=1.2)
        ax.axvline(100, color="white", lw=1.2)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_facecolor("#0d1b2a")

        ax.set_xlabel("RS-Ratio (Relative Strength)", fontsize=11)
        ax.set_ylabel("RS-Momentum", fontsize=11)
        ax.set_title("Relative Rotation Graph (RRG)", fontsize=14, weight="bold")

        cmap = plt.cm.get_cmap("tab10")
        df_sorted = df.sort_values("timestamp")

        for i, (ticker, group) in enumerate(df_sorted.groupby("ticker")):
            color = cmap(i % 10)
            if not latest_only:
                trail = group.tail(trail_length)
                ax.plot(trail["rs"], trail["rm"], color=color, lw=1.8, alpha=0.9, zorder=2)

                if len(trail) >= 2:
                    x1, y1 = trail["rs"].iloc[-2], trail["rm"].iloc[-2]
                    x2, y2 = trail["rs"].iloc[-1], trail["rm"].iloc[-1]
                    ax.arrow(x1, y1, x2 - x1, y2 - y1,
                            color=color, width=0.02, head_width=0.3,
                            length_includes_head=True, zorder=5)
                else:
                    ax.scatter(trail["rs"].iloc[-1], trail["rm"].iloc[-1],
                            color=color, edgecolor="black", s=60, zorder=5)
            else:
                last = group.iloc[-1]
                ax.scatter(last["rs"], last["rm"],
                        color=color, edgecolor="black", s=70, zorder=5)

            last_point = group.iloc[-1]
            ax.text(last_point["rs"] + span * 0.005, last_point["rm"],
                    ticker, fontsize=9, color=color, weight="bold")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()
