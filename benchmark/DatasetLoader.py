import sklearn
import numpy as np

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_openml,
    fetch_covtype,
    fetch_california_housing,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state


class TabularDataLoader:
    """
    Loader for standard tabular ML benchmarks with numeric-only feature selection.

    Datasets supported (toggles in load_tabular_data):
      - Adult / Census Income (OpenML: 'adult')
      - Covertype (sklearn.fetch_covtype)
      - HIGGS (OpenML: 'HIGGS' / 'Higgs')  [very large!]
      - California Housing (sklearn.fetch_california_housing)
      - Ames Housing (OpenML: 'house_prices')
      - Credit Card Default (OpenML: 'default-of-credit-card-clients')
      - YearPredictionMSD (OpenML: 'YearPredictionMSD')  [large]
      - (Optional) Wine Quality Red (OpenML: 'wine-quality-red') and White ('wine-quality-white')
    """

    def __init__(self, cache_dir: Optional[str] = None, random_state: int = 42):
        self.cache_dir = cache_dir
        self.random_state = check_random_state(random_state)

    # ---------- helpers ----------

    @staticmethod
    def _to_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only numeric columns (float/int); drop others."""
        return df.select_dtypes(include=[np.number])

    @staticmethod
    def _finalize_Xy_numeric(df: pd.DataFrame, target: str):
        """
        Keep only numeric feature columns; drop rows with NaN (after selection).
        Returns (X, y, feature_names).
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")
        y = df[target].copy()
        X = df.drop(columns=[target])

        X_num = X.select_dtypes(include=[np.number]).copy()
        # Align y to rows that remain after dropping any NaNs in numeric features
        keep = ~X_num.isna().any(axis=1)
        X_num = X_num.loc[keep]
        y = y.loc[keep]

        return X_num.to_numpy(), y.to_numpy(), list(X_num.columns)

    @staticmethod
    def _labelize_binary(y: pd.Series):
        """
        Map string/categorical binary labels to {0,1} deterministically.
        If already numeric, just return as np.array.
        """
        if y.dtype.kind in "biufc":
            return y.to_numpy()
        le = LabelEncoder()
        return le.fit_transform(y.astype(str))

    # ---------- dataset loaders ----------

    def _load_adult(self) -> Dict:
        """Adult / Census Income (binary classification)."""
        ds = fetch_openml(
            name="adult", version=2, as_frame=True, cache=True, parser="auto"
        )
        df = ds.frame.copy()
        target_col = "class" if "class" in df.columns else ds.target_names[0]
        # numeric-only feature subset
        X, y, feat = self._finalize_Xy_numeric(df, target_col)
        y = self._labelize_binary(pd.Series(y))
        return {"X": X, "y": y, "feature_names": feat, "target_name": target_col}

    def _load_covtype(self) -> Dict:
        """Covertype (multiclass classification)."""
        cov = fetch_covtype(as_frame=True)
        df = cov.frame.copy()
        target_col = "target"
        X, y, feat = self._finalize_Xy_numeric(df, target_col)
        return {"X": X, "y": y.astype(int), "feature_names": feat, "target_name": target_col}

    def _load_higgs(self, max_rows: Optional[int] = None) -> Dict:
        """HIGGS (binary classification) via OpenML. Extremely large (11M rows)."""
        # Try common names
        for name_try in ["HIGGS", "Higgs"]:
            try:
                ds = fetch_openml(name=name_try, as_frame=True, cache=True, parser="auto")
                break
            except Exception:
                ds = None
        if ds is None:
            raise RuntimeError("Could not fetch HIGGS from OpenML (tried 'HIGGS' and 'Higgs').")
        df = ds.frame.copy()
        # Identify target heuristically
        tgt = "class" if "class" in df.columns else "signal" if "signal" in df.columns else ds.target_names[0]
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=0)
        X, y, feat = self._finalize_Xy_numeric(df, tgt)
        y = self._labelize_binary(pd.Series(y))
        return {"X": X, "y": y, "feature_names": feat, "target_name": tgt}

    def _load_california(self) -> Dict:
        """California Housing (regression)."""
        ds = fetch_california_housing(as_frame=True)
        df = ds.frame.copy()
        target_col = "MedHouseVal"
        X, y, feat = self._finalize_Xy_numeric(df, target_col)
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": target_col}

    def _load_ames(self) -> Dict:
        """Ames Housing / House Prices (regression)."""
        ds = fetch_openml(name="house_prices", as_frame=True, cache=True, parser="auto")
        df = ds.frame.copy()
        # Common target is 'SalePrice'
        target_col = "SalePrice" if "SalePrice" in df.columns else ds.target_names[0]
        X, y, feat = self._finalize_Xy_numeric(df, target_col)
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": target_col}

    def _load_credit_default(self) -> Dict:
        """Default of Credit Card Clients (binary classification)."""
        ds = fetch_openml(
            name="default-of-credit-card-clients",
            as_frame=True,
            cache=True,
            parser="auto",
        )
        df = ds.frame.copy()
        # Targets in the wild: 'default.payment.next.month' or 'default payment next month'
        possible = [c for c in df.columns if "default" in c.lower()]
        if not possible:
            possible = [ds.target_names[0]]
        target_col = possible[0]
        X, y, feat = self._finalize_Xy_numeric(df, target_col)
        y = self._labelize_binary(pd.Series(y))
        return {"X": X, "y": y, "feature_names": feat, "target_name": target_col}

    def _load_year_msd(self, max_rows: Optional[int] = None) -> Dict:
        """YearPredictionMSD (regression)."""
        ds = fetch_openml(name="YearPredictionMSD", as_frame=True, cache=True, parser="auto")
        df = ds.frame.copy()
        # Target usually 'year'
        target_col = "year" if "year" in df.columns else ds.target_names[0]
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=0)
        X, y, feat = self._finalize_Xy_numeric(df, target_col)
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": target_col}

    def _load_wine_quality(self, which: str = "red") -> Dict:
        """Wine Quality (regression). which in {'red','white'}."""
        name = f"wine-quality-{which}"
        ds = fetch_openml(name=name, as_frame=True, cache=True, parser="auto")
        df = ds.frame.copy()
        target_col = "quality"
        X, y, feat = self._finalize_Xy_numeric(df, target_col)
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": target_col}

    # ---------- public API ----------

    def load_tabular_data(
        self,
        include_adult: bool = True,
        include_covtype: bool = True,
        include_higgs: bool = True,            # VERY large; consider setting to False or limit with higgs_max_rows
        include_california: bool = True,
        include_ames: bool = True,
        include_credit: bool = True,
        include_yearmsd: bool = True,          # Large; limit with year_msd_max_rows
        include_wine_red: bool = False,
        include_wine_white: bool = False,
        # controls for large datasets:
        higgs_max_rows: Optional[int] = None,  # e.g., 1_000_000
        year_msd_max_rows: Optional[int] = None,  # e.g., 200_000
    ) -> Dict[str, Dict]:
        """
        Fetch selected datasets and return a dict:
            name -> { 'X': np.ndarray, 'y': np.ndarray, 'feature_names': list[str], 'target_name': str }

        Notes
        -----
        - All non-numeric features are dropped by design (per your requirement).
        - Rows with NaNs remaining in numeric columns are dropped.
        - HIGGS / YearMSD are huge; pass *_max_rows to subsample server-side.
        """
        out: Dict[str, Dict] = {}

        def _add(name: str, loader_fn, *args, **kwargs):
            try:
                out[name] = loader_fn(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"[{name}] failed to load: {e}")

        if include_adult:
            _add("Adult", self._load_adult)

        if include_covtype:
            _add("Covertype", self._load_covtype)

        if include_higgs:
            _add("HIGGS", self._load_higgs, higgs_max_rows)

        if include_california:
            _add("CaliforniaHousing", self._load_california)

        if include_ames:
            _add("AmesHousing", self._load_ames)

        if include_credit:
            _add("CreditDefault", self._load_credit_default)

        if include_yearmsd:
            _add("YearPredictionMSD", self._load_year_msd, year_msd_max_rows)

        if include_wine_red:
            _add("WineQualityRed", self._load_wine_quality, "red")

        if include_wine_white:
            _add("WineQualityWhite", self._load_wine_quality, "white")

        return out
