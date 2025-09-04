import warnings
from typing import Dict, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state


class TabularDataLoader:
    """
    Loader for common UCI/OpenML tabular benchmarks (numeric-only features).

    Datasets:
      - Concrete (Concrete Compressive Strength)                  [reg]
      - Wine Quality Red / White                                  [reg]
      - Naval Propulsion Plant                                    [reg]
      - Protein (CASP / Protein tertiary structure)               [reg]
      - YearPredictionMSD                                         [reg]
      - Power (Combined Cycle Power Plant)                        [reg]
      - Yacht (Yacht Hydrodynamics)                               [reg]
      - Boston (OpenML 'boston')                                  [reg]
      - Shifts (placeholder; requires external source)            [various]
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
    def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _finalize_Xy_numeric_auto(df: pd.DataFrame, target_candidates: Iterable[str]) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        """
        Choose target column from candidate list (fall back to last column if none match),
        keep only numeric features, drop rows with NaNs in numeric features.
        """
        tgt = TabularDataLoader._pick_first_present(df, target_candidates)
        if tgt is None:
            # fallback: use last column as target
            tgt = df.columns[-1]
            warnings.warn(f"[{df.attrs.get('name','dataset')}] None of {list(target_candidates)} found; using last column '{tgt}' as target.")
        y = df[tgt].copy()
        X = df.drop(columns=[tgt])

        X_num = X.select_dtypes(include=[np.number]).copy()
        keep = ~X_num.isna().any(axis=1)
        X_num = X_num.loc[keep]
        y = y.loc[keep]

        return X_num.to_numpy(), y.to_numpy(), list(X_num.columns), tgt

    @staticmethod
    def _labelize_binary(y: pd.Series):
        """Map string/categorical binary labels to {0,1} deterministically. If already numeric, return np.array."""
        if getattr(y, "dtype", None) is not None and y.dtype.kind in "biufc":
            return y.to_numpy()
        le = LabelEncoder()
        return le.fit_transform(pd.Series(y).astype(str))

    @staticmethod
    def _fetch_openml_any(name_candidates: Iterable[str], as_frame=True):
        """Try multiple dataset names on OpenML until one works."""
        err_last = None
        for nm in name_candidates:
            try:
                ds = fetch_openml(name=nm, as_frame=as_frame, cache=True, parser="auto")
                ds.frame.attrs["name"] = nm
                return ds
            except Exception as e:
                err_last = e
        raise RuntimeError(f"Could not fetch any of {list(name_candidates)} from OpenML. Last error: {err_last}")

    @staticmethod
    def _apply_row_cap(df: pd.DataFrame, row_cap: Optional[int], rng: np.random.Generator) -> pd.DataFrame:
        if row_cap is not None and len(df) > row_cap:
            idx = rng.choice(len(df), size=row_cap, replace=False)
            return df.iloc[idx].reset_index(drop=True)
        return df

    # ---------- dataset loaders ----------

    def _load_concrete(self, row_cap: Optional[int]) -> Dict:
        ds = self._fetch_openml_any([
            "Concrete Compressive Strength", "concrete"
        ], as_frame=True)
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df,
            target_candidates=[
                "Concrete compressive strength",  # common
                "csMPa", "ccs", "CompressiveStrength", "concrete_strength", "strength", ds.target_names[0] if ds.target_names else ""
            ]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_wine_quality(self, which: str, row_cap: Optional[int]) -> Dict:
        name = f"wine-quality-{which}"
        ds = fetch_openml(name=name, as_frame=True, cache=True, parser="auto")
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["quality"])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_naval(self, row_cap: Optional[int]) -> Dict:
        ds = self._fetch_openml_any([
            "naval-propulsion-plant", "naval-propulsion", "Naval Propulsion Plant"
        ], as_frame=True)
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        # Typical targets (two decay coefficients). Pick turbine by default if present.
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df,
            target_candidates=[
                "GT Turbine decay state coefficient",         # common
                "GT Compressor decay state coefficient",
                "turbine", "compressor", ds.target_names[0] if ds.target_names else ""
            ]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_protein(self, row_cap: Optional[int]) -> Dict:
        # Often published as CASP / “Physicochemical Properties of Protein Tertiary Structure”
        ds = self._fetch_openml_any([
            "CASP", "Physicochemical Properties of Protein Tertiary Structure", "protein"
        ], as_frame=True)
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df,
            target_candidates=["RMSD", ds.target_names[0] if ds.target_names else ""]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_year_msd(self, row_cap: Optional[int]) -> Dict:
        ds = fetch_openml(name="YearPredictionMSD", as_frame=True, cache=True, parser="auto")
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["year", ds.target_names[0] if ds.target_names else ""])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_power(self, row_cap: Optional[int]) -> Dict:
        # Combined Cycle Power Plant (CCPP)
        ds = self._fetch_openml_any([
            "Combined Cycle Power Plant", "CCPP", "power-plant"
        ], as_frame=True)
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df,
            target_candidates=["PE", "Net hourly electrical energy output (PE)", ds.target_names[0] if ds.target_names else ""]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_yacht(self, row_cap: Optional[int]) -> Dict:
        ds = self._fetch_openml_any([
            "yacht", "Yacht Hydrodynamics", "yacht_hydrodynamics"
        ], as_frame=True)
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df,
            target_candidates=[
                "residuary resistance", "Rr", ds.target_names[0] if ds.target_names else ""
            ]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_boston(self, row_cap: Optional[int]) -> Dict:
        # Research only; deprecated in sklearn, but available on OpenML.
        ds = self._fetch_openml_any(["boston", "Boston"], as_frame=True)
        df = ds.frame.copy()
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["MEDV", "medv", ds.target_names[0] if ds.target_names else ""])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_shifts_placeholder(self, row_cap: Optional[int]) -> Dict:
        raise RuntimeError(
            "The 'Shifts' benchmark is not hosted on OpenML/sklearn. "
            "It’s a separate suite (e.g., Malinin et al., NeurIPS Datasets & Benchmarks 2021). "
            "Please provide local CSVs or install its dedicated package and plug it here."
        )

    # ---------- public API ----------

    def load_tabular_data(
        self,
        include_concrete: bool = True,
        include_wine_red: bool = True,
        include_wine_white: bool = True,
        include_naval: bool = True,
        include_protein: bool = True,
        include_yearmsd: bool = True,
        include_power: bool = True,
        include_yacht: bool = True,
        include_boston: bool = True,
        include_shifts: bool = False,  # off by default; needs external source
        row_cap: Optional[int] = 100_000,
    ) -> Dict[str, Dict]:
        """
        Return: name -> {'X','y','feature_names','target_name'}
        - Numeric features only; rows with NaNs dropped.
        - If row_cap is set, subsample rows uniformly without replacement.
        """
        out: Dict[str, Dict] = {}

        def _add(name: str, loader_fn, *args, **kwargs):
            try:
                out[name] = loader_fn(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"[{name}] failed to load: {e}")

        if include_concrete:
            _add("Concrete", self._load_concrete, row_cap)
        if include_wine_red:
            _add("WineQualityRed", self._load_wine_quality, "red", row_cap)
        if include_wine_white:
            _add("WineQualityWhite", self._load_wine_quality, "white", row_cap)
        if include_naval:
            _add("Naval", self._load_naval, row_cap)
        if include_protein:
            _add("Protein", self._load_protein, row_cap)
        if include_yearmsd:
            _add("YearMSD", self._load_year_msd, row_cap)
        if include_power:
            _add("Power", self._load_power, row_cap)
        if include_yacht:
            _add("Yacht", self._load_yacht, row_cap)
        if include_boston:
            _add("Boston", self._load_boston, row_cap)
        if include_shifts:
            _add("Shifts", self._load_shifts_placeholder, row_cap)

        return out


# ------------- quick smoke test -------------
if __name__ == "__main__":
    from collections import Counter

    loader = TabularDataLoader(random_state=42)
    data_dict = loader.load_tabular_data(
        include_concrete=True,
        include_wine_red=True,
        include_wine_white=False,  # flip on for OOD swap experiments
        include_naval=True,
        include_protein=False,     # can be big; enable if you want
        include_yearmsd=False,     # large; enable if you want
        include_power=True,
        include_yacht=True,
        include_boston=True,
        include_shifts=False,
        row_cap=100_000,
    )

    def summarize_one(name: str, pack: dict):
        X, y = pack["X"], pack["y"]
        feats, tgt = pack["feature_names"], pack["target_name"]
        print(f"\n=== {name} ===")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"# numeric features: {len(feats)}  | target: {tgt}")
        print(f"NaNs in X: {np.isnan(X).sum()}  | NaNs in y: {np.isnan(y).sum() if np.issubdtype(y.dtype, np.floating) else 0}")
        # regression summary
        print(f"y stats -> mean: {float(np.mean(y)):.4f}, std: {float(np.std(y)):.4f}, "
              f"min: {float(np.min(y)):.4f}, max: {float(np.max(y)):.4f}")

    if not data_dict:
        print("No datasets loaded (check warnings above).")
    else:
        for ds_name, pack in data_dict.items():
            summarize_one(ds_name, pack)
        print("\nAll done ✔️")
