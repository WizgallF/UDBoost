import warnings
from typing import Dict, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
import io, zipfile, requests
from urllib.parse import quote
try:
    from ucimlrepo import fetch_ucirepo
except Exception:
    fetch_ucirepo = None

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

    @staticmethod
    def _uci_df_by_id(uci_id: int) -> pd.DataFrame:
        ds = fetch_ucirepo(id=uci_id)  # e.g., 294 = CCPP
        df = getattr(ds.data, "original", None)
        if df is None or df.empty:
            parts = []
            for attr in ("ids", "features", "targets"):
                part = getattr(ds.data, attr, None)
                if part is not None and not part.empty:
                    parts.append(part)
            df = pd.concat(parts, axis=1)
        df.attrs["name"] = ds.metadata.name
        return df

    @staticmethod
    def _uci_df_by_name(name: str) -> pd.DataFrame:
        ds = fetch_ucirepo(name=name)
        df = getattr(ds.data, "original", None)
        if df is None or df.empty:
            parts = []
            for attr in ("ids", "features", "targets"):
                part = getattr(ds.data, attr, None)
                if part is not None and not part.empty:
                    parts.append(part)
            df = pd.concat(parts, axis=1)
        df.attrs["name"] = ds.metadata.name
        return df
    # ---------- dataset loaders ----------

    def _load_concrete(self, row_cap: Optional[int]) -> Dict:
        # UCI: "Concrete Compressive Strength" (Excel .xls on UCI)
        # Requires `xlrd` for .xls parsing.
        df = self._uci_df_by_name("Concrete Compressive Strength")  # stable UCI name
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df, target_candidates=[
                "Concrete compressive strength", "csMPa", "CompressiveStrength"
            ]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_wine_quality(self, which: str, row_cap: Optional[int]) -> Dict:
        base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
        url = base + ( "winequality-red.csv" if which == "red" else "winequality-white.csv" )
        df = pd.read_csv(url, sep=";")
        df.attrs["name"] = f"WineQuality-{which.capitalize()} (UCI)"
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["quality"])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_naval(self, row_cap: Optional[int]) -> Dict:
        # Prefer ucimlrepo if your version supports this dataset
        if fetch_ucirepo is not None:
            try:
                ds = fetch_ucirepo(id=316)  # UCI: Condition Based Maintenance of Naval Propulsion Plants
                X = ds.data.features.copy()
                y_df = ds.data.targets.copy()

                # Prefer turbine, then compressor
                preferred = [
                    "GT Turbine decay state coefficient",
                    "GT Compressor decay state coefficient",
                ]
                y_col = next((c for c in preferred if c in y_df.columns), None)
                if y_col is None:
                    # fallback to metadata or first target
                    meta_tgt = getattr(ds.metadata, "target_col", None)
                    if isinstance(meta_tgt, (list, tuple)) and meta_tgt:
                        y_col = meta_tgt[0]
                    elif isinstance(meta_tgt, str):
                        y_col = meta_tgt
                    else:
                        y_col = y_df.columns[-1]

                # numeric-only features and joint NA filter
                X_num = X.select_dtypes(include=[np.number]).copy()
                y_series = y_df[y_col].astype(float)

                df_all = pd.concat([X_num, y_series.rename(y_col)], axis=1)

                # optional subsample prior to NA drop
                if row_cap is not None and len(df_all) > row_cap:
                    idx = np.random.default_rng(0).choice(len(df_all), size=row_cap, replace=False)
                    df_all = df_all.iloc[idx].reset_index(drop=True)

                keep = ~df_all.drop(columns=[y_col]).isna().any(axis=1) & ~df_all[y_col].isna()
                df_all = df_all.loc[keep]

                X_out = df_all.drop(columns=[y_col]).to_numpy()
                y_out = df_all[y_col].to_numpy(dtype=float)
                feat_names = list(df_all.drop(columns=[y_col]).columns)
                return {"X": X_out, "y": y_out, "feature_names": feat_names, "target_name": y_col}
            except Exception as e:
                # Falls through to ZIP loader if this dataset isn't import-enabled yet.
                if "not available for import" not in str(e).lower():
                    raise

        # --- Fallback: download official UCI ZIP and parse data.txt ---
        import io, zipfile, requests
        zip_url = "https://archive.ics.uci.edu/static/public/316/condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip"
        r = requests.get(zip_url, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            member = next((n for n in zf.namelist() if n.lower().endswith("data.txt")), None)
            if member is None:
                raise RuntimeError("Naval zip: data.txt not found in archive")
            with zf.open(member) as fh:
                df = pd.read_csv(fh, sep=r"\s+", header=None, engine="python")

        # 16 GT sensor features + 2 decay coefficients (targets) as per UCI page
        cols = [
            "lp","v","GTT","GTn","GGn","Ts","Tp","T48","T1","T2","P48","P1","P2","Pexh","TIC","mf",
            "GT Compressor decay state coefficient","GT Turbine decay state coefficient"
        ]
        if df.shape[1] == len(cols):
            df.columns = cols
        df.attrs["name"] = "Naval CBM (UCI)"

        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))

        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df,
            target_candidates=[
                "GT Turbine decay state coefficient",
                "GT Compressor decay state coefficient",
            ],
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_protein(self, row_cap: Optional[int]) -> Dict:
        # UCI CASP.csv
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
        df = pd.read_csv(url)
        df.attrs["name"] = "CASP / Protein (UCI)"
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["RMSD"])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_year_msd(self, row_cap: Optional[int]) -> Dict:
        # Download once and read from zip in-memory (UCI provides a zip)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
        r = requests.get(url, timeout=120); r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open("YearPredictionMSD.txt") as fh:
                df = pd.read_csv(fh, header=None)
        # name the columns: first is year (target), 90 features
        df.columns = ["year"] + [f"x{i}" for i in range(1, df.shape[1])]
        df.attrs["name"] = "YearPredictionMSD (UCI)"
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["year"])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_power(self, row_cap: Optional[int]) -> Dict:
        # UCI ID: 294 (Combined Cycle Power Plant)
        df = self._uci_df_by_id(294)
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df, target_candidates=["PE", "Net hourly electrical energy output (PE)"]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_yacht(self, row_cap: Optional[int]) -> Dict:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
        df = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
        df.columns = [
            "LongPos_COB","Prismatic_Coeff","Length_Displacement_Ratio",
            "Beam_Draught_Ratio","Length_Beam_Ratio","Froude_Number",
            "Residuary_resistance"
        ]
        df.attrs["name"] = "Yacht (UCI)"
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df, target_candidates=["Residuary_resistance","residuary resistance","Rr"]
        )
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
        if include_shifts:
            _add("Shifts", self._load_shifts_placeholder, row_cap)

        return out


# ------------- quick smoke test -------------
if __name__ == "__main__":
    from collections import Counter

    loader = TabularDataLoader(random_state=42)
    data_dict = loader.load_tabular_data(
        include_concrete=False,
        include_wine_red=False,
        include_wine_white=False,  # flip on for OOD swap experiments
        include_naval=True,
        include_protein=False,     # can be big; enable if you want
        include_yearmsd=False,     # large; enable if you want
        include_power=False,
        include_yacht=False,
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
