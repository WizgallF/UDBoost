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
    """Utility to load common UCI/OpenML tabular benchmarks (numeric-only features).

    This class provides small, self-contained loader methods for several
    widely-used regression datasets. Each loader returns a dict with standardized
    keys: ``{"X", "y", "feature_names", "target_name"}``.

    Attributes:
        cache_dir: Optional path for caching (currently unused; reserved).
        random_state: Random state used for reproducible subsampling.

    Datasets covered (see individual loader docstrings):
        - Concrete — compressive strength from mix proportions and age. [reg]
        - Wine Quality (red/white) — physico-chemical tests → quality score. [reg]
        - Naval — gas turbine sensors → component decay coefficients. [reg]
        - Protein (CASP) — 9 physico-chemical features → RMSD. [reg]
        - YearPredictionMSD — audio features → release year. [reg]
        - Power (CCPP) — ambient variables → net electrical output. [reg]
        - Yacht — hull/velocity params → residuary resistance. [reg]
        - Shifts — placeholder; requires external source.

    Notes:
        - Features are restricted to numeric columns. Rows with NaNs in features
          are dropped by default.
        - When a dataset has multiple plausible targets, the loader prefers a
          canonical column name and falls back deterministically.

    """

    def __init__(self, cache_dir: Optional[str] = None, random_state: int = 42):
        """Initialize the loader.

        Args:
            cache_dir: Optional path to a local cache directory. Reserved for
                future use; loaders here fetch directly from UCI or OpenML.
            random_state: Seed for reproducible subsampling.
        """
        self.cache_dir = cache_dir
        self.random_state = check_random_state(random_state)

    # ---------- helpers ----------

    @staticmethod
    def _to_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy keeping only numeric columns.

        Args:
            df: Input dataframe.

        Returns:
            A dataframe containing only numeric (int/float) columns.
        """
        return df.select_dtypes(include=[np.number])

    @staticmethod
    def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        """Pick the first column present in ``df`` from a list of candidates.

        Args:
            df: Input dataframe.
            candidates: Ordered iterable of column names to try.

        Returns:
            The first matching column name, or ``None`` if none are present.
        """
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _finalize_Xy_numeric_auto(
        df: pd.DataFrame,
        target_candidates: Iterable[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        """Split dataframe into numeric-only ``X`` and a chosen target ``y``.

        Chooses the target column from ``target_candidates`` (first match wins);
        if none match, uses the **last** column and emits a warning. Non-numeric
        feature columns are dropped. Rows with any NaNs in features are removed.

        Args:
            df: Source dataframe containing features and target.
            target_candidates: Ordered names to try for the target column.

        Returns:
            Tuple ``(X, y, feature_names, target_name)`` where:
                - ``X``: 2D numpy array of numeric features.
                - ``y``: 1D numpy array (target).
                - ``feature_names``: List of kept numeric feature column names.
                - ``target_name``: The selected target column name.

        Raises:
            ValueError: If no numeric feature columns remain after filtering.
        """
        tgt = TabularDataLoader._pick_first_present(df, target_candidates)
        if tgt is None:
            tgt = df.columns[-1]
            warnings.warn(
                f"[{df.attrs.get('name','dataset')}] None of {list(target_candidates)} "
                f"found; using last column '{tgt}' as target."
            )
        y = df[tgt].copy()
        X = df.drop(columns=[tgt])

        X_num = X.select_dtypes(include=[np.number]).copy()
        if X_num.shape[1] == 0:
            raise ValueError("No numeric feature columns remain after filtering.")
        keep = ~X_num.isna().any(axis=1)
        X_num = X_num.loc[keep]
        y = y.loc[keep]

        return X_num.to_numpy(), y.to_numpy(), list(X_num.columns), tgt

    @staticmethod
    def _labelize_binary(y: pd.Series):
        """Convert a binary label vector to {0,1} deterministically.

        If ``y`` is already numeric, it is returned as a NumPy array unchanged.
        Otherwise, labels are stringified and encoded via ``LabelEncoder``.

        Args:
            y: 1D pandas Series of labels.

        Returns:
            A 1D NumPy array of labels in {0, 1}.
        """
        if getattr(y, "dtype", None) is not None and y.dtype.kind in "biufc":
            return y.to_numpy()
        le = LabelEncoder()
        return le.fit_transform(pd.Series(y).astype(str))

    @staticmethod
    def _fetch_openml_any(name_candidates: Iterable[str], as_frame=True):
        """Fetch an OpenML dataset by trying several names.

        Attempts each candidate name until one succeeds.

        Args:
            name_candidates: Ordered dataset names to try on OpenML.
            as_frame: Whether to return a pandas DataFrame.

        Returns:
            The OpenML ``Bunch`` object.

        Raises:
            RuntimeError: If none of the provided names can be fetched.
        """
        err_last = None
        for nm in name_candidates:
            try:
                ds = fetch_openml(name=nm, as_frame=as_frame, cache=True, parser="auto")
                ds.frame.attrs["name"] = nm
                return ds
            except Exception as e:
                err_last = e
        raise RuntimeError(
            f"Could not fetch any of {list(name_candidates)} from OpenML. Last error: {err_last}"
        )

    @staticmethod
    def _apply_row_cap(df: pd.DataFrame, row_cap: Optional[int], rng: np.random.Generator) -> pd.DataFrame:
        """Optionally subsample rows uniformly without replacement.

        Args:
            df: Input dataframe.
            row_cap: If provided and smaller than the number of rows, randomly
                sample exactly ``row_cap`` rows.
            rng: NumPy random generator for reproducibility.

        Returns:
            A (potentially) subsampled dataframe with index reset.
        """
        if row_cap is not None and len(df) > row_cap:
            idx = rng.choice(len(df), size=row_cap, replace=False)
            return df.iloc[idx].reset_index(drop=True)
        return df

    @staticmethod
    def _uci_df_by_id(uci_id: int) -> pd.DataFrame:
        """Fetch a UCI dataset by numeric ID via ``ucimlrepo``.

        Falls back to concatenating ``ids``, ``features`` and ``targets`` if
        ``data.original`` is unavailable in your installed package version.

        Args:
            uci_id: Numeric UCI dataset identifier (e.g., ``294`` for CCPP).

        Returns:
            A pandas DataFrame representing the **original** dataset.

        Raises:
            ImportError: If ``ucimlrepo`` is not installed.
            RuntimeError: If the dataset cannot be assembled into a dataframe.
        """
        if fetch_ucirepo is None:
            raise ImportError("`ucimlrepo` is required for UCI imports.")
        ds = fetch_ucirepo(id=uci_id)
        df = getattr(ds.data, "original", None)
        if df is None or df.empty:
            parts = []
            for attr in ("ids", "features", "targets"):
                part = getattr(ds.data, attr, None)
                if part is not None and not part.empty:
                    parts.append(part)
            if not parts:
                raise RuntimeError("UCI dataset has no original/parts to assemble.")
            df = pd.concat(parts, axis=1)
        df.attrs["name"] = ds.metadata.name
        return df

    @staticmethod
    def _uci_df_by_name(name: str) -> pd.DataFrame:
        """Fetch a UCI dataset by name via ``ucimlrepo``.

        Falls back to concatenating ``ids``, ``features`` and ``targets`` if
        ``data.original`` is unavailable in your installed package version.

        Args:
            name: Dataset name as listed on UCI.

        Returns:
            A pandas DataFrame representing the **original** dataset.

        Raises:
            ImportError: If ``ucimlrepo`` is not installed.
            RuntimeError: If the dataset cannot be assembled into a dataframe.
        """
        if fetch_ucirepo is None:
            raise ImportError("`ucimlrepo` is required for UCI imports.")
        ds = fetch_ucirepo(name=name)
        df = getattr(ds.data, "original", None)
        if df is None or df.empty:
            parts = []
            for attr in ("ids", "features", "targets"):
                part = getattr(ds.data, attr, None)
                if part is not None and not part.empty:
                    parts.append(part)
            if not parts:
                raise RuntimeError("UCI dataset has no original/parts to assemble.")
            df = pd.concat(parts, axis=1)
        df.attrs["name"] = ds.metadata.name
        return df

    # ---------- dataset loaders ----------

    def _load_concrete(self, row_cap: Optional[int]) -> Dict:
        """Concrete Compressive Strength (UCI id=165).

        Predict compressive strength (MPa) of concrete from mix proportions
        (cement, slag, fly ash, water, superplasticizer, coarse/fine aggregate)
        and age. 1,030 instances, 8 numeric features; target is continuous.

        Args:
            row_cap: Optional maximum number of rows to keep (uniform subsample).

        Returns:
            Dict with keys ``X``, ``y``, ``feature_names``, ``target_name``.

        References:
            - UCI: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
            - File (XLS): https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls
        """
        df = self._uci_df_by_name("Concrete Compressive Strength")
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df,
            target_candidates=[
                "Concrete compressive strength",
                "csMPa",
                "CompressiveStrength",
            ],
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_wine_quality(self, which: str, row_cap: Optional[int]) -> Dict:
        """Wine Quality (red/white).

        Two related datasets of Portuguese *Vinho Verde* wines with 11
        physico-chemical tests (e.g., acidity, sulphates, alcohol). Target is
        the sensory quality score (0–10). The red set has 1,599 rows; the white
        set 4,898 rows.

        Args:
            which: Either ``"red"`` or ``"white"``.
            row_cap: Optional maximum number of rows to keep (uniform subsample).

        Returns:
            Dict with keys ``X``, ``y``, ``feature_names``, ``target_name``.

        References:
            - UCI: https://archive.ics.uci.edu/dataset/186/wine+quality
            - Files (CSV, ';'-sep): 
              https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
              https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
        """
        base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
        url = base + ("winequality-red.csv" if which == "red" else "winequality-white.csv")
        df = pd.read_csv(url, sep=";")
        df.attrs["name"] = f"WineQuality-{which.capitalize()} (UCI)"
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["quality"])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_naval(self, row_cap: Optional[int]) -> Dict:
        """Condition Based Maintenance of Naval Propulsion Plants (UCI id=316).

        Gas-turbine (GT) sensor readings on a CODLAG frigate simulator. The
        canonical targets are two decay state coefficients for the compressor
        and turbine components.

        This loader first tries ``ucimlrepo`` (if your version enables import);
        if unavailable, it falls back to downloading the official ZIP and
        parsing ``UCI CBM Dataset/data.txt`` (whitespace-delimited).

        Args:
            row_cap: Optional maximum number of rows to keep (uniform subsample).

        Returns:
            Dict with keys ``X``, ``y``, ``feature_names``, ``target_name``.

        Raises:
            RuntimeError: If the official ZIP cannot be parsed.

        References:
            - UCI: https://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants
        """
        # Prefer ucimlrepo if your version supports this dataset
        if fetch_ucirepo is not None:
            try:
                ds = fetch_ucirepo(id=316)  # may be disabled in some versions
                X = ds.data.features.copy()
                y_df = ds.data.targets.copy()
                preferred = [
                    "GT Turbine decay state coefficient",
                    "GT Compressor decay state coefficient",
                ]
                y_col = next((c for c in preferred if c in y_df.columns), None)
                if y_col is None:
                    meta_tgt = getattr(ds.metadata, "target_col", None)
                    if isinstance(meta_tgt, (list, tuple)) and meta_tgt:
                        y_col = meta_tgt[0]
                    elif isinstance(meta_tgt, str):
                        y_col = meta_tgt
                    else:
                        y_col = y_df.columns[-1]

                X_num = X.select_dtypes(include=[np.number]).copy()
                y_series = y_df[y_col].astype(float)
                df_all = pd.concat([X_num, y_series.rename(y_col)], axis=1)

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
                if "not available for import" not in str(e).lower():
                    raise

        # Fallback: download official UCI ZIP and parse data.txt
        zip_url = "https://archive.ics.uci.edu/static/public/316/condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip"
        r = requests.get(zip_url, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            member = next((n for n in zf.namelist() if n.lower().endswith("data.txt")), None)
            if member is None:
                raise RuntimeError("Naval zip: data.txt not found in archive")
            with zf.open(member) as fh:
                df = pd.read_csv(fh, sep=r"\s+", header=None, engine="python")

        # 16 GT sensor features + 2 decay coefficients (targets)
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
        """CASP / Physicochemical Properties of Protein Tertiary Structure (UCI id=265).

        Predict RMSD from 9 numeric physico-chemical properties; ~45,730 rows.
        (Data origin: CASP 5–9 decoy structures.)

        Args:
            row_cap: Optional maximum number of rows to keep (uniform subsample).

        Returns:
            Dict with keys ``X``, ``y``, ``feature_names``, ``target_name``.

        References:
            - UCI: https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
            - File (CSV): https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
        df = pd.read_csv(url)
        df.attrs["name"] = "CASP / Protein (UCI)"
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["RMSD"])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_year_msd(self, row_cap: Optional[int]) -> Dict:
        """YearPredictionMSD (UCI id=203).

        Predict the release year of a song from audio features (Million Song
        Dataset). ~515k instances total (train/test split available via UCI).

        Args:
            row_cap: Optional maximum number of rows to keep (uniform subsample).

        Returns:
            Dict with keys ``X``, ``y``, ``feature_names``, ``target_name``.

        References:
            - UCI: https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
            - Zip: https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
        r = requests.get(url, timeout=120); r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open("YearPredictionMSD.txt") as fh:
                df = pd.read_csv(fh, header=None)
        df.columns = ["year"] + [f"x{i}" for i in range(1, df.shape[1])]
        df.attrs["name"] = "YearPredictionMSD (UCI)"
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(df, target_candidates=["year"])
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_power(self, row_cap: Optional[int]) -> Dict:
        """Combined Cycle Power Plant (CCPP; UCI id=294).

        Hourly average ambient variables—Temperature (T), Ambient Pressure (AP),
        Relative Humidity (RH), and Exhaust Vacuum (V)—to predict net hourly
        electrical energy output (PE). 9,568 rows.

        Args:
            row_cap: Optional maximum number of rows to keep (uniform subsample).

        Returns:
            Dict with keys ``X``, ``y``, ``feature_names``, ``target_name``.

        References:
            - UCI: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
            - File (Excel): https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP/Folds5x2_pp.xlsx
        """
        df = self._uci_df_by_id(294)
        df = self._apply_row_cap(df, row_cap, np.random.default_rng(0))
        X, y, feat, tgt = self._finalize_Xy_numeric_auto(
            df, target_candidates=["PE", "Net hourly electrical energy output (PE)"]
        )
        return {"X": X, "y": y.astype(float), "feature_names": feat, "target_name": tgt}

    def _load_yacht(self, row_cap: Optional[int]) -> Dict:
        """Yacht Hydrodynamics (UCI id=243).

        Predict residuary resistance of sailing yachts at the initial design
        stage from basic hull dimensions and the Froude number. 308 instances,
        6 numeric predictors, 1 numeric target.

        Args:
            row_cap: Optional maximum number of rows to keep (uniform subsample).

        Returns:
            Dict with keys ``X``, ``y``, ``feature_names``, ``target_name``.

        References:
            - UCI: https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
            - File: https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data
        """
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
        """Placeholder for the 'Shifts' benchmark suite.

        The Shifts benchmark (e.g., Malinin et al., NeurIPS Datasets &
        Benchmarks 2021) is not hosted on UCI/OpenML. Integrate via local files
        or the dedicated package.

        Args:
            row_cap: Unused.

        Raises:
            RuntimeError: Always raised to indicate no built-in source.
        """
        raise RuntimeError(
            "The 'Shifts' benchmark is not hosted on OpenML/sklearn. "
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
        include_shifts: bool = False,
        row_cap: Optional[int] = 100_000,
    ) -> Dict[str, Dict]:
        """Load selected tabular datasets with a consistent return schema.

        Each selected dataset is loaded via its dedicated private loader
        (e.g., ``_load_concrete``) and materialized as a dict with:

        ``{"X": np.ndarray, "y": np.ndarray, "feature_names": List[str], "target_name": str}``.

        Args:
            include_concrete: Include Concrete dataset.
            include_wine_red: Include Wine Quality (red).
            include_wine_white: Include Wine Quality (white).
            include_naval: Include Naval CBM.
            include_protein: Include Protein (CASP).
            include_yearmsd: Include YearPredictionMSD.
            include_power: Include CCPP power plant.
            include_yacht: Include Yacht Hydrodynamics.
            include_shifts: Include Shifts placeholder (raises by design).
            row_cap: Optional maximum number of rows per dataset (uniform subsample).

        Returns:
            Mapping from dataset name to its standardized dict.

        Notes:
            For dataset-specific details (columns, targets, references), see:
                - :py:meth:`TabularDataLoader._load_concrete`
                - :py:meth:`TabularDataLoader._load_wine_quality`
                - :py:meth:`TabularDataLoader._load_naval`
                - :py:meth:`TabularDataLoader._load_protein`
                - :py:meth:`TabularDataLoader._load_year_msd`
                - :py:meth:`TabularDataLoader._load_power`
                - :py:meth:`TabularDataLoader._load_yacht`
                - :py:meth:`TabularDataLoader._load_shifts_placeholder`
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
        include_concrete=True,
        include_wine_red=True,
        include_wine_white=False,   # flip on for OOD swap experiments
        include_naval=True,
        include_protein=True,       # can be big; enable if you want
        include_yearmsd=False,      # large; enable if you want
        include_power=True,
        include_yacht=True,
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
