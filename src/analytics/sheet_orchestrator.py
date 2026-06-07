"""
SheetOrchestrator — pre-report sheet filtering, similarity classification,
and selection logic. Zero LLM calls. Pure pandas + metadata.

Integration:
    from src.analytics.sheet_orchestrator import SheetOrchestrator

    orchestrator = SheetOrchestrator()
    result = orchestrator.prepare(sheets_data)
    # result.sheets_to_process  → List[Dict]  (what report generation receives)
    # result.case               → 'homogeneous' | 'heterogeneous' | 'single'
    # result.excluded_sheets    → List[Dict]   (for executive summary transparency)
    # result.selection_notes    → str          (human-readable summary for report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Caps & Thresholds ────────────────────────────────────────────────────────

MIN_ROWS_FOR_REPORT: int = 5         # sheets below this are excluded
MAX_SHEETS_BEFORE_CAP: int = 3        # hard cap sent into analysis
MIN_COLUMN_QUALITY_RATIO: float = 0.3 # fraction of cols that must be non-trivial
HOMOGENEOUS_OVERLAP_THRESHOLD: float = 0.60  # jaccard similarity for "same schema"
MAX_COLUMNS_IN_SCHEMA_PROMPT: int = 40       # wide-sheet guard for LLM context
MAX_CONCAT_ROWS: int = 50_000               # sample threshold for concat path


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class OrchestrationResult:
    sheets_to_process: List[Dict[str, Any]]          # final list ready for report
    case: str                                         # 'single' | 'homogeneous' | 'heterogeneous'
    excluded_sheets: List[Dict[str, str]] = field(default_factory=list)
    selection_notes: str = ""
    concat_used: bool = False                        # True when homogeneous concat was applied


# ── Main class ───────────────────────────────────────────────────────────────

class SheetOrchestrator:
    """
    Stateless helper. Call prepare() once per report request.
    All methods are pure functions over the input data — no side effects.
    """

    def prepare(self, sheets_data: List[Dict[str, Any]]) -> OrchestrationResult:
        """
        Entry point. Returns an OrchestrationResult describing what to process.

        sheets_data items are expected to have:
            'file_name', 'sheet_name', 'df' (cleaned DataFrame), 'metadata'
        """
        if not sheets_data:
            raise ValueError("sheets_data is empty — nothing to orchestrate.")

        # ── 1. Pre-filter ────────────────────────────────────────────────────
        kept, excluded = self._prefilter_sheets(sheets_data)

        if not kept:
            raise ValueError(
                "All sheets were excluded by pre-filter. "
                "Check MIN_ROWS_FOR_REPORT and MIN_COLUMN_QUALITY_RATIO thresholds. "
                f"Excluded: {[e['sheet_name'] for e in excluded]}"
            )

        # ── 2. Single-sheet fast path ────────────────────────────────────────
        if len(kept) == 1:
            logger.info("[Orchestrator] Single sheet — fast path, no classification needed.")
            return OrchestrationResult(
                sheets_to_process=kept,
                case="single",
                excluded_sheets=excluded,
                selection_notes=self._build_notes("single", kept, excluded),
            )

        # ── 3. Apply MAX_SHEETS_BEFORE_CAP ──────────────────────────────────
        kept, newly_excluded = self._apply_sheet_cap(kept)
        excluded.extend(newly_excluded)

        # ── 4. Classify ─────────────────────────────────────────────────────
        case, _ = self._classify_similarity(kept)
        logger.info("[Orchestrator] Classified as: %s", case)

        # ── 5. Route ─────────────────────────────────────────────────────────
        if case == "homogeneous":
            merged = self._merge_sheets_for_analysis(kept)
            return OrchestrationResult(
                sheets_to_process=[merged],
                case="homogeneous",
                excluded_sheets=excluded,
                selection_notes=self._build_notes("homogeneous", kept, excluded),
                concat_used=True,
            )

        # heterogeneous → top-1
        selected, not_selected = self._select_top_k_heterogeneous(kept, k=3)
        excluded.extend(not_selected)
        return OrchestrationResult(
            sheets_to_process=selected,
            case="heterogeneous",
            excluded_sheets=excluded,
            selection_notes=self._build_notes("heterogeneous", selected, excluded),
        )

    # ── Pre-filter ────────────────────────────────────────────────────────────

    def _prefilter_sheets(
        self, sheets_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        Apply three exclusion rules — all pure pandas, no LLM.

        Rules (applied in order):
          R1: Row count < MIN_ROWS_FOR_REPORT
          R2: Column quality ratio < MIN_COLUMN_QUALITY_RATIO
               (too many trivial/all-null/all-same-value columns)
          R3: Sheet has fewer than 2 usable columns after quality check
        """
        kept: List[Dict[str, Any]] = []
        excluded: List[Dict[str, str]] = []

        ORCHESTRATOR_EXCLUDE = "[Orchestrator] Excluding '%s': %s"
        
        for sheet in sheets_data:
            df: pd.DataFrame = sheet.get("df")
            sheet_name: str = sheet.get("sheet_name", "unknown")

            if df is None or not isinstance(df, pd.DataFrame):
                excluded.append({"sheet_name": sheet_name, "reason": "No DataFrame found"})
                continue

            # R1 — row count
            if len(df) < MIN_ROWS_FOR_REPORT:
                reason = f"Too few rows ({len(df)} < {MIN_ROWS_FOR_REPORT})"
                logger.info(ORCHESTRATOR_EXCLUDE, sheet_name, reason)
                excluded.append({"sheet_name": sheet_name, "reason": reason})
                continue

            # R2 — column quality
            quality_ratio = self._column_quality_ratio(df)
            if quality_ratio < MIN_COLUMN_QUALITY_RATIO:
                reason = (
                    f"Low column quality ({quality_ratio:.0%} non-trivial columns "
                    f"< {MIN_COLUMN_QUALITY_RATIO:.0%} threshold)"
                )
                logger.info(ORCHESTRATOR_EXCLUDE, sheet_name, reason)
                excluded.append({"sheet_name": sheet_name, "reason": reason})
                continue

            # R3 — usable column count
            usable_cols = self._count_usable_columns(df)
            if usable_cols < 2:
                reason = f"Fewer than 2 usable columns ({usable_cols})"
                logger.info(ORCHESTRATOR_EXCLUDE, sheet_name, reason)
                excluded.append({"sheet_name": sheet_name, "reason": reason})
                continue

            kept.append(sheet)

        logger.info(
            "[Orchestrator] Pre-filter: %d kept, %d excluded (of %d total)",
            len(kept), len(excluded), len(sheets_data),
        )
        return kept, excluded

    def _column_quality_ratio(self, df: pd.DataFrame) -> float:
        """
        Fraction of columns that are 'non-trivial':
          - Not all-null
          - Not all same value (cardinality > 1)
          - Not a pure-index integer column (0,1,2,3...)
        """
        if df.empty or len(df.columns) == 0:
            return 0.0

        non_trivial = 0
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if series.nunique() <= 1:
                continue
            # Detect pure sequential integer index columns
            if pd.api.types.is_integer_dtype(series):
                if series.is_monotonic_increasing and (series.max() - series.min()) == (len(series) - 1):
                    continue
            non_trivial += 1

        return non_trivial / len(df.columns)

    def _count_usable_columns(self, df: pd.DataFrame) -> int:
        """Count columns with at least some non-null data and cardinality > 1."""
        count = 0
        for col in df.columns:
            s = df[col].dropna()
            if len(s) > 0 and s.nunique() > 1:
                count += 1
        return count

    # ── Sheet cap ────────────────────────────────────────────────────────────

    def _apply_sheet_cap(
        self, sheets: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        If sheet count exceeds MAX_SHEETS_BEFORE_CAP, keep the most data-dense ones.
        Density = rows × (non-null column ratio).
        Returns (kept, newly_excluded).
        """
        if len(sheets) <= MAX_SHEETS_BEFORE_CAP:
            return sheets, []

        def density(sheet: Dict[str, Any]) -> float:
            df = sheet["df"]
            non_null_ratio = df.notna().mean().mean()
            return len(df) * len(df.columns) * non_null_ratio

        ranked = sorted(sheets, key=density, reverse=True)
        kept = ranked[:MAX_SHEETS_BEFORE_CAP]
        dropped = ranked[MAX_SHEETS_BEFORE_CAP:]

        newly_excluded = [
            {
                "sheet_name": s["sheet_name"],
                "reason": (
                    f"Sheet cap: only top {MAX_SHEETS_BEFORE_CAP} sheets by data density are analysed "
                    f"(workbook had {len(sheets)} qualifying sheets)"
                ),
            }
            for s in dropped
        ]
        logger.info(
            "[Orchestrator] Sheet cap applied: kept %d, dropped %d",
            len(kept), len(dropped),
        )
        return kept, newly_excluded

    # ── Similarity classification ─────────────────────────────────────────────

    def _classify_similarity(
        self, sheets: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Classify sheets as 'homogeneous' or 'heterogeneous' using:
          - Jaccard similarity of lowercased column name sets
          - Dtype profile match (fraction of numeric / categorical / datetime cols)

        Returns (case_str, similarity_matrix).
        similarity_matrix is NxN float32 array of pairwise scores (None if N==1).
        """
        n = len(sheets)
        if n == 1:
            return "homogeneous", None

        col_sets = [
            {c.lower().strip() for c in s["df"].columns}
            for s in sheets
        ]
        dtype_profiles = [self._dtype_profile(s["df"]) for s in sheets]

        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                jaccard = self._jaccard(col_sets[i], col_sets[j])
                dtype_sim = self._dtype_profile_similarity(dtype_profiles[i], dtype_profiles[j])
                # Weighted combination: column names carry more signal
                score = 0.7 * jaccard + 0.3 * dtype_sim
                matrix[i, j] = score
                matrix[j, i] = score

        # Average pairwise similarity across all pairs
        pair_count = n * (n - 1) / 2
        avg_similarity = matrix[np.triu_indices(n, k=1)].sum() / pair_count

        logger.info(
            "[Orchestrator] Average pairwise schema similarity: %.3f (threshold: %.2f)",
            avg_similarity, HOMOGENEOUS_OVERLAP_THRESHOLD,
        )

        case = "homogeneous" if avg_similarity >= HOMOGENEOUS_OVERLAP_THRESHOLD else "heterogeneous"
        return case, matrix

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _dtype_profile(df: pd.DataFrame) -> Dict[str, float]:
        """
        Fraction of columns that are numeric / categorical / datetime.
        Returns a dict with three float values summing to ~1.0.
        """
        total = max(len(df.columns), 1)
        numeric = sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))
        datetime_ = sum(1 for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]))
        categorical = total - numeric - datetime_
        return {
            "numeric": numeric / total,
            "categorical": categorical / total,
            "datetime": datetime_ / total,
        }

    @staticmethod
    def _dtype_profile_similarity(p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """1 - mean absolute difference across dtype fractions."""
        keys = ["numeric", "categorical", "datetime"]
        mad = sum(abs(p1[k] - p2[k]) for k in keys) / len(keys)
        return 1.0 - mad

    # ── Homogeneous merge ─────────────────────────────────────────────────────

    def _merge_sheets_for_analysis(
        self, sheets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Concat all homogeneous sheets into one synthetic sheet entry.
        Adds a '_source_sheet' column for provenance.
        Samples down to MAX_CONCAT_ROWS if needed (stratified by source).
        Returns a single sheet_data dict compatible with the existing pipeline.
        """
        frames: List[pd.DataFrame] = []
        for sheet in sheets:
            df = sheet["df"].copy()
            df["_source_sheet"] = sheet["sheet_name"] or sheet["file_name"]
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True, sort=False)

        # Fill NaN for columns that only some sheets had
        # (outer-join columns from different sheets)
        for col in combined.columns:
            if col == "_source_sheet":
                continue
            if combined[col].dtype == "object":
                combined[col] = combined[col].fillna("n/a")

        if len(combined) > MAX_CONCAT_ROWS:
            logger.info(
                "[Orchestrator] Concat DataFrame too large (%d rows) — stratified sampling to %d",
                len(combined), MAX_CONCAT_ROWS,
            )
            combined = self._stratified_sample(combined, MAX_CONCAT_ROWS)

        # Limit columns for schema prompt (wide-sheet guard)
        if len(combined.columns) > MAX_COLUMNS_IN_SCHEMA_PROMPT:
            combined = self._select_top_columns(combined)

        # Reconstruct a metadata dict for the combined sheet
        # (reuse existing extract_file_metadata logic downstream)
        merged_sheet = {
            "file_name": sheets[0]["file_name"],
            "sheet_name": f"Combined ({len(sheets)} sheets)",
            "df": combined,
            "s3_url": sheets[0].get("s3_url"),
            # metadata will be regenerated by ReportGenerator as before
        }

        logger.info(
            "[Orchestrator] Merged %d sheets → %d rows × %d columns",
            len(sheets), len(combined), len(combined.columns),
        )
        return merged_sheet

    def _stratified_sample(self, df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
        """
        Stratified sample by '_source_sheet' column so each sheet contributes
        proportionally. Falls back to random sample if stratification fails.
        """
        try:
            groups = df.groupby("_source_sheet", group_keys=False)
            return groups.apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, int(target_rows * len(g) / len(df)))),
                    random_state=42,
                )
            ).reset_index(drop=True)
        except Exception as exc:
            logger.warning("[Orchestrator] Stratified sample failed (%s), using random sample.", exc)
            return df.sample(n=target_rows, random_state=42).reset_index(drop=True)

    def _select_top_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For wide DataFrames, keep the MAX_COLUMNS_IN_SCHEMA_PROMPT most informative columns.
        Always retain '_source_sheet'. Prioritises by non-null count then cardinality.
        """
        protected = ["_source_sheet"]
        candidates = [c for c in df.columns if c not in protected]

        scores = {
            col: df[col].notna().sum() * min(df[col].nunique(), 100)
            for col in candidates
        }
        top_cols = sorted(candidates, key=lambda c: scores[c], reverse=True)[
            : MAX_COLUMNS_IN_SCHEMA_PROMPT - len(protected)
        ]

        final_cols = protected + top_cols
        logger.info(
            "[Orchestrator] Wide sheet trimmed: %d → %d columns for schema prompt",
            len(df.columns), len(final_cols),
        )
        return df[final_cols]

    # ── Heterogeneous selection ───────────────────────────────────────────────

    def _select_top_k_heterogeneous(
        self, sheets: List[Dict[str, Any]], k: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        For heterogeneous workbooks, select the top-k sheets by data density.
        density = rows × non-null column ratio × usable column count

        Returns (selected_sheets, excluded_as_dicts).
        """
        def density(sheet: Dict[str, Any]) -> float:
            df = sheet["df"]
            non_null_ratio = df.notna().mean().mean()
            usable = self._count_usable_columns(df)
            return len(df) * non_null_ratio * usable

        ranked = sorted(sheets, key=density, reverse=True)
        selected = ranked[:k]
        not_selected = ranked[k:]

        excluded = [
            {
                "sheet_name": s["sheet_name"],
                "reason": (
                    "Heterogeneous workbook: only the most data-dense sheet is analysed. "
                    "Upload this sheet separately for a dedicated report."
                ),
            }
            for s in not_selected
        ]

        logger.info(
            "[Orchestrator] Heterogeneous: selected '%s' (top-%d of %d sheets by density)",
            selected[0]["sheet_name"], k, len(sheets),
        )
        return selected, excluded

    # ── Human-readable notes ──────────────────────────────────────────────────

    def _build_notes(
        self,
        case: str,
        processed: List[Dict[str, Any]],
        excluded: List[Dict[str, str]],
    ) -> str:
        processed_names = [s.get("sheet_name", "?") for s in processed]
        excluded_summary = (
            "; ".join(f"{e['sheet_name']} ({e['reason']})" for e in excluded)
            if excluded
            else "None"
        )

        if case == "single":
            return f"Single sheet analysed: {processed_names[0]}. No other sheets were present."

        if case == "homogeneous":
            return (
                f"All {len(processed)} sheets share a compatible schema and have been merged "
                f"into a single analysis. Sheets: {', '.join(processed_names)}. "
                f"Excluded: {excluded_summary}."
            )

        return (
            f"Sheets have heterogeneous schemas. The most data-dense sheet was selected for analysis: "
            f"{processed_names[0]}. "
            f"Excluded from this report: {excluded_summary}. "
            f"Upload individual sheets separately for dedicated reports."
        )