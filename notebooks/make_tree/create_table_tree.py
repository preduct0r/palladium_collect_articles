#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as import_error:  # pragma: no cover
    sys.stderr.write(
        "[ERROR] pandas is required to run this script. Install with: pip install pandas openpyxl\n"
    )
    raise


CANONICAL_COLUMNS = [
    "Плитка на портале",
    "Наименование услуги",
    "Типовой запрос",
]


def normalize_text(value: object) -> str:
    text = str(value) if value is not None and value is not pd.NA else ""
    # Normalize spaces and case
    text = text.replace("\xa0", " ")  # non-breaking -> space
    text = text.replace("\n", " ")
    text = " ".join(text.split())  # collapse whitespace
    return text.strip().lower()


def find_required_columns_by_header_row(sheet_df: pd.DataFrame) -> Optional[Tuple[int, Dict[str, int]]]:
    """Find a row where all required headers are present, return (row_index, mapping)
    mapping maps canonical header -> column index in the sheet_df where header was found.
    """
    required_norm = {normalize_text(h): h for h in CANONICAL_COLUMNS}

    # Search only first 50 rows to avoid scanning huge sheets unnecessarily
    max_rows_to_scan = min(50, len(sheet_df))
    for row_idx in range(max_rows_to_scan):
        row_values = sheet_df.iloc[row_idx].tolist()
        norm_to_col_index: Dict[str, int] = {}
        for col_idx, cell in enumerate(row_values):
            norm_cell = normalize_text(cell)
            if norm_cell:
                norm_to_col_index[norm_cell] = col_idx
        mapping: Dict[str, int] = {}
        found_all = True
        for norm_name, canonical in required_norm.items():
            if norm_name not in norm_to_col_index:
                found_all = False
                break
            mapping[canonical] = norm_to_col_index[norm_name]
        if found_all:
            return row_idx, mapping
    return None


def normalize_column_name(column_name: object) -> str:
    return normalize_text(column_name)


def find_required_columns(columns: List[object]) -> Optional[Dict[str, str]]:
    normalized_to_actual: Dict[str, str] = {}
    for actual in columns:
        normalized_to_actual[normalize_column_name(actual)] = str(actual)

    required_normalized = {normalize_column_name(c): c for c in CANONICAL_COLUMNS}

    mapping: Dict[str, str] = {}
    for normalized, canonical in required_normalized.items():
        if normalized not in normalized_to_actual:
            return None
        mapping[canonical] = normalized_to_actual[normalized]

    return mapping


def load_and_collect_rows(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_path}")

    # Read all sheets with header=None to detect header rows manually
    excel_data: Dict[str, pd.DataFrame] = pd.read_excel(
        input_path, sheet_name=None, dtype=object, engine="openpyxl", header=None
    )

    collected_frames: List[pd.DataFrame] = []

    for sheet_name, raw_df in excel_data.items():
        if raw_df is None or raw_df.empty:
            continue

        # Try to detect header row by values
        header_detection = find_required_columns_by_header_row(raw_df)
        if header_detection is not None:
            header_row_idx, col_index_map = header_detection
            # Data starts after header row
            data_df = raw_df.iloc[header_row_idx + 1 :].reset_index(drop=True)
            # Extract columns by position
            extracted = pd.DataFrame(
                {
                    canonical: data_df.iloc[:, col_index_map[canonical]] if col_index_map[canonical] < data_df.shape[1] else pd.Series([], dtype=object)
                    for canonical in CANONICAL_COLUMNS
                }
            )
        else:
            # Fallback: attempt to treat the first row as header and map by names
            fallback_df = pd.read_excel(
                input_path, sheet_name=sheet_name, dtype=object, engine="openpyxl", header=0
            )
            mapping = find_required_columns(list(fallback_df.columns))
            if mapping is None:
                continue
            extracted = fallback_df[[mapping[c] for c in CANONICAL_COLUMNS]].copy()
            extracted.columns = CANONICAL_COLUMNS

        # Clean and drop empty rows
        extracted.replace({"": pd.NA}, inplace=True)
        extracted = extracted.dropna(how="all", subset=CANONICAL_COLUMNS)

        # Normalize strings
        for col in CANONICAL_COLUMNS:
            extracted[col] = extracted[col].astype("string").map(
                lambda v: " ".join(v.split()).strip() if isinstance(v, str) else v
            )

        # Drop rows which are effectively headers repeated in body
        mask_is_header = extracted.apply(lambda r: all(normalize_text(r[c]) == normalize_text(c) for c in CANONICAL_COLUMNS), axis=1)
        extracted = extracted[~mask_is_header]

        if not extracted.empty:
            collected_frames.append(extracted)

    if not collected_frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    combined = pd.concat(collected_frames, ignore_index=True)

    # Final cleanup: drop duplicates
    combined = combined.drop_duplicates().reset_index(drop=True)

    return combined


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_columns = ["Плитка на портале", "Наименование услуги"]
    present_sort_columns = [c for c in sort_columns if c in df.columns]
    if not present_sort_columns:
        return df
    return df.sort_values(by=present_sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Tree", index=False)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Excel table with columns: 'Плитка на портале', 'Наименование услуги', 'Типовой запрос' "
            "by aggregating rows from all sheets in the input workbook that contain these columns."
        )
    )
    default_input = \
        "/home/ubuntu/kotov_projects/palladium_collect_articles/notebooks/Super.xlsx"
    default_output = \
        "/home/ubuntu/kotov_projects/palladium_collect_articles/notebooks/make_tree/table_tree.xlsx"

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=default_input,
        help=f"Path to input Excel file (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help=f"Path to output Excel file (default: {default_output})",
    )

    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        df = load_and_collect_rows(input_path)
        df_sorted = sort_rows(df)
        write_output(df_sorted, output_path)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[ERROR] {exc}\n")
        return 1

    total_rows = len(df_sorted.index)
    processed_info = (
        f"Created '{output_path}' with {total_rows} row(s) and columns: {', '.join(CANONICAL_COLUMNS)}\n"
    )
    sys.stdout.write(processed_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
