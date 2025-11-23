"""
Utility script to append the rows from `alignment_eval_interactive_sleeper_agent.csv`
onto the end of `alignment_eval_interactive_sleeper_agent_v1.csv`.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC_PATH = BASE_DIR / "alignment_eval_interactive_sleeper_agent.csv"
TARGET_PATH = BASE_DIR / "alignment_eval_interactive_sleeper_agent_v1.csv"
OUTPUT_PATH = BASE_DIR / "alignment_eval_interactive_sleeper_agent_combined.csv"


def append_csv() -> None:
    """Append the sleeper agent CSV onto the v1 file in-place."""
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Missing source CSV: {SRC_PATH}")
    if not TARGET_PATH.exists():
        raise FileNotFoundError(f"Missing target CSV: {TARGET_PATH}")

    src_df = pd.read_csv(SRC_PATH)
    target_df = pd.read_csv(TARGET_PATH)

    combined = pd.concat([target_df, src_df], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Wrote combined CSV with {len(combined)} rows to {OUTPUT_PATH.name} "
        f"(source={len(src_df)} rows, target={len(target_df)} rows)."
    )


if __name__ == "__main__":
    append_csv()
