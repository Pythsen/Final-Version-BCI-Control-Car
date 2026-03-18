import argparse
from pathlib import Path
from typing import Optional

import mne
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "BCICIV_2a_gdf"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "timelines"


def generate(subject_id: int = 5, out_file: Optional[Path] = None) -> Path:
    gdf_path = DATA_DIR / f"A0{subject_id}E.gdf"
    if not gdf_path.exists():
        raise FileNotFoundError(f"GDF file not found: {gdf_path}")

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=False, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    target_id = None
    for name, code in event_id.items():
        if "783" in str(name):
            target_id = code
            break
    if target_id is None:
        raise RuntimeError("No event 783 found in GDF annotations.")

    cues = events[events[:, 2] == target_id][:, 0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = out_file if out_file else OUTPUT_DIR / "cue_timeline0tt.txt"
    np.savetxt(output_path, cues, fmt="%d")

    print(f"[OK] Saved {len(cues)} cues to: {output_path}")
    print(f"[Info] First 5 cue indices: {cues[:5] if len(cues) >= 5 else cues}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cue timeline from BCI Competition IV 2a GDF file.")
    parser.add_argument("--subject", type=int, default=5, choices=range(1, 10), help="Subject ID (1-9), default=5.")
    parser.add_argument("--output", type=str, default="", help="Optional output txt file path.")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate(subject_id=args.subject, out_file=output_path)


if __name__ == "__main__":
    main()
