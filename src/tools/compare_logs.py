import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "online_accuracy"


def parse_table_file(filepath: Path) -> dict[int, dict]:
    data = {}
    pattern = re.compile(r"#(\d+)\s+\|\s+(\w+)\s+\|\s+(\w+)\s+\|\s+([\d\.]+)")
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line.strip())
            if not m:
                continue
            trial_id = int(m.group(1))
            data[trial_id] = {
                "true_label": m.group(2),
                "pred_label": m.group(3),
            }
    return data


def parse_runtime_log(filepath: Path) -> dict[int, dict]:
    data = {}
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    matches = re.findall(r"Cue #(\d+) triggered.*?RESULT: (\w+) \(([\d\.]+)\)", content, re.DOTALL)
    for trial_id, pred, conf in matches:
        data[int(trial_id)] = {"pred_label": pred, "conf": float(conf)}
    return data


def compare(table_path: Path, runtime_path: Path) -> None:
    if not table_path.exists():
        raise FileNotFoundError(f"Table file not found: {table_path}")
    if not runtime_path.exists():
        raise FileNotFoundError(f"Runtime file not found: {runtime_path}")

    table_data = parse_table_file(table_path)
    runtime_data = parse_runtime_log(runtime_path)
    common_ids = sorted(set(table_data.keys()) & set(runtime_data.keys()))

    if not common_ids:
        print("[Warn] No overlapping trial IDs found.")
        return

    table_correct = 0
    runtime_correct = 0
    mismatch = 0

    for tid in common_ids:
        true_label = table_data[tid]["true_label"]
        table_pred = table_data[tid]["pred_label"]
        runtime_pred = runtime_data[tid]["pred_label"]
        if table_pred == true_label:
            table_correct += 1
        if runtime_pred == true_label:
            runtime_correct += 1
        if table_pred != runtime_pred:
            mismatch += 1

    total = len(common_ids)
    print("=" * 64)
    print(f"Samples compared: {total}")
    print(f"Table accuracy : {table_correct}/{total} ({table_correct / total:.2%})")
    print(f"Runtime accuracy: {runtime_correct}/{total} ({runtime_correct / total:.2%})")
    print(f"Prediction mismatch count: {mismatch}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare online table output and runtime logs.")
    parser.add_argument("--table", default=str(LOG_DIR / "Ee5.txt"), help="Path to table file (Ee*.txt).")
    parser.add_argument("--runtime", default=str(LOG_DIR / "e5.txt"), help="Path to runtime file (e*.txt).")
    args = parser.parse_args()
    compare(Path(args.table), Path(args.runtime))


if __name__ == "__main__":
    main()
