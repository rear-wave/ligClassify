"""Distance distribution analysis for each classified year directory.

For each <year>_distance directory, read the four non-IC type prediction CSVs
and report, per type and overall:
  - piece count per 100-km distance bin (0-3000 km)
  - share of each bin
  - mean expected_distance_km, mean distance_confidence
  - coverage / total pieces
Writes a per-directory analysis text report and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
NBINS = 30


def analyze_type(csv_path: Path) -> dict:
    bins = [0] * NBINS
    total = 0
    exp_sum = 0.0
    conf_sum = 0.0
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                b = int(row["distance_bin"])
                exp = float(row["expected_distance_km"])
                conf = float(row["distance_confidence"])
            except (KeyError, ValueError):
                continue
            if 0 <= b < NBINS:
                bins[b] += 1
            total += 1
            exp_sum += exp
            conf_sum += conf
    return {
        "bins": bins,
        "total": total,
        "mean_expected_km": (exp_sum / total) if total else 0.0,
        "mean_confidence": (conf_sum / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="dir containing <year>_distance subdirs")
    parser.add_argument("--output", default=None,
                        help="optional directory to write per-year reports")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    year_dirs = sorted(p for p in root.iterdir()
                       if p.is_dir() and p.name.endswith("_distance"))
    for yd in year_dirs:
        lines = [f"===== {yd.name} 距离分布分析 =====", ""]
        grand_total = 0
        grand_bins = [0] * NBINS
        for t in DISTANCE_NAMES:
            csv_path = yd / t / f"{t}_distance_predictions.csv"
            if not csv_path.is_file():
                lines.append(f"[{t}] 缺失")
                continue
            st = analyze_type(csv_path)
            grand_total += st["total"]
            for i in range(NBINS):
                grand_bins[i] += st["bins"][i]
            lines.append(f"--- {t} (共 {st['total']} 片段) ---")
            lines.append(f"  平均期望距离: {st['mean_expected_km']:.1f} km | 平均置信度: {st['mean_confidence']:.3f}")
            lines.append("  距离档(km)     片段数    占比")
            for i in range(NBINS):
                c = st["bins"][i]
                share = (c / st["total"]) if st["total"] else 0.0
                if c > 0:
                    lines.append(f"  {i*100:04d}-{i*100+100:04d}      {c:>8d}   {share*100:5.2f}%")
            lines.append("")
        # overall
        lines.append(f"=== 合计 (共 {grand_total} 片段) ===")
        lines.append("  距离档(km)     片段数    占比")
        for i in range(NBINS):
            c = grand_bins[i]
            share = (c / grand_total) if grand_total else 0.0
            if c > 0:
                lines.append(f"  {i*100:04d}-{i*100+100:04d}      {c:>8d}   {share*100:5.2f}%")
        report = "\n".join(lines) + "\n"
        print(report, flush=True)
        if out_dir:
            rp = out_dir / f"{yd.name}_distribution.txt"
            rp.write_text(report, encoding="utf-8")
            print(f"  -> 写入 {rp}", flush=True)


if __name__ == "__main__":
    main()