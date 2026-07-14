"""
Quick MTL inference: type + distance for existing classified .lig files.
Filters by date in filename (YYMMDD pattern).
"""
import argparse
import csv
import os
import struct

import numpy as np
import torch
from models import create_mtl_model
from data.preprocessing import preprocess_batch
from distance_ordinal import decode_distance_logits
from open_set import decode_with_rejection

MAX_PER_FILE = 512
PREDICTION_FIELDS = [
    "source_file",
    "piece_index",
    "type",
    "distance_km",
    "bin_start_km",
    "low_km",
    "high_km",
    "confidence",
    "type_confidence",
    "raw_type",
    "type_margin",
    "feature_distance",
    "rejection_reason",
    "final_type",
    "head_source",
    "status",
]


class PredictionCsvWriter:
    """Streaming CSV writer for auditable piece-level predictions."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self.handle = None
        self.writer = None

    def __enter__(self):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        self.handle = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=PREDICTION_FIELDS)
        self.writer.writeheader()
        return self

    def write(self, row):
        self.writer.writerow({field: row.get(field, "") for field in PREDICTION_FIELDS})

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()


def load_mtl_checkpoint(path, device):
    """Load either the legacy or ordinal-v2 structured checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Inference requires a structured MTL checkpoint")
    architecture = checkpoint.get("model_name", "mtl_resnet")
    model = create_mtl_model(
        base_channels=checkpoint.get("base_channels", 64),
        architecture=architecture,
        num_types=len(checkpoint.get("type_names", [])) or 5,
        dist_mlp_dim=checkpoint.get("dist_mlp_dim", 128),
        dist_dropout=checkpoint.get("dist_dropout", 0.2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def checkpoint_schema(checkpoint):
    """Return an explicit supported type-label contract."""
    if checkpoint.get("task_schema") == "four_class_rejection_v1":
        return "four_class_rejection_v1"
    if checkpoint.get("type_names") == ["IC", "NCG", "NNBE", "PCG", "PNBE"]:
        return "legacy_five_class"
    raise ValueError("Unsupported checkpoint type schema")


def decode_four_class_predictions(type_logits, features, checkpoint):
    """Decode four researched types and route rejected rows to IC."""
    if checkpoint_schema(checkpoint) != "four_class_rejection_v1":
        raise ValueError("Four-class rejection requires a four-class checkpoint")
    decoded = decode_with_rejection(
        type_logits, features, checkpoint["type_rejection"]
    )
    predictions = []
    for row, type_index in enumerate(decoded["predicted"].tolist()):
        raw_type = checkpoint["type_names"][type_index]
        accepted = bool(decoded["accepted"][row].item())
        final_type = raw_type if accepted else checkpoint.get(
            "rejected_type_name", "IC"
        )
        predictions.append({
            "type": final_type,
            "raw_type": raw_type,
            "final_type": final_type,
            "class_name": final_type,
            "distance_km": None,
            "bin_start_km": None,
            "low_km": None,
            "high_km": None,
            "confidence": float(decoded["confidence"][row].item()),
            "type_confidence": float(decoded["confidence"][row].item()),
            "type_margin": float(decoded["margin"][row].item()),
            "feature_distance": float(
                decoded["feature_distance"][row].item()
            ),
            "rejection_reason": decoded["reason"][row],
            "head_source": "type",
            "status": "reliable" if accepted else "rejected",
        })
    return predictions


def validate_type_only_options(checkpoint, min_type_confidence=0.0):
    """Prevent legacy confidence overrides on calibrated four-class models."""
    if (
        checkpoint_schema(checkpoint) == "four_class_rejection_v1"
        and float(min_type_confidence) != 0.0
    ):
        raise ValueError(
            "--min_type_confidence is not valid for a four-class rejection "
            "checkpoint"
        )


def validate_checkpoint_pair(old_checkpoint, new_checkpoint):
    """Reject hybrid checkpoints that do not use the same label contract."""
    for field in ("type_names", "dist_names", "dist_bin_starts"):
        if old_checkpoint.get(field) != new_checkpoint.get(field):
            raise ValueError(f"Hybrid checkpoints disagree on {field}")
    old_mode = old_checkpoint.get("preprocessing", {}).get(
        "normalize_mode", "minmax"
    )
    new_mode = new_checkpoint.get("preprocessing", {}).get(
        "normalize_mode", "minmax"
    )
    if old_mode != new_mode:
        raise ValueError("Hybrid checkpoints disagree on preprocessing")


def decode_hybrid_prediction(
    old_type_logits,
    old_distance_logits,
    new_distance_logits,
    old_checkpoint,
    new_checkpoint,
    old_distance_types=("NCG",),
):
    """Use the proven old type head and the best distance head per type."""
    probabilities = torch.softmax(old_type_logits, dim=0)
    type_index = int(probabilities.argmax().item())
    type_name = old_checkpoint["type_names"][type_index]
    if type_name == "IC":
        prediction = decode_piece_prediction(
            type_index, old_distance_logits, old_checkpoint
        )
        prediction["type_confidence"] = float(probabilities[type_index].item())
        prediction["head_source"] = "type"
        return prediction
    use_old = type_name in old_distance_types
    checkpoint = old_checkpoint if use_old else new_checkpoint
    distance_logits = old_distance_logits if use_old else new_distance_logits
    prediction = decode_piece_prediction(type_index, distance_logits, checkpoint)
    prediction["type_confidence"] = float(probabilities[type_index].item())
    prediction["head_source"] = "old" if use_old else "new"
    return prediction


def decode_type_prediction(type_logits, checkpoint, min_confidence=0.0):
    """Decode one type prediction without invoking any distance logic."""
    probabilities = torch.softmax(type_logits, dim=0)
    type_index = int(probabilities.argmax().item())
    type_name = checkpoint["type_names"][type_index]
    confidence = float(probabilities[type_index].item())
    uncertain = type_name != "IC" and confidence < min_confidence
    return {
        "type": type_name,
        "class_name": f"uncertained_{type_name}" if uncertain else type_name,
        "distance_km": None,
        "bin_start_km": None,
        "low_km": None,
        "high_km": None,
        "confidence": confidence,
        "type_confidence": confidence,
        "head_source": "type",
        "status": "uncertain" if uncertain else "reliable",
    }


def decode_piece_prediction(type_index, distance_logits, checkpoint):
    """Decode one routed distance prediction with checkpoint calibration."""
    type_names = checkpoint["type_names"]
    type_name = type_names[type_index]
    if type_name == "IC":
        return {
            "type": type_name,
            "class_name": "IC",
            "distance_km": None,
            "bin_start_km": None,
            "low_km": None,
            "high_km": None,
            "confidence": None,
            "status": "reliable",
        }

    head_index = checkpoint["dist_names"].index(type_name)
    logits = distance_logits[head_index]
    training = checkpoint.get("distance_training", {})
    mode = training.get("prediction", "argmax")
    calibration = checkpoint.get("distance_calibration", {})
    temperatures = calibration.get("temperatures", [1.0] * 4)
    temperature = float(temperatures[head_index])

    if mode == "expected":
        decoded = decode_distance_logits(
            logits.unsqueeze(0), temperature=temperature
        )
        dist_bin = int(decoded["bin"].item())
        distance_km = float(decoded["distance_km"].item())
        low_km = float(decoded["low_km"].item())
        high_km = float(decoded["high_km"].item())
        confidence = float(decoded["confidence"].item())
    else:
        probabilities = torch.softmax(logits / temperature, dim=0)
        dist_bin = int(probabilities.argmax().item())
        distance_km = float(100 * (dist_bin + 0.5))
        low_km = float(100 * dist_bin)
        high_km = float(100 * (dist_bin + 1))
        confidence = float(probabilities[dist_bin].item())

    bin_start = int(checkpoint["dist_bin_starts"][dist_bin])
    threshold = float(calibration.get("confidence_threshold", 0.0))
    reliable = confidence >= threshold
    status = "reliable" if reliable else "uncertain"
    class_name = (
        f"{type_name}_{bin_start}-{bin_start + 100}km"
        if reliable else f"UNCERTAIN_{type_name}"
    )
    return {
        "type": type_name,
        "class_name": class_name,
        "distance_km": distance_km,
        "bin_start_km": bin_start,
        "low_km": low_km,
        "high_km": high_km,
        "confidence": confidence,
        "status": status,
    }


def read_pieces_stream(fpath, chunk=256):
    """Yield waveforms, original piece bytes, timestamps, and piece indices."""
    fsize = os.path.getsize(fpath)
    n_pieces = (fsize - 112) // 32208
    with open(fpath, 'rb') as f:
        batch_wf, batch_raw, batch_ts, batch_indices = [], [], [], []
        for pi in range(n_pieces):
            f.seek(112 + pi * 32208)
            raw_piece = f.read(32208)
            if len(raw_piece) != 32208:
                raise ValueError(f"Incomplete piece {pi} in {fpath}")
            wf = np.frombuffer(
                raw_piece, dtype=np.uint16, count=16000, offset=208
            ).astype(np.float32)
            year, month, day, hour, minute, second = struct.unpack_from(
                "<6i4x", raw_piece, 108
            )
            sec_frac = struct.unpack_from("<d", raw_piece, 136)[0]
            fraction_text = f"{sec_frac:.10f}"[1:]
            ts = (
                f"{year % 100:02d}{month:02d}{day:02d}{hour:02d}"
                f"{minute:02d}{second:02d}{fraction_text}"
            )
            batch_wf.append(wf)
            batch_raw.append(raw_piece)
            batch_ts.append(ts)
            batch_indices.append(pi)
            if len(batch_wf) >= chunk:
                yield batch_wf, batch_raw, batch_ts, batch_indices
                batch_wf, batch_raw, batch_ts, batch_indices = [], [], [], []
        if batch_wf:
            yield batch_wf, batch_raw, batch_ts, batch_indices


def write_lig(raw_pieces, outpath, lig_header):
    """Regroup original piece bytes without rewriting their metadata."""
    outpath = os.fspath(outpath)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    idx = 0
    base = outpath.rsplit('.', 1)[0]
    while idx < len(raw_pieces):
        chunk = raw_pieces[idx:idx + MAX_PER_FILE]
        fname = f"{base}.lig" if idx == 0 else f"{base}_{idx//MAX_PER_FILE+1}.lig"
        with open(fname, 'wb') as f:
            gh = bytearray(lig_header[:112])
            if len(gh) < 112:
                gh.extend(b"\x00" * (112 - len(gh)))
            struct.pack_into('<i', gh, 4, len(chunk))
            f.write(gh)
            for raw_piece in chunk:
                if len(raw_piece) != 32208:
                    raise ValueError("Each raw piece must contain exactly 32208 bytes")
                f.write(raw_piece)
        idx += MAX_PER_FILE


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Classify .lig pieces with the validated hybrid model route."
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--date", help="Optional YYMMDD filename filter, e.g. 210413")
    p.add_argument("--old_model", default="./weights/old/model.pt")
    p.add_argument("--new_model", default="./weights/new/model.pt")
    p.add_argument("--type_only", action="store_true")
    p.add_argument("--model", default="./weights/fresh/candidate.pt")
    p.add_argument("--min_type_confidence", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--keep_uncertain_in_type_bin",
        action="store_true",
        help="Keep uncertain pieces in their estimated bin instead of UNCERTAIN_<TYPE>",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    if not 0.0 <= args.min_type_confidence <= 1.0:
        raise ValueError("--min_type_confidence must be between 0 and 1")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if args.type_only:
        type_model, type_ckpt = load_mtl_checkpoint(args.model, dev)
        validate_type_only_options(type_ckpt, args.min_type_confidence)
        inference_ckpt = type_ckpt
    else:
        old_model, old_ckpt = load_mtl_checkpoint(args.old_model, dev)
        new_model, new_ckpt = load_mtl_checkpoint(args.new_model, dev)
        validate_checkpoint_pair(old_ckpt, new_ckpt)
        inference_ckpt = old_ckpt

    files = []
    for root, _, names in os.walk(args.input_dir):
        for name in names:
            if name.lower().endswith(".lig") and (
                not args.date or args.date in name
            ):
                files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError("No matching .lig files were found")
    print(f"Found {len(files)} files")
    with open(sorted(files)[0], "rb") as handle:
        lh = handle.read(112)

    # Cache per output class
    os.makedirs(args.output_dir, exist_ok=True)
    cache = {}
    total, processed = 0, 0
    csv_writer = PredictionCsvWriter(
        os.path.join(args.output_dir, "predictions.csv")
    )
    csv_writer.__enter__()

    try:
        for fp in sorted(files):
            for batch_wf, batch_raw, batch_ts, batch_indices in read_pieces_stream(
                fp, args.batch_size
            ):
                if not batch_wf:
                    continue
                normalize_mode = inference_ckpt.get('preprocessing', {}).get('normalize_mode', 'minmax')
                wf_pp = preprocess_batch(np.stack(batch_wf), normalize_mode=normalize_mode)
                x = torch.from_numpy(wf_pp).unsqueeze(1).to(dev)
                with torch.no_grad():
                    if args.type_only:
                        if checkpoint_schema(type_ckpt) == "four_class_rejection_v1":
                            features = type_model.extract_type_features(x)
                            type_logits = type_model.type_head(features)
                            predictions = decode_four_class_predictions(
                                type_logits, features, type_ckpt
                            )
                        else:
                            type_logits = type_model.forward_type(x)
                            predictions = [
                                decode_type_prediction(
                                    row,
                                    type_ckpt,
                                    min_confidence=args.min_type_confidence,
                                )
                                for row in type_logits
                            ]
                    else:
                        old_type_logits, old_distance_logits = old_model(x)
                        predicted_types = old_type_logits.argmax(dim=1)
                        new_positions = torch.nonzero(
                            predicted_types > 1, as_tuple=False
                        ).flatten()
                        new_by_position = {}
                        if len(new_positions):
                            _, selected_logits = new_model(x[new_positions])
                            for row, position in enumerate(new_positions.tolist()):
                                new_by_position[position] = [
                                    head[row] for head in selected_logits
                                ]
                        predictions = [
                            decode_hybrid_prediction(
                                old_type_logits[i],
                                [head[i] for head in old_distance_logits],
                                new_by_position.get(
                                    i, [head[i] for head in old_distance_logits]
                                ),
                                old_ckpt,
                                new_ckpt,
                            )
                            for i in range(len(batch_wf))
                        ]

                for i in range(len(batch_wf)):
                    prediction = predictions[i]
                    cls_out = prediction["class_name"]
                    if (
                        args.keep_uncertain_in_type_bin
                        and prediction["status"] == "uncertain"
                    ):
                        lo = prediction["bin_start_km"]
                        cls_out = f'{prediction["type"]}_{lo}-{lo + 100}km'
                    csv_writer.write({
                        "source_file": fp,
                        "piece_index": batch_indices[i],
                        "type": prediction["type"],
                        "distance_km": prediction["distance_km"],
                        "bin_start_km": prediction["bin_start_km"],
                        "low_km": prediction["low_km"],
                        "high_km": prediction["high_km"],
                        "confidence": prediction["confidence"],
                        "type_confidence": prediction["type_confidence"],
                        "raw_type": prediction.get("raw_type", prediction["type"]),
                        "type_margin": prediction.get("type_margin"),
                        "feature_distance": prediction.get("feature_distance"),
                        "rejection_reason": prediction.get("rejection_reason"),
                        "final_type": prediction.get("final_type", prediction["type"]),
                        "head_source": prediction["head_source"],
                        "status": prediction["status"],
                    })

                    if cls_out not in cache:
                        cache[cls_out] = []
                    cache[cls_out].append((batch_raw[i], batch_ts[i]))

                    if len(cache[cls_out]) >= MAX_PER_FILE:
                        chunk = cache[cls_out][:MAX_PER_FILE]
                        path = os.path.join(args.output_dir, cls_out,
                                           f"GZ_{batch_ts[i].replace('.','')}.lig")
                        write_lig([piece for piece, _ in chunk], path, lh)
                        cache[cls_out] = cache[cls_out][MAX_PER_FILE:]

                    processed += 1
                    if processed % 10000 == 0:
                        print(f"  Processed {processed} pieces...")
    finally:
        csv_writer.__exit__(None, None, None)

    # Flush remaining
    for cls_out, pieces in cache.items():
        if pieces:
            ts = pieces[0][1]
            path = os.path.join(args.output_dir, cls_out,
                               f"GZ_{ts.replace('.','')}.lig")
            write_lig([piece for piece, _ in pieces], path, lh)

    print(f"\nDone. {processed} pieces classified into {len(cache)} categories.")
    for cls_out in sorted(os.listdir(args.output_dir)):
        d = os.path.join(args.output_dir, cls_out)
        if os.path.isdir(d):
            n = len(os.listdir(d))
            print(f"  {cls_out}: {n} files")


if __name__ == "__main__":
    main()
