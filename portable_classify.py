"""Portable Windows launcher with the bundled v3 NBE decision policy."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import classify
import numpy as np
import torch
from checkpoints import (
    load_decision_config,
    load_model_bundle,
    override_decision_config,
)


def _runtime_root() -> Path:
    """Return the PyInstaller data root or this source directory."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    return Path(frozen_root) if frozen_root else Path(__file__).resolve().parent


def _has_option(arguments: Sequence[str], name: str) -> bool:
    return name in arguments or any(
        argument.startswith(f"{name}=") for argument in arguments
    )


def portable_arguments(arguments: Sequence[str]) -> list[str]:
    """Inject bundled model, v3 policy, and NBE-only output defaults."""
    selected = list(arguments)
    root = _runtime_root()
    if not _has_option(selected, "--model_dir"):
        selected.extend(
            ["--model_dir", str(root / "weights" / "multi_model")]
        )
    if not _has_option(selected, "--decision_config"):
        selected.extend(
            [
                "--decision_config",
                str(
                    root
                    / "configs"
                    / "decision_nbe_human_202407_v3.json"
                ),
            ]
        )
    if not _has_option(selected, "--output_type"):
        selected.extend(
            ["--output_type", "NNBE", "--output_type", "PNBE"]
        )
    if not _has_option(selected, "--device"):
        selected.extend(["--device", "cpu"])
    return selected


def self_test() -> None:
    """Load bundled assets and execute one real inference pass."""
    root = _runtime_root()
    device = torch.device("cpu")
    bundle = load_model_bundle(root / "weights" / "multi_model", device)
    decision = load_decision_config(
        root / "configs" / "decision_nbe_human_202407_v3.json"
    )
    bundle = replace(
        bundle,
        type_checkpoint=override_decision_config(
            bundle.type_checkpoint, decision
        ),
    )
    predictions = classify.predict_bundle_batch(
        bundle,
        np.zeros((1, 16000), dtype=np.float32),
        [datetime(2020, 1, 1)],
        device=device,
        type_only=False,
    )
    print(
        "Self-test passed: "
        f"torch={torch.__version__}, device={device}, "
        f"prediction={predictions[0].final_type}",
        flush=True,
    )


def main(arguments: Sequence[str] | None = None) -> Path:
    """Run bundled v3 classification and print the completed audit path."""
    raw = sys.argv[1:] if arguments is None else arguments
    if raw == ["--self_test"]:
        self_test()
        return Path()
    result = classify.main(portable_arguments(raw))
    print(f"Classification complete: {result}", flush=True)
    return result


if __name__ == "__main__":
    if Path(sys.executable).stem.endswith("_GUI"):
        from portable_gui import main as gui_main

        gui_main()
    else:
        try:
            main()
        except KeyboardInterrupt:
            print(
                "Classification stopped safely; use --resume to continue.",
                flush=True,
            )
            raise SystemExit(130)
