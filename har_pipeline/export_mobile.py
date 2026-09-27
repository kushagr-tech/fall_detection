"""
export_mobile.py
================
Convert the trained neural-network model to a TorchScript (.ptl) file and
package it together with the label-order file for deployment on Android.

Usage
-----
    # After running pipeline.py at least once:
    python export_mobile.py

    # Specify a different model checkpoint:
    python export_mobile.py --model_path outputs/models/neural_net.pt \\
                             --output_dir ../activity_collector/app/src/main/assets

Output
------
    <output_dir>/har_model.ptl      — TorchScript lite interpreter model
    <output_dir>/har_labels.txt     — one label per line, matching model output order

Android side
------------
Place both files in app/src/main/assets/.  The InferenceEngine.kt class will
load them automatically on first run.

Conversion steps (explained)
-----------------------------
1.  Load the NeuralNetModel from the .pt checkpoint.
2.  Set to eval() mode and trace with a dummy (1, 100, 6) input tensor.
    torch.jit.trace records the computation graph without Python control flow.
3.  Call _save_for_lite_interpreter() which applies operator fusion and
    removes any Python-only operators — producing a .ptl file that the
    PyTorch Mobile runtime (pytorch_android_lite) can load with no Python
    dependency at all.
4.  Write har_labels.txt listing the classes in the same alphabetical order
    the LabelEncoder used during training.

Model size
----------
The CNN+LSTM architecture has ~37 k parameters.
    FP32 .ptl:   ~1.2 MB
    INT8 .ptl:   ~0.4 MB  (quantise with torch.quantization.quantize_dynamic)
Both sizes are well within Android storage and RAM budgets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.neural_net import NeuralNetModel


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH  = Path("outputs/models/neural_net.pt")
DEFAULT_OUTPUT_DIR  = Path("../activity_collector/app/src/main/assets")
DEFAULT_WINDOW_SIZE = 100
DEFAULT_N_CHANNELS  = 6


# ── Main ───────────────────────────────────────────────────────────────────────

def export(
    model_path:  Path,
    output_dir:  Path,
    window_size: int,
    quantize:    bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading model from {model_path} …")
    nn = NeuralNetModel.load(model_path)
    nn.model.eval()

    # ── Trace ─────────────────────────────────────────────────────────────────
    dummy = torch.zeros(1, window_size, DEFAULT_N_CHANNELS)
    print(f"Tracing with input shape {list(dummy.shape)} …")

    with torch.no_grad():
        scripted = torch.jit.trace(nn.model.cpu(), dummy)

    # ── Optional INT8 quantisation ─────────────────────────────────────────────
    if quantize:
        print("Applying dynamic INT8 quantisation …")
        scripted = torch.quantization.quantize_dynamic(
            scripted,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8,
        )

    # ── Save as .ptl (lite interpreter) ───────────────────────────────────────
    ptl_path = output_dir / "har_model.ptl"
    scripted._save_for_lite_interpreter(str(ptl_path))
    size_kb  = ptl_path.stat().st_size // 1024
    print(f"Saved TorchScript model → {ptl_path}  ({size_kb} KB)")

    # ── Write label file ───────────────────────────────────────────────────────
    labels_path = output_dir / "har_labels.txt"
    labels = list(nn.label_encoder.classes_)
    labels_path.write_text("\n".join(labels) + "\n")
    print(f"Saved label file → {labels_path}")
    print(f"Label order: {labels}")

    # ── Sanity check: run a forward pass on the exported model ─────────────────
    loaded = torch.jit.load(str(ptl_path))
    loaded.eval()
    with torch.no_grad():
        out = loaded(dummy)
    assert out.shape == (1, len(labels)), \
        f"Output shape mismatch: expected (1, {len(labels)}), got {tuple(out.shape)}"
    print(f"✓ Sanity check passed — output shape {tuple(out.shape)}")

    print("\nDeploy checklist:")
    print(f"  1. Copy {ptl_path.name} to activity_collector/app/src/main/assets/")
    print(f"  2. Copy {labels_path.name} to activity_collector/app/src/main/assets/")
    print("  3. Sync Gradle and run the app.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export trained NN model to TorchScript for Android",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path",  type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--output_dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--window_size", type=int,  default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--quantize",    action="store_true",
                   help="Apply dynamic INT8 quantisation to halve model size.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(
        model_path  = args.model_path,
        output_dir  = args.output_dir,
        window_size = args.window_size,
        quantize    = args.quantize,
    )
