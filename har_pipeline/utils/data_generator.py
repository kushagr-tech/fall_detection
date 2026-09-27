"""
utils/data_generator.py
=======================
Generates synthetic, physically realistic accelerometer + gyroscope CSV files
so the pipeline can be trained and evaluated with realistic activity dynamics.

Each activity models real biomechanical movement patterns:
  - standing : vertical gravitational vector with natural human postural sway
  - sitting  : tilted gravitational vector (horizontal thigh in pocket/chair)
  - lying    : horizontal gravitational vector, minimal variance
  - walking  : rhythmic gait harmonics (~1.8-2.0 Hz) on acceleration & angular velocity
  - running  : high-energy gait harmonics (~2.8-3.2 Hz) with high acceleration peaks
  - falling  : 3-phase temporal sequence: loss-of-balance / weightlessness drop,
               high-impact deceleration spike (>25 m/s²), followed by post-fall stillness.

Subject variations are modeled with individual sensor mounting angles and sensor biases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Base signal profiles per activity ──────────────────────────────────────────
# am: acceleration mean (ax, ay, az) m/s²
# as_: acceleration std (ax, ay, az)
# gm: gyroscope mean (gx, gy, gz) rad/s
# gs: gyroscope std (gx, gy, gz)
_PROFILES = {
    "standing": dict(
        am=(0.2, 0.3, 9.78),
        as_=(0.10, 0.10, 0.12),
        gm=(0.0, 0.0, 0.0),
        gs=(0.03, 0.03, 0.03),
    ),
    "sitting": dict(
        am=(0.2, 5.8, 7.8),   # Thigh horizontal: gravity shifts significantly toward Y
        as_=(0.07, 0.07, 0.08),
        gm=(0.0, 0.0, 0.0),
        gs=(0.02, 0.02, 0.02),
    ),
    "lying": dict(
        am=(9.72, 0.3, 0.5),  # Recumbent / flat: gravity predominantly on X
        as_=(0.05, 0.05, 0.05),
        gm=(0.0, 0.0, 0.0),
        gs=(0.015, 0.015, 0.015),
    ),
    "walking": dict(
        am=(0.2, 0.5, 9.75),
        as_=(1.2, 1.4, 1.8),
        gm=(0.0, 0.0, 0.0),
        gs=(0.6, 0.8, 0.7),
    ),
    "running": dict(
        am=(0.3, 0.8, 9.70),
        as_=(2.8, 3.2, 4.5),
        gm=(0.0, 0.0, 0.0),
        gs=(1.8, 2.2, 1.9),
    ),
    "falling": dict(
        am=(0.0, 0.0, 2.0),
        as_=(8.0, 8.0, 12.0),
        gm=(0.0, 0.0, 0.0),
        gs=(3.0, 3.0, 3.0),
    ),
}

# Approximate seconds per activity segment in a synthetic session
_SEGMENT_DURATIONS = {
    "standing": 25,
    "sitting":  25,
    "lying":    20,
    "walking":  30,
    "running":  20,
    "falling":  6,   # multi-phase fall
}


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_demo_data(
    output_dir: str | Path,
    n_subjects: int = 16,
    duration_s: int = 180,   # total session length per subject (seconds)
    fs: int = 50,
) -> list[Path]:
    """
    Generate one CSV file per subject in *output_dir*.

    Returns list of created file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)
    files: list[Path] = []

    for subj_idx in range(1, n_subjects + 1):
        subj_id = f"subject{subj_idx:02d}"
        path = output_dir / f"{subj_id}_session.csv"
        df = _generate_subject(subj_id, duration_s, fs, rng)
        df.to_csv(path, index=False)
        log.info("Generated %s  (%d rows)", path.name, len(df))
        files.append(path)

    return files


# ── Internal ───────────────────────────────────────────────────────────────────

def _generate_subject(
    subject: str,
    duration_s: int,
    fs: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    t_ms = int(rng.integers(1_600_000_000_000, 1_700_000_000_000))

    # Subject-specific posture offset (slight angle variation in pocket/hand)
    subj_offset_acc = rng.normal(0.0, 0.25, size=3)
    subj_offset_gyr = rng.normal(0.0, 0.015, size=3)

    activities = list(_SEGMENT_DURATIONS.keys())
    elapsed = 0

    while elapsed < duration_s:
        activity = rng.choice(activities)
        seg_s = min(_SEGMENT_DURATIONS[activity], duration_s - elapsed)
        n_rows = seg_s * fs

        if activity == "falling":
            acc, gyr = _simulate_realistic_fall(n_rows, fs, rng)
        else:
            prof = _PROFILES[activity]
            am = np.array(prof["am"]) + subj_offset_acc
            as_ = np.array(prof["as_"])
            gm = np.array(prof["gm"]) + subj_offset_gyr
            gs = np.array(prof["gs"])

            acc = rng.normal(am, as_, size=(n_rows, 3))
            gyr = rng.normal(gm, gs, size=(n_rows, 3))

            # Add low-frequency sensor drift
            drift_a = np.linspace(0, rng.uniform(-0.08, 0.08, 3), n_rows)
            drift_g = np.linspace(0, rng.uniform(-0.01, 0.01, 3), n_rows)
            acc += drift_a
            gyr += drift_g

            # Standing: add gentle postural sway (~0.2-0.4 Hz)
            if activity == "standing":
                t_vec = np.arange(n_rows) / fs
                sway_freq = rng.uniform(0.2, 0.35)
                sway = 0.15 * np.sin(2 * np.pi * sway_freq * t_vec)
                acc[:, 0] += sway
                acc[:, 1] += 0.5 * sway

            # Walking / running cadence harmonics
            elif activity in ("walking", "running"):
                freq = rng.uniform(1.8, 2.1) if activity == "walking" else rng.uniform(2.8, 3.2)
                amp_v = 1.4 if activity == "walking" else 3.2
                amp_ap = 0.8 if activity == "walking" else 2.0
                t_vec = np.arange(n_rows) / fs

                # Vertical acceleration bounce
                acc[:, 2] += amp_v * np.sin(2 * np.pi * freq * t_vec)
                # Antero-posterior acceleration
                acc[:, 1] += amp_ap * np.cos(2 * np.pi * freq * t_vec)
                # Rhythmic torso rotation on gyroscope
                gyr[:, 0] += 0.4 * np.sin(2 * np.pi * freq * t_vec)
                gyr[:, 2] += 0.6 * np.cos(2 * np.pi * freq * t_vec)

        for i in range(n_rows):
            rows.append(dict(
                timestamp=t_ms,
                ax=round(float(acc[i, 0]), 5),
                ay=round(float(acc[i, 1]), 5),
                az=round(float(acc[i, 2]), 5),
                gx=round(float(gyr[i, 0]), 5),
                gy=round(float(gyr[i, 1]), 5),
                gz=round(float(gyr[i, 2]), 5),
                label=activity,
            ))
            t_ms += 1000 // fs

        elapsed += seg_s

    return pd.DataFrame(rows)


def _simulate_realistic_fall(n_rows: int, fs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a physiologically realistic fall sequence matching the 4-phase protocol:
      1. Pre-fall / free-fall: acceleration drop (<5 m/s²), rapid angular velocity (>2.5 rad/s)
      2. Impact spike: massive deceleration peak (25-35 m/s²), jerk spike
      3. Post-fall stillness: lying motionless (<11.5 m/s² SVM, low gyro variance)
    """
    acc = np.zeros((n_rows, 3), dtype=np.float64)
    gyr = np.zeros((n_rows, 3), dtype=np.float64)

    # Fall timing in samples
    t_prefall = int(rng.uniform(0.4, 0.6) * fs)
    t_impact  = int(rng.uniform(0.15, 0.25) * fs)
    t_impact_start = int(rng.uniform(0.8, 1.5) * fs)
    t_impact_end = min(t_impact_start + t_impact, n_rows)

    # 1. Normal pre-fall motion (walking or standing before stumble)
    acc[:t_impact_start] = rng.normal((0.2, 0.5, 9.8), (0.4, 0.4, 0.5), size=(t_impact_start, 3))
    gyr[:t_impact_start] = rng.normal(0.0, 0.1, size=(t_impact_start, 3))

    # Free fall drop just before impact
    freefall_start = max(0, t_impact_start - t_prefall)
    acc[freefall_start:t_impact_start] = rng.normal((0.0, 0.0, 2.5), (0.8, 0.8, 0.8), size=(t_impact_start - freefall_start, 3))
    gyr[freefall_start:t_impact_start] = rng.normal((1.5, 2.2, 1.8), (0.8, 0.8, 0.8), size=(t_impact_start - freefall_start, 3))

    # 2. Impact deceleration spike (> 26 m/s²)
    impact_dir = rng.normal(0.0, 1.0, 3)
    impact_dir /= (np.linalg.norm(impact_dir) + 1e-6)
    impact_mag = rng.uniform(26.0, 36.0)

    n_imp = t_impact_end - t_impact_start
    if n_imp > 0:
        shape_factor = np.sin(np.linspace(0, np.pi, n_imp))[:, None]
        acc[t_impact_start:t_impact_end] = impact_dir * impact_mag * shape_factor + rng.normal(0.0, 1.5, size=(n_imp, 3))
        gyr[t_impact_start:t_impact_end] = rng.normal(0.0, 2.5, size=(n_imp, 3))

    # 3. Post-fall stillness (lying on ground)
    resting_dir = rng.normal(0.0, 1.0, 3)
    resting_dir /= (np.linalg.norm(resting_dir) + 1e-6)
    resting_acc = resting_dir * 9.81

    if t_impact_end < n_rows:
        n_rest = n_rows - t_impact_end
        acc[t_impact_end:] = rng.normal(resting_acc, (0.05, 0.05, 0.05), size=(n_rest, 3))
        gyr[t_impact_end:] = rng.normal(0.0, 0.015, size=(n_rest, 3))

    return acc, gyr
