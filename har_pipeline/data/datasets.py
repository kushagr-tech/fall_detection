"""
data/datasets.py
================
Registry of all known public HAR / fall-detection datasets.

For every dataset this module stores:
  - How to download it (URL + license)
  - Sensor specification (axes, units, sampling rate, coordinate convention)
  - Subject metadata (count, age range, gender split)
  - Label taxonomy (original labels → description)
  - Recording protocol (phone positions, orientations, sessions)
  - Data quality notes

The adapter for each dataset lives in data/adapters/<dataset_id>.py.
The harmonizer (data/harmonizer.py) uses the metadata here to align axes,
rescale units, resample, and map labels before merging.

License notes
-------------
ALL datasets listed here are publicly available for academic research.
Before using any dataset, verify the license for your specific use case.
Commercial use may require separate agreements with the dataset authors.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SensorSpec:
    sampling_rate_hz: int
    acc_unit: str           # "m/s2" | "g"
    gyr_unit: str           # "rad/s" | "deg/s"
    acc_range_g: float      # ± g full-scale
    gyr_range_dps: float    # ± degrees/s full-scale
    has_gyroscope: bool
    # Axis convention relative to our canonical frame (phone portrait, screen up):
    # Our canonical: ax=right, ay=up, az=out-of-screen
    # Record any rotation/flip needed to align source axes to canonical
    axis_remap: str         # e.g. "identity" | "x→-x,z→-z" | "transpose_xz"


@dataclass
class PhonePosition:
    name: str
    description: str


@dataclass
class DatasetMetadata:
    dataset_id: str
    full_name: str
    year: int
    n_subjects: int
    age_range: tuple[int, int]   # (min, max)
    gender_notes: str
    n_recordings: int
    total_duration_minutes: float
    sensor_spec: SensorSpec
    phone_positions: list[PhonePosition]
    original_labels: dict[str, str]      # code → description
    unified_label_map: dict[str, str]    # original_code → our unified label
    download_url: str
    license: str
    citation: str
    notes: str
    # Whether the raw data has been downloaded (updated at runtime)
    available: bool = False


# ── Canonical (unified) label set ─────────────────────────────────────────────

UNIFIED_LABELS = {
    # Activities of Daily Living
    "standing":  "Standing still",
    "walking":   "Level walking (any speed)",
    "running":   "Running / jogging",
    "sitting":   "Sitting on chair/floor",
    "lying":     "Lying down",
    # Transitional
    "stairs_up":   "Going up stairs",
    "stairs_down": "Going down stairs",
    "transition":  "Posture transition (sit↔stand, lie↔stand)",
    # Falls
    "falling":   "Fall event (any direction)",
    # Meta
    "unknown":   "Unknown / undefined",
}

# Alias map — extra synonyms that adapters may use
LABEL_ALIASES: dict[str, str] = {
    "fall":         "falling",
    "fall_forward": "falling",
    "fall_backward":"falling",
    "fall_left":    "falling",
    "fall_right":   "falling",
    "fall_lateral": "falling",
    "stumble":      "falling",
    "walk":         "walking",
    "slow_walk":    "walking",
    "fast_walk":    "walking",
    "jog":          "running",
    "run":          "running",
    "sit":          "sitting",
    "sit_chair":    "sitting",
    "sit_floor":    "sitting",
    "lay":          "lying",
    "lie":          "lying",
    "lie_down":     "lying",
    "stand":        "standing",
    "stand_up":     "transition",
    "sit_down":     "transition",
    "stair_up":     "stairs_up",
    "stair_down":   "stairs_down",
    "upstairs":     "stairs_up",
    "downstairs":   "stairs_down",
}


# ── Dataset definitions ────────────────────────────────────────────────────────

UNIMIB_SHAR = DatasetMetadata(
    dataset_id   = "unimib_shar",
    full_name    = "UniMiB SHAR — University of Milano-Bicocca Smartphone-based HAR",
    year         = 2017,
    n_subjects   = 30,
    age_range    = (18, 60),
    gender_notes = "18 female, 12 male",
    n_recordings = 11771,   # labelled segments
    total_duration_minutes = 180.0,
    sensor_spec  = SensorSpec(
        sampling_rate_hz = 50,
        acc_unit         = "m/s2",
        gyr_unit         = "n/a",   # no gyroscope
        acc_range_g      = 2.0,
        gyr_range_dps    = 0.0,
        has_gyroscope    = False,
        axis_remap       = "identity",
    ),
    phone_positions = [
        PhonePosition("trouser_front_pocket", "Trouser front pocket, screen facing thigh"),
    ],
    original_labels = {
        # Falls (9 types)
        "FKL": "Fall forward while walking",
        "BSC": "Fall backwards while sitting on chair",
        "SDL": "Fall forward while sitting in a lateral position",
        "LFW": "Fall forward while walking, lateral",
        "FOL": "Fall forward from standing",
        "BFAR": "Fall backward from a reclining position",
        "BARD": "Fall backward while running",
        "FPSA": "Fall forward from sitting, arms out",
        "SBW": "Stumble backward while walking",
        # ADL (9 types)
        "StdUP": "Standing up from sitting",
        "SitDN": "Sitting down",
        "WAL":   "Walking",
        "JOG":   "Jogging",
        "JUM":   "Jumping",
        "GPK":   "Getting in/out of car",
        "SCH":   "Going up/down stairs",
        "STD":   "Standing still",
        "LIE":   "Lying down",
    },
    unified_label_map = {
        "FKL":   "falling", "BSC":   "falling", "SDL":   "falling",
        "LFW":   "falling", "FOL":   "falling", "BFAR":  "falling",
        "BARD":  "falling", "FPSA":  "falling", "SBW":   "falling",
        "StdUP": "transition", "SitDN": "transition",
        "WAL":   "walking", "JOG":  "running", "JUM":   "running",
        "GPK":   "transition", "SCH":  "stairs_up",
        "STD":   "standing", "LIE":  "lying",
    },
    download_url = "https://www.sal.disco.unimib.it/technologies/unimib-shar/",
    license      = "Academic/research use. Cite: Micucci et al., Appl. Sci. 2017.",
    citation     = "Micucci D., Mobilio M., Napoletano P. (2017). UniMiB SHAR: A Dataset for Human Activity Recognition Using Acceleration Data from Smartphones. Applied Sciences, 7(10), 1101.",
    notes        = (
        "Accelerometer only (no gyroscope). Subjects wore phone in trouser front pocket. "
        "Activity segments are pre-cut (not continuous streams). "
        "Data available as .mat files; adapter converts to our CSV format. "
        "Download requires registration on the dataset webpage."
    ),
)

MOBIACT = DatasetMetadata(
    dataset_id   = "mobiact",
    full_name    = "MobiAct — Mobile Activity and Fall Dataset (v2)",
    year         = 2016,
    n_subjects   = 57,
    age_range    = (20, 47),
    gender_notes = "36 male, 21 female",
    n_recordings = 3184,
    total_duration_minutes = 600.0,
    sensor_spec  = SensorSpec(
        sampling_rate_hz = 200,
        acc_unit         = "m/s2",
        gyr_unit         = "deg/s",
        acc_range_g      = 16.0,
        gyr_range_dps    = 2000.0,
        has_gyroscope    = True,
        axis_remap       = "identity",
    ),
    phone_positions = [
        PhonePosition("trouser_right_pocket", "Right trouser pocket"),
    ],
    original_labels = {
        "STD": "Standing", "WAL": "Walking", "JOG": "Jogging",
        "JUM": "Jumping",  "STU": "Going up stairs", "STN": "Going down stairs",
        "SCH": "Sit on chair", "CSI": "Car step in", "CSO": "Car step out",
        "LYI": "Lying down",
        "FOL": "Forward fall", "FKL": "Fall while walking", "BSC": "Fall backward",
        "SDL": "Fall with lateral roll", "SFW": "Stumble forward while walking",
        "SBW": "Stumble backward while walking",
    },
    unified_label_map = {
        "STD": "standing", "WAL": "walking", "JOG": "running",
        "JUM": "running",  "STU": "stairs_up", "STN": "stairs_down",
        "SCH": "sitting",  "CSI": "transition", "CSO": "transition",
        "LYI": "lying",
        "FOL": "falling", "FKL": "falling", "BSC": "falling",
        "SDL": "falling", "SFW": "falling", "SBW": "falling",
    },
    download_url = "https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/",
    license      = "Academic/research use. Cite: Vavoulas et al., MobiCASE 2016.",
    citation     = "Vavoulas G., Chatzaki C., Malliotakis T., Tsiknakis M. (2016). The MobiAct Dataset: Recognition of Activities of Daily Living using Smartphones. ICT4AgeingWell.",
    notes        = (
        "Gyroscope sampled at 200 Hz — must resample to 50 Hz before merging. "
        "Gyroscope unit is deg/s — must convert to rad/s (÷ 57.2958). "
        "57 subjects, multiple sessions per subject. "
        "Activity segments are pre-annotated with start/end timestamps. "
        "Data provided as CSV files per subject per activity."
    ),
)

SISFALL = DatasetMetadata(
    dataset_id   = "sisfall",
    full_name    = "SisFall — A Fall and Normal Movement Dataset",
    year         = 2017,
    n_subjects   = 38,
    age_range    = (19, 75),
    gender_notes = "15 young adults (19-30), 23 older adults (60-75); mixed gender",
    n_recordings = 4505,
    total_duration_minutes = 420.0,
    sensor_spec  = SensorSpec(
        sampling_rate_hz = 200,
        acc_unit         = "g",           # raw ADC units convertible to g
        gyr_unit         = "deg/s",
        acc_range_g      = 16.0,
        gyr_range_dps    = 2000.0,
        has_gyroscope    = True,
        axis_remap       = "identity",
    ),
    phone_positions = [
        PhonePosition("waist_belt", "Waist belt / hip, facing outward"),
    ],
    original_labels = {
        # Falls (15 types)
        "F01": "Fall forward while walking", "F02": "Fall backward while walking",
        "F03": "Fall, right lateral while walking", "F04": "Fall, left lateral while walking",
        "F05": "Fall forward while standing", "F06": "Fall backward while standing",
        "F07": "Fall, right lateral while standing", "F08": "Fall, left lateral while standing",
        "F09": "Fall, sitting and hitting table", "F10": "Fall from sitting to floor, forward",
        "F11": "Fall from sitting to floor, lateral", "F12": "Fall backward with chair",
        "F13": "Stumble forward", "F14": "Fall on both knees",
        "F15": "Fall forward, hitting table while walking",
        # ADL (19 types)
        "D01": "Walking slowly", "D02": "Walking at normal speed", "D03": "Walking quickly",
        "D04": "Jogging slowly", "D05": "Jogging quickly",
        "D06": "Walking upstairs slowly", "D07": "Walking upstairs quickly",
        "D08": "Walking downstairs slowly", "D09": "Walking downstairs quickly",
        "D10": "Standing up from sitting on a bed", "D11": "Sitting on a bed",
        "D12": "Sitting on a chair", "D13": "Sitting in a car",
        "D14": "Getting up from a chair", "D15": "Getting out of a car",
        "D16": "Bending over to pick up an object", "D17": "Drinking water",
        "D18": "Combing hair", "D19": "Lying down on a bed",
    },
    unified_label_map = {
        "F01": "falling", "F02": "falling", "F03": "falling", "F04": "falling",
        "F05": "falling", "F06": "falling", "F07": "falling", "F08": "falling",
        "F09": "falling", "F10": "falling", "F11": "falling", "F12": "falling",
        "F13": "falling", "F14": "falling", "F15": "falling",
        "D01": "walking",  "D02": "walking",  "D03": "walking",
        "D04": "running",  "D05": "running",
        "D06": "stairs_up", "D07": "stairs_up",
        "D08": "stairs_down", "D09": "stairs_down",
        "D10": "transition", "D11": "lying", "D12": "sitting",
        "D13": "sitting",  "D14": "transition", "D15": "transition",
        "D16": "transition", "D17": "standing", "D18": "standing",
        "D19": "lying",
    },
    download_url = "http://sistemic.udea.edu.co/en/research/projects/english-falls/",
    license      = "Academic/research use. Cite: Sucerquia et al., Sensors 2017.",
    citation     = "Sucerquia A., López J.D., Vargas-Bonilla J.F. (2017). SisFall: A Fall and Normal Movement Dataset. Sensors, 17(1), 198.",
    notes        = (
        "Waist-mounted device — different from phone-in-pocket. "
        "Accelerometer unit in raw ADC (±16g scale): multiply by 16/32768 to get g, then by 9.81 for m/s². "
        "Gyroscope in deg/s: divide by 57.2958. "
        "Sampling rate 200 Hz — resample to 50 Hz. "
        "Includes older adults (60-75) — critical for fall-detection diversity. "
        "15 distinct fall types with different directions and surfaces."
    ),
)

KFALL = DatasetMetadata(
    dataset_id   = "kfall",
    full_name    = "KFall — A Fall Detection Dataset with Wrist and Waist Sensors",
    year         = 2021,
    n_subjects   = 36,
    age_range    = (18, 70),
    gender_notes = "Mixed; includes elderly participants",
    n_recordings = 2160,
    total_duration_minutes = 360.0,
    sensor_spec  = SensorSpec(
        sampling_rate_hz = 100,
        acc_unit         = "m/s2",
        gyr_unit         = "rad/s",
        acc_range_g      = 8.0,
        gyr_range_dps    = 573.0,   # ≈ 10 rad/s
        has_gyroscope    = True,
        axis_remap       = "identity",
    ),
    phone_positions = [
        PhonePosition("waist",  "Waist, clipped to belt"),
        PhonePosition("wrist",  "Wrist, dominant hand"),
    ],
    original_labels = {
        "ADL01": "Standing still",   "ADL02": "Sitting on a chair",
        "ADL03": "Walking",          "ADL04": "Jogging",
        "ADL05": "Going upstairs",   "ADL06": "Going downstairs",
        "ADL07": "Lying down",       "ADL08": "Sit-to-stand",
        "ADL09": "Stand-to-sit",     "ADL10": "Lying-to-sitting",
        "FALL01": "Forward fall",    "FALL02": "Backward fall",
        "FALL03": "Left lateral fall", "FALL04": "Right lateral fall",
        "FALL05": "Stumble forward", "FALL06": "Stumble backward",
    },
    unified_label_map = {
        "ADL01": "standing",  "ADL02": "sitting",   "ADL03": "walking",
        "ADL04": "running",   "ADL05": "stairs_up",  "ADL06": "stairs_down",
        "ADL07": "lying",     "ADL08": "transition", "ADL09": "transition",
        "ADL10": "transition",
        "FALL01": "falling",  "FALL02": "falling",   "FALL03": "falling",
        "FALL04": "falling",  "FALL05": "falling",   "FALL06": "falling",
    },
    download_url = "https://github.com/ylysa/KFall-Dataset",
    license      = "CC BY 4.0 — free for academic and commercial use with attribution.",
    citation     = "Yeh C.-Y., Su H.-Y., Lee S.-M. (2021). KFall: An Open Dataset for Falls and Activities of Daily Living Using Inertial Sensors. MDPI Sensors.",
    notes        = (
        "CC BY 4.0 license — most permissive of all listed datasets. "
        "Two sensor positions (waist + wrist) per recording. "
        "100 Hz — resample to 50 Hz. "
        "Gyroscope already in rad/s (no unit conversion needed). "
        "Includes elderly participants for age diversity."
    ),
)

# ── Master registry ────────────────────────────────────────────────────────────

ALL_DATASETS: dict[str, DatasetMetadata] = {
    d.dataset_id: d
    for d in [UNIMIB_SHAR, MOBIACT, SISFALL, KFALL]
}


def get_dataset(dataset_id: str) -> DatasetMetadata:
    if dataset_id not in ALL_DATASETS:
        raise KeyError(f"Unknown dataset '{dataset_id}'. Available: {list(ALL_DATASETS)}")
    return ALL_DATASETS[dataset_id]


def map_label(dataset_id: str, original_label: str) -> str:
    """
    Map an original dataset-specific label to our unified label set.
    Falls back to LABEL_ALIASES, then 'unknown'.
    """
    meta = get_dataset(dataset_id)
    if original_label in meta.unified_label_map:
        return meta.unified_label_map[original_label]
    # Try lowercase alias lookup
    lower = original_label.lower().strip()
    if lower in LABEL_ALIASES:
        return LABEL_ALIASES[lower]
    return "unknown"
