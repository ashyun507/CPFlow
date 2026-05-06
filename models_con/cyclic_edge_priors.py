import csv
from collections import defaultdict
from pathlib import Path

import torch


CYCLIC_TYPE_TO_ID = {
    "HEADTAIL": 0,
    "ISOPEPTIDE": 1,
    "DISULFIDE": 2,
}

_TYPE_ORDER = ["HEADTAIL", "ISOPEPTIDE", "DISULFIDE"]
_NUM_BINS = 128
_BIN_MIN = 3.0
_BIN_MAX = 8.0


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _properties_dir():
    return _repo_root() / "datasets" / "cpcore" / "CPCore_properties"


def _load_terminal_priors():
    vectors = defaultdict(lambda: [0.0] * _NUM_BINS)
    tsv_path = _properties_dir() / "CPCore_Cb_distance_binned_128.tsv"
    with open(tsv_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cyclic_type = row["cyclic_type"]
            if cyclic_type not in CYCLIC_TYPE_TO_ID:
                continue
            vectors[cyclic_type][int(row["bin_index"])] = float(row["prob_mass"])
    return torch.tensor([vectors[cyclic_type] for cyclic_type in _TYPE_ORDER], dtype=torch.float32)


def _load_adjacent_priors():
    basic_path = _properties_dir() / "CPCore_Basic.tsv"
    adjacent_path = _properties_dir() / "CPCore_peptide_adjacent_cb_distances.tsv"

    sample_to_type = {}
    with open(basic_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cyclic_type = row["cyclic_type"]
            if cyclic_type not in CYCLIC_TYPE_TO_ID:
                continue
            sample_to_type[row["id"]] = cyclic_type

    counts = defaultdict(lambda: [0.0] * _NUM_BINS)
    bin_width = (_BIN_MAX - _BIN_MIN) / _NUM_BINS
    with open(adjacent_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cyclic_type = sample_to_type.get(row["sample_id"])
            if cyclic_type is None:
                continue
            cb_distance = float(row["cb_distance"])
            if cb_distance < _BIN_MIN or cb_distance > _BIN_MAX:
                continue
            bin_index = min(int((cb_distance - _BIN_MIN) / bin_width), _NUM_BINS - 1)
            counts[cyclic_type][bin_index] += 1.0

    priors = []
    for cyclic_type in _TYPE_ORDER:
        vector = torch.tensor(counts[cyclic_type], dtype=torch.float32)
        if torch.sum(vector) > 0:
            vector = vector / torch.sum(vector)
        priors.append(vector)
    return torch.stack(priors, dim=0)


CYCLIC_ADJACENT_PRIORS = _load_adjacent_priors()
CYCLIC_TERMINAL_PRIORS = _load_terminal_priors()
