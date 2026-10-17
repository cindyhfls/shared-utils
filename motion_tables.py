"""Read persistent per-subject frame counts; never consult a scratch cache."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_retained_frames(table_path, subject_ids):
    """Return validated counts in exactly the requested subject order."""
    path = Path(table_path)
    if not path.is_file():
        raise FileNotFoundError(f'Retained-frame table missing: {path}. Export it before running variance partitioning.')
    table = pd.read_csv(path, dtype={'participant_id': str})
    required = {'participant_id', 'n_retained_frames', 'n_total_frames'}
    if not required.issubset(table):
        raise ValueError(f'{path}: missing columns {sorted(required - set(table))}')
    if table.participant_id.isna().any() or table.participant_id.duplicated().any():
        raise ValueError(f'{path}: missing or duplicate participant IDs')
    table = table.set_index('participant_id')
    ids = list(map(str, subject_ids))
    missing = sorted(set(ids) - set(table.index))
    if missing:
        raise ValueError(f'{path}: {len(missing)} requested subjects missing, e.g. {missing[:5]}')
    selected = table.loc[ids]
    counts = pd.to_numeric(selected.n_retained_frames, errors='raise')
    totals = pd.to_numeric(selected.n_total_frames, errors='raise')
    if not (np.isfinite(counts).all() and np.isfinite(totals).all()
            and (totals > 0).all() and (counts >= 0).all() and (counts <= totals).all()
            and (counts == np.floor(counts)).all() and (totals == np.floor(totals)).all()):
        raise ValueError(f'{path}: invalid retained/total frame counts')
    return counts.astype(np.int64)


def read_subject_subset(path):
    """Read an explicit cohort without deriving or silently filtering membership."""
    ids = Path(path).read_text().split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'{path}: empty subset or duplicate IDs')
    return ids


def subset_output_tag(subject_list, default_list):
    """Separate nondefault cohorts, including different ID lists of the same size."""
    import hashlib
    ids = sorted(read_subject_subset(subject_list))
    if ids == sorted(read_subject_subset(default_list)):
        return ''
    return '_subset' + hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:12]


def load_temporal_overlap(mask_path, subject_ids, expected_counts=None):
    """Return shared retained-frame counts in scipy condensed-pair order.

    NPZ must contain sids/tmask or subject_ids/retained_masks, with a binary
    (subject, time) mask on one common
    temporal grid. Integer multiplication counts intersection, not Jaccard
    overlap or average frame count. Reorders by IDs before forming pairs.
    """
    with np.load(mask_path, allow_pickle=False) as data:
        ids = data['sids' if 'sids' in data else 'subject_ids'].astype(str)
        masks = data['tmask' if 'tmask' in data else 'retained_masks']
    if ids.ndim != 1 or len(set(ids)) != len(ids):
        raise ValueError('Temporal-mask IDs must be unique and one-dimensional')
    if masks.ndim != 2 or masks.shape[0] != len(ids) or masks.shape[1] == 0:
        raise ValueError('Temporal masks must have shape (subjects, time)')
    if not np.isin(masks, [0, 1]).all():
        raise ValueError('Temporal masks must be binary')
    lookup = {sid: i for i, sid in enumerate(ids)}
    requested = list(map(str, subject_ids))
    if len(set(requested)) != len(requested) or not set(requested).issubset(lookup):
        raise ValueError('Duplicate or missing requested temporal-mask IDs')
    selected = masks[[lookup[sid] for sid in requested]].astype(np.int64)
    if expected_counts is not None and not np.array_equal(selected.sum(axis=1), np.asarray(expected_counts)):
        raise ValueError('Temporal masks do not match retained-frame counts')
    overlap = selected @ selected.T
    return overlap[np.triu_indices(len(requested), k=1)].astype(float)
