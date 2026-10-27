"""Glasser-360 (HCP-MMP1) parcels and the Sydnor sensorimotor-association (S-A) axis, in one region order.

Region order everywhere in this module: index 0-179 = left-hemisphere parcels 1-180, index 180-359 =
right-hemisphere parcels 1-180. Only numpy/pandas are needed at import, so envs without neuroboros
(e.g. `ann` for the layer sweep) can use it; neuroboros is imported only to compute indices for a
space other than onavg-ico32.

    from shared_utils.atlas_utils import load_glasser360_region_indices, load_glasser360_table
    region_indices = load_glasser360_region_indices()   # 360 arrays into the 19341-vertex space
    table = load_glasser360_table()                     # label, cortex (22 groups), lobe, sa_rank, ...
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GLASSER360_DIR = os.path.join(HERE, "support_files", "Parcellation-Glasser360")
GLASSER360_INDICES_NPZ = os.path.join(GLASSER360_DIR, "glasser360_onavg-ico32_indices.npz")
GLASSER360_REGION_CSV = os.path.join(GLASSER360_DIR, "HCP-MMP1_UniqueRegionList.csv")
SA_AXIS_CSV = os.path.join(HERE, "support_files", "SensorimotorAssociationAxis-Sydnor2021",
                           "Sensorimotor_Association_Axis_AverageRanks.csv")
NB_DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/jiaxin/nb-data"

N_GLASSER360_PER_HEMI = 180
N_GLASSER360 = 2 * N_GLASSER360_PER_HEMI

# Read from the CIFTI map names of Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Final_Final_Areas
# .32k_fs_LR.dscalar.nii ("L_V1_ROI" -> "V1"); row order verified to equal parcel numbering 1-180 by
# reproducing the lh overlap-8div_parc.npy exactly. Right-hemisphere names assumed identical (HCP-MMP1
# areas are symmetric L/R pairs), not checked against the rh dscalar.
GLASSER360_BASE_LABELS = [
    "V1", "MST", "V6", "V2", "V3", "V4", "V8", "4", "3b", "FEF",
    "PEF", "55b", "V3A", "RSC", "POS2", "V7", "IPS1", "FFC", "V3B", "LO1",
    "LO2", "PIT", "MT", "A1", "PSL", "SFL", "PCV", "STV", "7Pm", "7m",
    "POS1", "23d", "v23ab", "d23ab", "31pv", "5m", "5mv", "23c", "5L", "24dd",
    "24dv", "7AL", "SCEF", "6ma", "7Am", "7PL", "7PC", "LIPv", "VIP", "MIP",
    "1", "2", "3a", "6d", "6mp", "6v", "p24pr", "33pr", "a24pr", "p32pr",
    "a24", "d32", "8BM", "p32", "10r", "47m", "8Av", "8Ad", "9m", "8BL",
    "9p", "10d", "8C", "44", "45", "47l", "a47r", "6r", "IFJa", "IFJp",
    "IFSp", "IFSa", "p9-46v", "46", "a9-46v", "9-46d", "9a", "10v", "a10p", "10pp",
    "11l", "13l", "OFC", "47s", "LIPd", "6a", "i6-8", "s6-8", "43", "OP4",
    "OP1", "OP2-3", "52", "RI", "PFcm", "PoI2", "TA2", "FOP4", "MI", "Pir",
    "AVI", "AAIC", "FOP1", "FOP3", "FOP2", "PFt", "AIP", "EC", "PreS", "H",
    "ProS", "PeEc", "STGa", "PBelt", "A5", "PHA1", "PHA3", "STSda", "STSdp", "STSvp",
    "TGd", "TE1a", "TE1p", "TE2a", "TF", "TE2p", "PHT", "PH", "TPOJ1", "TPOJ2",
    "TPOJ3", "DVT", "PGp", "IP2", "IP1", "IP0", "PFop", "PF", "PFm", "PGi",
    "PGs", "V6A", "VMV1", "VMV3", "PHA2", "V4t", "FST", "V3CD", "LO3", "VMV2",
    "31pd", "31a", "VVC", "25", "s32", "pOFC", "PoI1", "Ig", "FOP5", "p10p",
    "p47r", "TGv", "MBelt", "LBelt", "A4", "STSva", "TE1m", "PI", "a32pr", "p24",
]
GLASSER360_LABELS = ([f"L_{n}" for n in GLASSER360_BASE_LABELS]
                     + [f"R_{n}" for n in GLASSER360_BASE_LABELS])


def load_glasser360_region_indices(source_space="onavg-ico32"):
    """360 vertex-index arrays into the masked L-then-R vertex space of `source_space`.

    onavg-ico32 (19341 vertices, e.g. dms_lr_float32.npy) reads the precomputed npz; any other space
    is computed with neuroboros (compute_glasser360_region_indices).
    """
    if source_space != "onavg-ico32":
        return compute_glasser360_region_indices(source_space)
    with np.load(GLASSER360_INDICES_NPZ) as data:
        return [data[f"region_{i:03d}"] for i in range(N_GLASSER360)]


def compute_glasser360_region_indices(source_space="onavg-ico32", nb_data_root=NB_DATA_ROOT):
    """Q1-Q6_RelatedParcellation210 parcels remapped into the official cortex mask (nb.mask).

    The parcellation arrays are per-hemisphere-local and use HCP-MMP's own medial wall (label 0/-1);
    data live in nb.mask's L-then-R space, whose medial wall differs slightly. Right-hemisphere indices
    are offset by the left cortex size, and parcel vertices outside the mask are dropped.
    """
    import neuroboros as nb
    region_indices = []
    offset = 0
    for hemi in ("l", "r"):
        mask = nb.mask(hemi, space=source_space).astype(bool)
        parc = np.load(os.path.join(nb_data_root, "core", source_space, "parcellations", "HCP_MMP",
                                    f"{hemi}h", "Q1-Q6_RelatedParcellation210", "overlap-8div_parc.npy"))
        remap = np.full(mask.shape[0], -1, dtype=np.int64)
        remap[mask] = offset + np.arange(int(mask.sum()))
        for iparc in range(1, N_GLASSER360_PER_HEMI + 1):
            mapped = remap[np.flatnonzero(parc == iparc)]
            region_indices.append(mapped[mapped >= 0])
        offset += int(mask.sum())
    return region_indices


def save_glasser360_region_indices(out_path=GLASSER360_INDICES_NPZ):
    """Regenerate the onavg-ico32 npz (needs the nb env); only if nb-data's parcellation or mask changes."""
    region_indices = compute_glasser360_region_indices("onavg-ico32")
    np.savez(out_path, **{f"region_{i:03d}": arr for i, arr in enumerate(region_indices)})
    return out_path


def load_glasser360_table():
    """One row per region in region-index order (0-359).

    Columns: label ("L_V1"), hemi, region ("V1"), long_name, lobe, cortex_id (1-22) and cortex (the 22
    HCP-MMP1 groupings), x/y/z_cog, volmm, sa_rank and sa_avg_rank (S-A axis, see load_sa_axis_ranks).
    """
    csv = pd.read_csv(GLASSER360_REGION_CSV, encoding="utf-8-sig")
    # The CSV names regions "V1_L" and writes 7PL as "7Pl"; check it is positionally in label order.
    csv_labels = csv.LR + "_" + csv.region
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(csv_labels, GLASSER360_LABELS)) if a.lower() != b.lower()]
    if len(csv) != N_GLASSER360 or bad:
        raise ValueError(f"{GLASSER360_REGION_CSV}: rows not in Glasser-360 label order, e.g. {bad[:5]}")
    if (csv.groupby("Cortex_ID").cortex.nunique() != 1).any():
        raise ValueError(f"{GLASSER360_REGION_CSV}: a Cortex_ID has more than one cortex name")
    sa = _load_sa_axis_csv()
    table = pd.DataFrame({
        "label": GLASSER360_LABELS,
        "hemi": csv.LR.values,
        "region": GLASSER360_BASE_LABELS * 2,
        "long_name": csv.regionLongName.values,
        "lobe": csv.Lobe.values,
        "cortex_id": csv.Cortex_ID.values.astype(int),
        "cortex": csv.cortex.values,
        "x_cog": csv["x-cog"].values,
        "y_cog": csv["y-cog"].values,
        "z_cog": csv["z-cog"].values,
        "volmm": csv.volmm.values,
        "sa_rank": np.tile(sa["final.rank"].values, 2),
        "sa_avg_rank": np.tile(sa["brains.average.rank"].values, 2),
    })
    table.index.name = "region_idx"
    return table


def load_sa_axis_ranks(column="final.rank"):
    """S-A axis rank per region (360, region order); both hemispheres get the same bilateral rank.

    Sydnor et al. 2021 (Neuron), Glasser360_MMP/Sensorimotor_Association_Axis_AverageRanks.csv:
    "final.rank" is 1 (most sensorimotor) to 180 (most association), ties averaged; "brains.average.rank"
    is the mean rank over the source maps before re-ranking.
    """
    return np.tile(_load_sa_axis_csv()[column].values, 2)


def _load_sa_axis_csv():
    sa = pd.read_csv(SA_AXIS_CSV)
    if sa.region.tolist() != GLASSER360_BASE_LABELS:
        raise ValueError(f"{SA_AXIS_CSV}: regions not in Glasser-360 parcel order")
    return sa
