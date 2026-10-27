# Glasser-360 (HCP-MMP1) parcellation

Load these through `shared_utils/atlas_utils.py`, not by path.

| File | Contents |
|---|---|
| `glasser360_onavg-ico32_indices.npz` | `region_000` ... `region_359`: vertex indices of each parcel in the 19341-vertex onavg-ico32 cortex space (`nb.mask`, left then right hemisphere). Region 0-179 = left parcels 1-180, 180-359 = right parcels 1-180. Parcels: `nb-data/core/onavg-ico32/parcellations/HCP_MMP/{lh,rh}/Q1-Q6_RelatedParcellation210/overlap-8div_parc.npy`. Regenerate with `atlas_utils.save_glasser360_region_indices()` in the `nb` env. |
| `HCP-MMP1_UniqueRegionList.csv` | One row per parcel, in the same order: names, lobe, `cortex` / `Cortex_ID` (the 22 HCP-MMP1 groupings), centre of gravity, volume. The PSL_L cortex-name typo was fixed on 2026-08-25; `7PL` is written `7Pl`. |
| `Hirokawa2020_iScience_groups.txt` | STAR Methods paragraph of Hirokawa et al. 2020 (iScience) defining 13 cortical areas (VC, TPJ, IPL, PC, STS, TE, MTC, insula, DLPFC, DMPFC, VMPFC, ACC, OFC) as sets of parcels. Source for `atlas_utils.HIROKAWA_2020_GROUPS` / `load_Hirokawa_2020_iScience_groups()`. |

Moved here from `litcoder_core/encoding/regions/` on 2026-10-26.
