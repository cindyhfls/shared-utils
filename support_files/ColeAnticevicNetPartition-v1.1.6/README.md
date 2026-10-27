# Cole-Anticevic Brain-wide Network Partition (CAB-NP), v1.1.6

From https://github.com/ColeLab/ColeAnticevicNetPartition at commit
`e4ea9fd709ead8616843924b717c6abce62ea05c` (latest on 2026-10-26; README version 1.1.6, Feb 2021).
Ji et al. 2019, NeuroImage. `README_ColeLab.md` and `LICENSE` are the repo's own.

| File | Contents |
|---|---|
| `cortex_parcel_network_assignments.txt` | Network id (1-12) of each of the 360 Glasser parcels, left hemisphere first. |
| `network_labelfile.txt` | Network names and RGBA colours, keyed by id. |
| `CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt` | Parcel key; its first 360 rows map each cortical parcel to its Glasser name. `atlas_utils` checks the names and network ids against the other two files on every load. |

Load through `atlas_utils.load_glasser_groups("ColeAnticevic")`, `load_ca_network_colors()`, or the
`ca_network` column of `load_glasser360_table()`. Identical (all 360 parcels) to hcp_utils 0.1.0's
`ca_network`, which was used before 2026-10-26.
