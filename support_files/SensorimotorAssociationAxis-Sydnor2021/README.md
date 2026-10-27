# Sensorimotor-association (S-A) axis, Glasser-360

`Sensorimotor_Association_Axis_AverageRanks.csv` from
https://github.com/PennLINC/S-A_ArchetypalAxis/tree/main/Glasser360_MMP (Sydnor et al. 2021, Neuron,
https://doi.org/10.1016/j.neuron.2021.06.016), downloaded 2026-10-20; byte-identical to commit
`edaad95a9c2eb3b5c44567907b8d4343d357651d` (latest as of 2026-10-26). The repo has no licence file; its
README asks that Sydnor et al. 2021 be cited when the S-A axis rankings are used.

180 rows, one per parcel in Glasser parcel order 1-180 (same names as `atlas_utils.GLASSER360_BASE_LABELS`);
the axis is bilateral, so both hemispheres get the same rank.

- `final.rank`: 1 (most sensorimotor) to 180 (most association); ties averaged.
- `brains.average.rank`: mean rank over the source maps, before re-ranking.

Load with `atlas_utils.load_sa_axis_ranks()` (360 values in region order) or the `sa_rank` /
`sa_avg_rank` columns of `atlas_utils.load_glasser360_table()`.
