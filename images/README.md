# Glasser-360 atlas maps

Drawn on onavg inflated surfaces, alpha 0.8 over sulci. Every file name ends in the HCP-MMP1.0 group whose parcel
boundaries are drawn: `_210P` (Q1-Q6_RelatedParcellation210, the boundaries of the precomputed
`load_glasser360_region_indices` npz) or `_210V` (Q1-Q6_RelatedValidation210, neuroboros' default). The values per
region are the same in both; only the borders differ.

- `glasser_groups_<set>_<group>.png`: one map per set in `atlas_utils.GLASSER_GROUPS`; legend counts are
  regions (L+R), parcels in no group are uncoloured.
  - `ColeAnticevic`: ColeLab CAB-NP v1.1.6 network colours (`load_ca_network_colors()`).
  - `Glasser-22`: the region colours of Glasser et al. 2016, Supplementary Neuroanatomical Results Figure 1
    (`nb.parcellation_colors("HCP_MMP", lr, networks="Glasser22")`).
  - `dAscoli-2026`: the Figure 8 legend colours of d'Ascoli et al. 2026 (ColorBrewer Set1;
    `atlas_utils.GLASSER_GROUP_COLORS`).
  - `Horikawa-2020`, `Mentch-2026`: no published colours, so tab20 hues are used (greys dropped). These colours
    carry no meaning.
- `glasser360_colors_<group>.png`: every parcel in its HCP-MMP1.0 colour (`load_glasser360_colors()`, from
  hcp_utils 0.1.0's mmp_1.0.npz; not yet checked against the original HCP-MMP dlabel).
- `sa_axis_rank_<group>.png`: Sydnor 2021 S-A rank (`load_sa_axis_ranks()`) in PennLINC's brainmaps.Rmd colours
  (goldenrod1 - white - #6f1282, midpoint 90), blended in RGB rather than ggplot's Lab.

Regenerate in the `nb` env: `python shared_utils/images/plot_atlas_maps.py [--group 210P|210V|420]` (default 210P).
