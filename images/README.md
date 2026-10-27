# Glasser-360 atlas maps

Drawn on onavg inflated surfaces with the Q1-Q6_RelatedParcellation210 parcels, alpha 0.8 over sulci.

- `glasser_groups_<set>.png`: one map per set in `atlas_utils.GLASSER_GROUPS`; legend counts are
  regions (L+R), parcels in no group are uncoloured.
  - `ColeAnticevic`: ColeLab CAB-NP v1.1.6 network colours (`load_ca_network_colors()`).
  - `Horikawa-2020`, `Mentch-2026`, `dAscoli-2026`: the papers give no colours, so tab20 hues are
    used (greys dropped). These colours carry no meaning.
- `glasser360_colors.png`: every parcel in its HCP-MMP1.0 colour (`load_glasser360_colors()`, from
  hcp_utils 0.1.0's mmp_1.0.npz; not yet checked against the original HCP-MMP dlabel).
- `sa_axis_rank.png`: Sydnor 2021 S-A rank (`load_sa_axis_ranks()`) in PennLINC's brainmaps.Rmd colours
  (goldenrod1 - white - #6f1282, midpoint 90), blended in RGB rather than ggplot's Lab.

Regenerate in the `nb` env: `python shared_utils/images/plot_atlas_maps.py`.
