# Glasser-360 group maps

One map per set in `atlas_utils.GLASSER_GROUPS` (`glasser_groups_<set>.png`), drawn on onavg
inflated surfaces with the Q1-Q6_RelatedParcellation210 parcels; legend counts are regions (L+R).
Parcels in no group are left uncoloured.

- `ColeAnticevic`: ColeLab CAB-NP v1.1.6 network colours (`load_ca_network_colors()`).
- `Hirokawa-2020`, `Mentch-2026`, `dAscoli-2026`: the papers give no colours, so tab20 hues are used
  (greys dropped). These colours carry no meaning.

Regenerate in the `nb` env: `python shared_utils/images/plot_glasser_groups.py`.
