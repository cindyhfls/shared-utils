# atlas_utils.py — sources and citations

Cite the original work for every resource you use from `atlas_utils.py`. Each data folder under
`support_files/` has its own README with download dates and exact versions.

## Parcellation

**Glasser-360 (HCP-MMP1.0)**: `load_glasser360_region_indices`, `GLASSER360_LABELS`,
`load_glasser360_table`, `load_glasser360_colors`, and every `load_glasser_groups` set.

- Glasser MF, Coalson TS, Robinson EC, et al. (2016). A multi-modal parcellation of human cerebral
  cortex. *Nature* 536, 171–178. https://doi.org/10.1038/nature18933
- Vertex indices are on the onavg-ico32 surface, from the `Q1-Q6_RelatedParcellation210` resampling in
  neuroboros' nb-data. Cite neuroboros / onavg as your project normally does.
- Which version: 210P is the 210 "parcellation" subjects. Glasser et al. (2016, Methods, cortical
  areal classifier) trained a multi-layer perceptron on 112 multi-modal features (myelin, thickness,
  task fMRI, 77 resting-state network maps, visuotopic maps, artefact maps) with the semi-automated
  210P parcellation as labels; applied it to each subject; averaged the winner-take-all individual
  parcellations within 210P and 210V ("Probabilistic maps"); argmax = group maximum probability map
  (MPM). The BALSA `Q1-Q6_RelatedParcellation210.L/R.CorticalAreas_dil_Final_Final_Areas` dscalars
  match this (values in 1/210 steps, per-vertex sum <= 1), so our parcels are the 210P MPM, not the
  semi-automated group borders used as the classifier's training labels ("Orig" on BALSA). CAB-NP
  (`ColeAnticevic`, GitHub v1.1.6) assigns networks per parcel; ColeLab's README (v1.1.3) says the
  Q1-Q6_RelatedValidation210 version was used (a BALSA scene note about "Orig" 210P refers to an older version).
  `Q1-Q6_Related420` (neuroboros' mean of 210P and 210V) and 210V give different boundaries.
- Choosing the version: every function that takes `group` accepts the full name or the short names
  `210P`, `210V`, `420` (`GLASSER360_GROUP_ALIASES`, `resolve_glasser360_group`). Only the 210P onavg-ico32
  indices are precomputed; other groups/spaces are computed with neuroboros on the fly.
- `cortex` / `cortex_id` (the 22 groupings, also `load_glasser_groups("Glasser-22")`) and `lobe` come from `HCP-MMP1_UniqueRegionList.csv`. The
  22 groupings are the ones in Glasser et al. 2016 (Supplementary Neuroanatomical Results).
  **TODO:** where the CSV itself came from is not recorded.
- Parcel colours (`load_glasser360_colors`) are the HCP-MMP1.0 colours shipped in hcp_utils 0.1.0
  (https://github.com/rmldj/hcp-utils).

## Network partition

**Cole-Anticevic Brain-wide Network Partition (CAB-NP) v1.1.6**: `load_glasser_groups("ColeAnticevic")`,
`load_ca_network_colors`, and the `ca_network` column of `load_glasser360_table`.

- Ji JL, Spronk M, Kulkarni K, Repovš G, Anticevic A, Cole MW (2019). Mapping the human brain's
  cortical-subcortical functional network organization. *NeuroImage* 185, 35–57.
  https://doi.org/10.1016/j.neuroimage.2018.10.006
- Files from https://github.com/ColeLab/ColeAnticevicNetPartition, commit `e4ea9fd`. Non-commercial
  use only; the licence is in `support_files/ColeAnticevicNetPartition-v1.1.6/LICENSE` and must stay
  with the files.

## Sensorimotor-association axis

**S-A axis ranks**: `load_sa_axis_ranks`, and the `sa_rank` / `sa_avg_rank` columns of
`load_glasser360_table`.

- Sydnor VJ, Larsen B, Bassett DS, et al. (2021). Neurodevelopment of the association cortices:
  Patterns, mechanisms, and implications for psychopathology. *Neuron* 109(18), 2820–2846.
  https://doi.org/10.1016/j.neuron.2021.06.016
- Files from https://github.com/PennLINC/S-A_ArchetypalAxis, commit `edaad95`. The repo has no licence
  file; its README asks that Sydnor et al. 2021 be cited.
- Map colours in `images/sa_axis_rank_<group>.png` follow that repo's `brainmaps.Rmd`.

## Region groups (`load_glasser_groups`)

**`"Horikawa-2020"`**
- Horikawa T, Cowen AS, Keltner D, Kamitani Y (2020). The neural representation of visually evoked
  emotion is high-dimensional, categorical, and distributed across transmodal brain regions.
  *iScience* 23, 101060. https://doi.org/10.1016/j.isci.2020.101060
- Groups from its STAR Methods, "analysis of individual cortical areas". The definition text is in
  `support_files/Parcellation-Glasser360/Horikawa2020_iScience_groups.txt`.

**`"Mentch-2026"`**
- Mentch J, Chen Y, Vanderwal T, Ghosh SS (2026). Preregistered movie-fMRI analyses reveal altered
  visual feature encoding in autism in pSTS. *eLife* (reviewed preprint).
  https://doi.org/10.7554/eLife.111008.1
- **TODO:** confirm this is the paper the groups were taken from.

**`"dAscoli-2026"`**
- d'Ascoli S, Rapin J, Benchetrit Y, Brooks T, Begany K, Raugel J, Banville HJ, King J-R (2026).
  A foundation model of vision, audition, and language for in-silico neuroscience (TRIBE v2). arXiv.
  https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
