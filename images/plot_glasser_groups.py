"""One surface map per atlas_utils.GLASSER_GROUPS set, each group in its own colour, legend below.

Cole-Anticevic uses the ColeLab network colours; the other sets have none, so they get tab20 hues
(even then odd entries, so neighbouring groups are not paired light/dark shades; the two greys are
dropped because they vanish into the sulci shading). Run in the nb env:
    python /dartfs/rc/lab/H/HaxbyLab/jiaxin/shared_utils/images/plot_glasser_groups.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import neuroboros as nb  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the folder that holds shared_utils/
from shared_utils.atlas_utils import (  # noqa: E402
    GLASSER_GROUPS, N_GLASSER360, load_ca_network_colors, load_glasser_groups)

# the boundaries load_glasser360_region_indices was built from
PARC_KWARGS = {"group": "Q1-Q6_RelatedParcellation210"}
FIG_WIDTH = 6.0  # inches
FONTSIZE = 11
_TAB20 = [c for i, c in enumerate(plt.get_cmap("tab20").colors) if i not in (14, 15)]  # 14/15: greys
PALETTE = _TAB20[0::2] + _TAB20[1::2]


def plot_groups(name):
    groups = load_glasser_groups(name)
    colors = (load_ca_network_colors() if name == "ColeAnticevic"
              else {g: PALETTE[i] for i, g in enumerate(groups)})
    values = np.full(N_GLASSER360, np.nan)  # regions in no group stay uncoloured
    for i, idx in enumerate(groups.values()):
        values[idx] = i
    cmap = ListedColormap([colors[g] for g in groups])
    # vmin/vmax at -0.5 / n-0.5 put group i in the centre of colour bin i
    img = nb.plot(values, cmap=cmap, vmin=-0.5, vmax=len(groups) - 0.5, alpha=0.8, use_sulci=True,
                  colorbar=False, parc_kwargs=PARC_KWARGS).img
    brain_h = FIG_WIDTH * img.height / img.width
    ncol = 2
    legend_h = 0.25 * int(np.ceil(len(groups) / ncol)) + 0.15
    fig = plt.figure(figsize=(FIG_WIDTH, brain_h + legend_h))
    ax = fig.add_axes([0, legend_h / (brain_h + legend_h), 1, brain_h / (brain_h + legend_h)])
    ax.imshow(img)
    ax.axis("off")
    handles = [Patch(facecolor=colors[g], alpha=0.8, label=f"{g} ({len(idx)})") for g, idx in groups.items()]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=FONTSIZE, frameon=False)
    fn = os.path.join(HERE, f"glasser_groups_{name}.png")
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"saved {fn}")


if __name__ == "__main__":
    for name in GLASSER_GROUPS:
        plot_groups(name)
