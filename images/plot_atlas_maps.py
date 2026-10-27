"""Surface maps of the atlas_utils parcel annotations, one PNG each, into this folder.

- glasser_groups_<set>.png: one per atlas_utils.GLASSER_GROUPS set, each group in its own colour,
  legend below. Cole-Anticevic uses the ColeLab network colours; the other sets have none, so they get
  tab20 hues (even then odd entries, so neighbouring groups are not paired light/dark shades; the two
  greys are dropped because they vanish into the sulci shading).
- glasser360_colors.png: every parcel in its HCP-MMP1.0 colour (load_glasser360_colors).
- sa_axis_rank.png: Sydnor 2021 S-A axis rank with PennLINC's brainmaps.Rmd scale
  (scale_fill_gradient2 low "goldenrod1", mid "white", high "#6f1282", midpoint 90). ggplot blends
  the gradient in Lab space, matplotlib here in RGB, so mid-tones differ slightly.

Run in the nb env:
    python /dartfs/rc/lab/H/HaxbyLab/jiaxin/shared_utils/images/plot_atlas_maps.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import neuroboros as nb  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the folder that holds shared_utils/
from shared_utils.atlas_utils import (  # noqa: E402
    GLASSER_GROUPS, N_GLASSER360, load_ca_network_colors, load_glasser360_colors, load_glasser_groups,
    load_sa_axis_ranks)

# the boundaries load_glasser360_region_indices was built from
PARC_KWARGS = {"group": "Q1-Q6_RelatedParcellation210"}
FIG_WIDTH = 6.0  # inches
FONTSIZE = 11
_TAB20 = [c for i, c in enumerate(plt.get_cmap("tab20").colors) if i not in (14, 15)]  # 14/15: greys
# + the darkest tab20b shade of each of its 5 hue blocks, so Glasser-22 gets 22 distinct colours
PALETTE = _TAB20[0::2] + _TAB20[1::2] + list(plt.get_cmap("tab20b").colors[0::4])
SA_CMAP = LinearSegmentedColormap.from_list("pennlinc_sa", ["#FFC125", "white", "#6f1282"])  # goldenrod1


def brain_image(values, cmap, vmin, vmax):
    return nb.plot(values, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, use_sulci=True, colorbar=False,
                   parc_kwargs=PARC_KWARGS).img


def save_with_footer(img, footer_h, draw_footer, fn):
    """Brain image across the top, footer_h inches below it for draw_footer(fig, footer_frac)."""
    brain_h = FIG_WIDTH * img.height / img.width
    fig_h = brain_h + footer_h
    fig = plt.figure(figsize=(FIG_WIDTH, fig_h))
    ax = fig.add_axes([0, footer_h / fig_h, 1, brain_h / fig_h])
    ax.imshow(img)
    ax.axis("off")
    if draw_footer is not None:
        draw_footer(fig, footer_h / fig_h)
    fig.savefig(fn, dpi=200, pil_kwargs={"optimize": True})  # keeps Glasser-22 under the 500 KB commit hook
    plt.close(fig)
    print(f"saved {fn}")


def plot_groups(name):
    groups = load_glasser_groups(name)
    colors = (load_ca_network_colors() if name == "ColeAnticevic"
              else {g: PALETTE[i] for i, g in enumerate(groups)})
    values = np.full(N_GLASSER360, np.nan)  # regions in no group stay uncoloured
    for i, idx in enumerate(groups.values()):
        values[idx] = i
    # vmin/vmax at -0.5 / n-0.5 put group i in the centre of colour bin i
    img = brain_image(values, ListedColormap([colors[g] for g in groups]), -0.5, len(groups) - 0.5)
    labels = {g: f"{g.replace('_', ' ')} ({len(idx)})" for g, idx in groups.items()}
    handles = [Patch(facecolor=colors[g], alpha=0.8, label=labels[g]) for g in groups]
    fn = os.path.join(HERE, f"glasser_groups_{name}.png")
    if max(map(len, labels.values())) > 30:  # too long for two columns below: one column on the right
        save_with_legend_right(img, handles, fn)
        return
    ncol = 2

    def legend(fig, _):
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=FONTSIZE, frameon=False)
    save_with_footer(img, 0.25 * int(np.ceil(len(groups) / ncol)) + 0.15, legend, fn)


def save_with_legend_right(img, handles, fn):
    """Brain image FIG_WIDTH wide, one-column legend to its right (the saved figure grows to fit it)."""
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * img.height / img.width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FONTSIZE, frameon=False)
    # 180 dpi: the wider canvas at 200 dpi is over the 500 KB commit hook
    fig.savefig(fn, dpi=180, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"saved {fn}")


def plot_glasser_colors():
    img = brain_image(np.arange(N_GLASSER360), ListedColormap(load_glasser360_colors()), -0.5, N_GLASSER360 - 0.5)
    save_with_footer(img, 0.05, None, os.path.join(HERE, "glasser360_colors.png"))


def plot_sa_axis():
    norm = Normalize(0, 180)  # symmetric about the Rmd's midpoint 90 for ranks 1-180
    img = brain_image(load_sa_axis_ranks(), SA_CMAP, norm.vmin, norm.vmax)

    def colorbar(fig, footer_frac):
        cax = fig.add_axes([0.2, footer_frac * 0.55, 0.6, footer_frac * 0.2])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SA_CMAP), cax=cax, orientation="horizontal",
                          ticks=[1, 90, 180])
        cb.ax.set_xticklabels(["1\nsensorimotor", "90", "180\nassociation"], fontsize=FONTSIZE)
        cb.ax.xaxis.set_label_position("top")
        cb.set_label("S-A axis rank (Sydnor et al. 2021)", fontsize=FONTSIZE)
    save_with_footer(img, 0.95, colorbar, os.path.join(HERE, "sa_axis_rank.png"))


if __name__ == "__main__":
    for name in GLASSER_GROUPS:
        plot_groups(name)
    plot_glasser_colors()
    plot_sa_axis()
