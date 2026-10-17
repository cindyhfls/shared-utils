'''
Helper function to call nilearn to plot on subcortical structures
'''
import nilearn as nli
from nilearn import image, plotting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACES_DIR = os.path.join(HERE, "surfaces")

T1_MNINLin2009cAsym_fn = os.path.join(HERE,"support_files","tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz")
T1_MNI152NLin6Asym_fn = os.path.join(HERE,"support_files","tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz")

def map_subcortical_roi_to_img(data, subcortical_atlas, rois, bg_img = 'T1_MNI2009cAsym',interactive=False, **kwargs):
   """
   Take the ROI data and map it back to a NIFTI image for visualization.
   """
   lut = pd.read_csv(os.path.join(HERE,"support_files",f"atlas-{subcortical_atlas}_space-MNI152NLin2009cAsym_res-02.lut.tsv"),sep="\t")
   img = image.load_img(T1_MNINLin2009cAsym_fn)
   if bg_img == 'T1_MNI152NLin6Asym':
      raise ValueError('The lookup table is for MNINLin2009cAsym so MNI152NLin6Asym is not supported')
   elif bg_img is not None:
      bg_img = img

   templatedata = img.get_fdata()
   stat_map = np.zeros_like(templatedata, dtype=np.float64)

   if isinstance(data,dict):
      l = []
      for roi in rois:
         l.append(data[roi])
      data = l
   elif not isinstance(rois,list):
      rois = [rois]
      data = [data]

   if len(data)==1 and len(rois)!=1:
      assert len(data[0])==len(lut),"if single data vector provided, it must match the full table length, i.e. all voxels in this specified atlas"
      stat_map = data
   else:
      assert len(data)==len(rois),"please provide data and rois in lists of the same size"
      for roi,d in zip(rois,data):
         stat_map[lut[lut.label==roi].i,lut[lut.label==roi].j,lut[lut.label==roi].k] = d

   stat_map_img = image.new_img_like(img, stat_map)
   if interactive:
      view = plotting.view_img(stat_map_img,bg_img=bg_img,**kwargs)
   else:
      view = plotting.plot_stat_map(stat_map_img,bg_img=bg_img,**kwargs)
   return view


def plot_surface_map(
    values,
    space="onavg-ico32",
    mask=None,
    surf_type="inflated",
    cmap="RdBu_r",
    vmax=None,
    vmin=None,
    colorbar=True,
    view_mode=None,
    title=None,
    fn=None,
    size=None,
    zoom=1.3,
    use_sulci=True,
    gyri_color=(0.8, 0.8, 0.8, 1.0),
    sulci_color=(0.6, 0.6, 0.6, 1.0),
    **kwargs,
):
    """Plot a cortical surface map using brainspace 3D rendering.

    Accepts data in any neuroboros space (e.g. onavg-ico8, onavg-ico32),
    masked or unmasked. Data is automatically upsampled to onavg-ico32 for
    rendering via nearest-neighbour surface mapping, so ico8 searchlight
    results look clean rather than blocky.

    Parameters
    ----------
    values : array-like
        1D array, masked or full length for the given space.
    space : str
        Neuroboros space the data is in (default "onavg-ico32").
    mask : array-like, optional
        Boolean cortex mask for ``space``. Defaults to the neuroboros mask.
    surf_type : str
        Surface type; falls back through midthickness/white/pial.
    cmap : str
        Matplotlib colormap name (default "RdBu_r").
    vmax : float, optional
        Color scale max. Defaults to 97.5th percentile of |values|.
    vmin : float, optional
        Color scale min. Defaults to -vmax.
    colorbar : bool
    view_mode : str or list of str
        One or more of "lateral", "medial", "dorsal", "ventral",
        "anterior", "posterior". Each entry becomes a row in the output.
        Default is ["lateral", "dorsal", "medial"].
    title : str, optional
        Not yet rendered; reserved for future use.
    fn : str or Path, optional
        If given, save a screenshot to this path (offscreen render).
        If None, return an embeddable notebook image.
    size : tuple of int, optional
        Window size (width, height). Defaults to (1200, 300 * n_views).
    zoom : float
        Camera zoom factor (default 1.55).
    use_sulci : bool
        If True (default), shade unmasked vertices with sulci/gyri grays based
        on curvature sign (negative curvature = gyri, positive = sulci),
        matching neuroboros.plot behaviour.
    gyri_color : tuple of float
        RGBA color for gyral (convex) vertices (default light gray).
    sulci_color : tuple of float
        RGBA color for sulcal (concave) vertices (default dark gray).
    **kwargs
        Extra keyword arguments forwarded to brainspace.plotting.plot_surf
        (e.g. background, scale, transparent_bg, suppress_warnings).

    Returns
    -------
    IPython.display.Image or None
        Notebook-embeddable image when fn is None; None when saving to file.
    """
    import matplotlib as mpl
    import neuroboros as nb
    from neuroboros.spaces import get_mapping, get_mask
    from brainspace.plotting import plot_surf
    from brainspace.mesh.mesh_creation import build_polydata
    from brainspace.mesh.mesh_io import read_surface

    _SURF_SPACE = "onavg-ico128"
    _SURF_NV    = 163842  # 128**2 * 10 + 2
    # _SURF_SPACE = "onavg-ico32"
    # _SURF_NV    = 10242  # 32**2 * 10 + 2

    # --- masks ---
    if mask is None:
        mask_lh = get_mask('l', space)
        mask_rh = get_mask('r', space)
    else:
        mask = np.asarray(mask, dtype=bool)
        n_lh_full = get_mask('l', space).size
        mask_lh = mask[:n_lh_full]
        mask_rh = mask[n_lh_full:]

    n_lh_full   = mask_lh.size
    n_lh_masked = int(mask_lh.sum())
    n_rh_masked = int(mask_rh.sum())

    # --- split data into lh/rh masked arrays ---
    values = np.asarray(values, dtype=np.float32).ravel()
    n = len(values)
    if n == n_lh_masked + n_rh_masked:
        v_lh = values[:n_lh_masked]
        v_rh = values[n_lh_masked:]
    elif n == n_lh_full + mask_rh.size:
        v_lh = values[:n_lh_full][mask_lh]
        v_rh = values[n_lh_full:][mask_rh]
    else:
        raise ValueError(
            f"values length {n} does not match masked "
            f"({n_lh_masked + n_rh_masked}) or full "
            f"({n_lh_full + mask_rh.size}) vertices for space={space!r}."
        )

    # --- vmax / vmin ---
    finite = np.concatenate([v_lh, v_rh])
    finite = finite[np.isfinite(finite)]
    if vmax is None:
        vmax = float(np.percentile(np.abs(finite), 97.5))
    if vmin is None:
        vmin = -vmax

    # --- upsample to ico128 for rendering (matches nb.plot: unmask first, then map) ---
    full_lh = np.full(mask_lh.size, np.nan, dtype=np.float32)
    full_rh = np.full(mask_rh.size, np.nan, dtype=np.float32)
    full_lh[mask_lh] = v_lh
    full_rh[mask_rh] = v_rh

    if space == _SURF_SPACE:
        data_lh = full_lh
        data_rh = full_rh
    else:
        M_lh = get_mapping('l', space, _SURF_SPACE, nn=True)
        M_rh = get_mapping('r', space, _SURF_SPACE, nn=True)
        data_lh = (M_lh.T @ full_lh).astype(np.float32)
        data_rh = (M_rh.T @ full_rh).astype(np.float32)

    # --- sulci / gyri background coloring (matches neuroboros.plot behaviour) ---
    # Strategy: replace NaN vertices with sentinel scalars just below vmin, then
    # prepend two extra colors (sulci, gyri) to the colormap so those sentinels
    # render with the correct gray shades.  Brainspace always uses 256 LUT
    # entries for float arrays, so we build a 256-entry ListedColormap:
    #   entry 0  → sulci_color  (sentinel = vmin - 2*eps)
    #   entry 1  → gyri_color   (sentinel = vmin - eps)
    #   entries 2-255 → 254-step version of user cmap over [vmin, vmax]
    # With eps = (vmax-vmin)/254 the LUT step equals eps, so each sentinel
    # lands in exactly its own bin.
    render_cmap = cmap
    render_color_range = (vmin, vmax)
    if use_sulci:
        from neuroboros.spaces import get_morphometry
        curv_lh = get_morphometry("curv", "l", space=_SURF_SPACE).astype(np.float32)
        curv_rh = get_morphometry("curv", "r", space=_SURF_SPACE).astype(np.float32)

        data_range = vmax - vmin
        if data_range == 0:
            data_range = 1.0
        N_DATA = 254
        eps = data_range / N_DATA
        SULCI_SENT = float(vmin - 2 * eps)
        GYRI_SENT  = float(vmin - eps)

        nan_lh = np.isnan(data_lh)
        nan_rh = np.isnan(data_rh)
        data_lh = data_lh.copy()
        data_rh = data_rh.copy()
        data_lh[nan_lh] = np.where(curv_lh[nan_lh] < 0, GYRI_SENT, SULCI_SENT)
        data_rh[nan_rh] = np.where(curv_rh[nan_rh] < 0, GYRI_SENT, SULCI_SENT)

        gyri_rgba  = np.clip(list(gyri_color)  + ([1.0] if len(gyri_color)  == 3 else []), 0, 1)
        sulci_rgba = np.clip(list(sulci_color) + ([1.0] if len(sulci_color) == 3 else []), 0, 1)
        user_cmap_fn = mpl.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        data_colors = user_cmap_fn(np.linspace(0, 1, N_DATA))
        all_colors  = np.vstack([sulci_rgba, gyri_rgba, data_colors])
        render_cmap = mpl.colors.ListedColormap(all_colors)
        render_color_range = (SULCI_SENT, vmax)

    # --- surfaces ---
    def _get_surf(lr):
        hk = "lh" if lr == "l" else "rh"

        # surf_type="neuroboros_<type>" → load directly from neuroboros (no gii needed)
        if surf_type.startswith("neuroboros_"):
            nb_st = surf_type[len("neuroboros_"):]
            coords, faces = nb.geometry(nb_st, lr, space=_SURF_SPACE)
            return build_polydata(coords, faces)

        for st in [surf_type, "midthickness", "white", "pial"]:
            gii_path = os.path.join(SURFACES_DIR, f"{_SURF_SPACE}_{hk}_{st}.surf.gii")
            if os.path.exists(gii_path):
                return read_surface(gii_path)
        raise RuntimeError(
            f"No .surf.gii found for hemi={lr!r} in {SURFACES_DIR}. "
            f"Expected e.g. {_SURF_SPACE}_{hk}_{surf_type}.surf.gii. "
            f"Pass surf_type='neuroboros_inflated' to load from neuroboros instead."
        )

    surf_lh = _get_surf('l')
    surf_rh = _get_surf('r')

    # _CORE_DIR = os.path.join(HERE, "support_files", "core")
    # _ico = _SURF_SPACE.split("-ico")[1]
    # _space_dir = f"onavg-ico{_ico}"
    # faces_lh = np.load(os.path.join(_CORE_DIR, _space_dir, "geometry", "faces", "lh.npy"))
    # faces_rh = np.load(os.path.join(_CORE_DIR, _space_dir, "geometry", "faces", "rh.npy"))
    # coords_lh = np.load(os.path.join(_CORE_DIR, _space_dir, "geometry", surf_type, "lh", "on1031_trimmed.npy"))
    # coords_rh = np.load(os.path.join(_CORE_DIR, _space_dir, "geometry", surf_type, "rh", "on1031_trimmed.npy"))
    # surf_lh = build_polydata(coords_lh, faces_lh)
    # surf_rh = build_polydata(coords_rh, faces_rh)

    # def _get_surf(lr):
    #     hk = "lh" if lr == "l" else "rh"
    #     for st in [surf_type, "midthickness", "white", "pial"]:
    #         gii_path = os.path.join(SURFACES_DIR, f"{_SURF_SPACE}_{hk}_{st}.surf.gii")
    #         if os.path.exists(gii_path):
    #             return read_surface(gii_path)
    #         try:
    #             coords, faces = nb.geometry(st, lr, space=_SURF_SPACE)
    #             return build_polydata(coords, faces)
    #         except Exception:
    #             continue
    #     raise RuntimeError(f"No surface available for hemi={lr!r}, space={_SURF_SPACE!r}")
    # surf_lh = _get_surf('l')
    # surf_rh = _get_surf('r')

    import io
    import tempfile
    from PIL import Image as PILImage

    # --- layout ---
    if view_mode is None:
        view_mode = ["lateral", "ventral", "medial"]
    elif isinstance(view_mode, str):
        view_mode = [view_mode]

    n_views    = len(view_mode)
    panel_size = (size[1] // n_views) if size is not None else 300  # square panels

    # n_views rows × 2 cols, lh left rh right (neuroboros order).
    # RH sits at +x in bilateral MNI space so lateral↔medial orientations are
    # mirrored relative to LH — swap them so each hemisphere shows correctly.
    _rh_view = {'lateral': 'medial', 'medial': 'lateral'}
    layout   = [['lh', 'rh'] for _ in range(n_views)]
    view_arr = [[v, _rh_view.get(v, v)] for v in view_mode]

    # --- render to temp file, post-process dorsal rotation with PIL ---
    arr_name = '__plot_data__'
    surf_lh.append_array(data_lh, name=arr_name, at='p')
    surf_rh.append_array(data_rh, name=arr_name, at='p')

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    try:
        plot_surf(
            {'lh': surf_lh, 'rh': surf_rh},
            layout,
            array_name=arr_name,
            view=view_arr,
            cmap=render_cmap,
            color_range=render_color_range,
            nan_color=(0.8, 0.8, 0.8, 1),
            zoom=zoom,
            size=(2 * panel_size, n_views * panel_size),
            share='both',
            color_bar=None,
            screenshot=True,
            filename=tmp,
            transparent_bg=True,
            suppress_warnings=True,
            **kwargs,
        )
        full_img = PILImage.open(tmp).copy()
    finally:
        os.unlink(tmp)
        surf_lh.remove_array(name=arr_name, at='p')
        surf_rh.remove_array(name=arr_name, at='p')

    # rebuild image row-by-row so dorsal rows can be cropped to content height
    actual_pw = full_img.size[0] // 2
    actual_ph = full_img.size[1] // n_views
    rows = []
    for row_i, v in enumerate(view_mode):
        y0, y1 = row_i * actual_ph, (row_i + 1) * actual_ph
        if v in ('dorsal', 'ventral'):
            panels = []
            angles = [-90, 90] if v == 'dorsal' else [90, -90]
            for col_i, angle in enumerate(angles):
                x0, x1 = col_i * actual_pw, (col_i + 1) * actual_pw
                panel = full_img.crop((x0, y0, x1, y1)).rotate(angle, expand=True)
                bbox = panel.getbbox()
                if bbox:
                    panel = panel.crop((0, bbox[1], panel.size[0], bbox[3]))
                panels.append(panel)
            row_h = max(p.size[1] for p in panels)
            row_img = PILImage.new('RGBA', (actual_pw * 2, row_h))
            for col_i, panel in enumerate(panels):
                row_img.paste(panel, (col_i * actual_pw, (row_h - panel.size[1]) // 2))
        else:
            row_img = full_img.crop((0, y0, actual_pw * 2, y1))
        rows.append(row_img)

    total_h = sum(r.size[1] for r in rows)
    stitched = PILImage.new('RGBA', (actual_pw * 2, total_h))
    y = 0
    for row_img in rows:
        stitched.paste(row_img, (0, y))
        y += row_img.size[1]
    img_w, img_h = stitched.size

    # colorbar strip on the right
    if colorbar:
        dpi = 150
        cbar_h_in = img_h / dpi
        fig, cax = plt.subplots(figsize=(0.8, cbar_h_in), dpi=dpi)
        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        )
        sm.set_array([])
        fig.colorbar(sm, cax=cax)
        fig.subplots_adjust(left=0.5, right=0.7, top=0.95, bottom=0.05)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, transparent=True, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        cbar_img = PILImage.open(buf).copy()
        cw, ch = cbar_img.size
        cbar_img = cbar_img.resize(
            (int(cw * img_h / ch), img_h), PILImage.LANCZOS
        )
        combined = PILImage.new('RGBA', (stitched.size[0] + cbar_img.size[0], img_h))
        combined.paste(stitched, (0, 0))
        combined.paste(cbar_img, (stitched.size[0], 0))
    else:
        combined = stitched

    if title:
        combined = _add_title(combined, title)

    if fn is not None:
        combined.save(str(fn))
        return None

    from IPython.display import Image as IPythonImage
    buf = io.BytesIO()
    combined.save(buf, format='png')
    return IPythonImage(buf.getvalue())


def _add_title(img, title, font_size=20):
    from PIL import ImageDraw, ImageFont, font_manager as fm
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(fm.findfont(fm.FontProperties()), font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 4), title, fill=(0, 0, 0, 255), font=font)
    return img


def plot_box_with_strip(df_curr, curr_ax, palette_curr, x="ISC", y="ROI",
                        cond="condition", orient="h"):
    """Box plot with the individual observations overlaid as a strip.

    Lifted from 30_save_ISC_plots.py (which keeps its own copy because that module runs
    plotting at import time and so cannot be imported) and from the notebook cells that
    were cut into 45_plot_ridgeCV_alpha_distribution.py.
    """
    import seaborn as sns
    sns.boxplot(
        data=df_curr, x=x, y=y, hue=cond, palette=palette_curr, orient=orient,
        fliersize=0,  # outlier markers; the strip already shows every point
        ax=curr_ax,
    )
    sns.stripplot(
        data=df_curr, x=x, y=y, hue=cond, dodge=True, palette='dark:.3',
        size=1.5, alpha=0.8, orient=orient, ax=curr_ax, legend=False,
    )
