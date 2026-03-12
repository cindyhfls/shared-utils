import nilearn as nli
from nilearn import image, plotting
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

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
