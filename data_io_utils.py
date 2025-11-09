import numpy as np
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TianLUT = pd.read_csv(os.path.join(HERE, "support_files", "atlas-TianS1S2S3S4_combined_keylabel.tsv"), sep="\t")

def parse_range(s):
    out = []
    for token in s.split():
        if "-" in token:
            a, b = map(int, token.split("-"))
            out.extend(range(a, b+1))
        else:
            out.append(int(token))
    return out

def return_Tian_labels(atlas_name):
    labels = np.unique(TianLUT[f'{atlas_name}_label'])
    return list(labels)

def return_aseg_labels(atlas_name='aseg'):
    labels = ['l-accumbens', 'l-amygdala', 'l-caudate','l-hippocampus','l-pallidum','l-putamen','l-thalamus','l-ventral-diencephalon',
                    'r-accumbens', 'r-amygdala', 'r-caudate','r-hippocampus','r-pallidum','r-putamen','r-thalamus','r-ventral-diencephalon']
    if atlas_name.endswith('subj'):
        labels = [f'{label}_subj' for label in labels]
    return labels

def convert_dataarray_to_dictionary_byroi(data, atlas_name):
    '''
    data: Input is nTimepoints x nVoxels
    TianLUT: table specifying the keys for each ROI

    Returns:
    d: the dictionary containing data for each ROI
    labels: the labels based on the atlas given use this to check it matches with the ROIS you have

    '''
    d = {}
    labels = return_Tian_labels(atlas_name)
    for label in labels:
        idx = TianLUT[f'{atlas_name}_label'] == label
        d[label] = data[:,idx]
    return d,labels

def get_rois_and_space(atlas,surface_space,volume_space):
    if atlas.startswith('onavg'): # e.g. onavg-ico32
        seedrois = ['l','r']
        space = surface_space
    elif atlas.startswith('Tian'):
        seedrois = return_Tian_labels(atlas)
        space = volume_space
    elif atlas.startswith('aseg'):
        seedrois = return_aseg_labels(atlas)
        space = volume_space
    return seedrois,space

def get_Tian_sls(atlas_name):
    labels = return_Tian_labels(atlas_name)
    sls = []
    for label in labels:
        idx = TianLUT[f'{atlas_name}_label'] == label
        sls.append(idx)
    return sls

def get_aseg_sls():
    '''
    Make a list of indices for the ROI identity of each voxel in the atlas.
    Make sure your subcortical_rois is loaded with the same function return_{atlas}_labels otherwise it might be wrong!
    '''
    subcortical_atlas = 'aseg'
    subcortical_rois = return_aseg_labels()
    subcortical_voxelcount = pd.read_csv(os.path.join(HERE, "support_files", f'atlas-{subcortical_atlas}_VoxelCount.txt'),sep='\t')

    sls = []
    offset = 0
    for roi in subcortical_rois:
        nvox = np.array(subcortical_voxelcount.VoxelCount[subcortical_voxelcount.Label==roi])[0]
        sls.append(np.arange(offset,offset+nvox))
        offset += nvox
    print('Please check this is the order of your ROIs \n')
    print(subcortical_rois)
    return sls

def unwrap_dm_dict(dm_dict):
    '''Reshape timeseries data from dictionary to array in shape nSubject x nTime x nV'''
    dm = []
    for sid in dm_dict.keys():
        allrois = dm_dict[sid].keys()
        dm.append(np.concatenate([dm_dict[sid][roi] for roi in allrois],axis=1))
    dm = np.stack(dm, axis=0)
    return dm

def get_dm(dset, sids, rois, task, run, space, atlas="aseg",saved_beta_path="",array_output = False):
    '''
    Returns a nested dictionary by subject and roi if array_output = False, else a ndarray of shape (nSubject, nTime, nVoxels/nVertices)
    '''
    if isinstance(rois,str):
        rois = [rois]
    if isinstance(sids,str):
        sids = [sids]
    dm = {}
    for sid in sids:
        ts = {}
        if atlas.startswith('Tian'):
            if saved_beta_path:
                curr_out = dset.get_data(sid, task, run, 'Tian_Subcortex', space=space,prep='saved_beta',prep_kwargs={'saved_beta_path':saved_beta_path,'load_beta':True})
            else:
                curr_out = dset.get_data(sid, task, run, 'Tian_Subcortex', space=space,prep='default')
            data = curr_out[0]if isinstance(curr_out, tuple) else curr_out
            if rois[0]=='Tian_Subcortex':
                ts = data
            elif rois[0].endswith('lh') or rois[0].endswith('rh'): # dirty check for Tian, we could prob improve that
                ts,labels = convert_dataarray_to_dictionary_byroi(data, atlas) # e.g. 'TianS1'
                assert(rois==labels)
        else:
            for roi in rois:
                if saved_beta_path:
                    curr_out = dset.get_data(sid, task, run, roi, space=space,prep='saved_beta',prep_kwargs={'saved_beta_path':saved_beta_path,'load_beta':True})
                else:
                    curr_out = dset.get_data(sid, task, run, roi, space=space,prep='default')
                ts[roi] = curr_out[0]if isinstance(curr_out, tuple) else curr_out
        dm[sid] = ts
    if array_output:
        dm = unwrap_dm_dict(dm)
        return np.squeeze(dm)
    else:
        return dm
