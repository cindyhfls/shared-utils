'''
Defining datasets similar to neuroboros.datasets
Renamed from "HBN_utils".
'''
import neuroboros as nb
import os

class HBN(nb.datasets.Dataset):
    def __init__(
        self,
        space=["onavg-ico32", "mni-2mm"],
        resample=["1step_pial_overlap", "1step_fmriprep_overlap"],
        prep="default",
        fp_version="sub*",
        name="HBN",
        root_dir= "/dartfs/rc/lab/D/DBIC/DBIC/archive/HBN/nipoppy_HBN/derivatives/nb_prep/0.1.0/output", #"/dartfs/rc/lab/D/DBIC/DBIC/archive/HBN/nb-data/HBN",
        dl_source=None,
    ):
        super().__init__(
                name,
                dl_source=dl_source,
                root_dir=root_dir,
                space=space,
                resample=resample,
                prep=prep,
                fp_version=fp_version,
            )
        self.tasks = ["rest", "movieDM", "movieTP","peer"]
    def rename_func(self, sid, task, run, suffix=".npy"):
        if task in ["movieDM", "movieTP"]:
            assert run==1, "movie tasks only have one run"
            basename=f"sub-{sid}*_task-{task}*{suffix}"
        else:
            basename=f"sub-{sid}*_task-{task}*run-{run:d}{suffix}"
        return basename

class HCD(nb.datasets.Dataset):
    def __init__(
        self,
        space=["onavg-ico32"],
        resample=["msmsulc_uncleaned"],
        prep="default",
        fp_version="hcp-minimal-preprocessing",
        name="HCD",
        root_dir= "/dartfs/rc/lab/H/HaxbyLab/datasets/HCPD/nb-data/", #"/dartfs/rc/lab/D/DBIC/DBIC/archive/HBN/nb-data/HBN",
        dl_source=None,
    ):
        super().__init__(
                name,
                dl_source=dl_source,
                root_dir=root_dir,
                space=space,
                resample=resample,
                prep=prep,
                fp_version=fp_version,
            )
        self.tasks = ["rest1", "rest2", "guessing","carit", "emotion"]

    def load_confounds(self, sid, task, run, fp_version=None):
        if fp_version is None:
            fp_version = self.fp_version
        suffix_li = [
            "desc-confounds_timeseries.npy",
            "desc-confounds_timeseries.tsv",
        ]
        output = []
        for suffix in suffix_li:
            if self.rename_func is not None:
                fn = [
                    fp_version,
                    "confounds",
                    self.rename_func(sid, task, run, "_" + suffix),
                ]
            elif self.renaming is None:
                fn = [
                    fp_version,
                    "confounds",
                    f"sub-{sid}_task-{task}_run-{run:02d}_{suffix}",
                ]
            else:
                fn = [
                    fp_version,
                    "renamed_confounds",
                    f"sub-{sid}_task-{task}_run-{run:02d}_{suffix}",
                ]
                fn = self.renaming["/".join(fn)].split("/")
            o = self.dl_dset.get(fn, on_missing="raise")
            output.append(o)
        output.append(None)  # no mask-timeseries file for HCD
        return output

    def rename_func(self, sid, task, run, suffix=".npy"):
        if isinstance(run,str):
            basename = f"sub-{sid}_ses-V1_task-{task}_run-{run}{suffix}"
        else:
            basename = f"sub-{sid}_ses-V1_task-{task}_run-{run:02d}{suffix}"
        return basename
