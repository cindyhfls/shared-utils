'''
functions specific for HBN data
'''
import neuroboros as nb
import os

HERE = os.path.dirname(os.path.abspath(__file__))

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
        with open(os.path.join(HERE,"support_files",f"study-HBN_desc-participants-qcpass-withDM_N843.txt")) as f:
                self.subjects = f.read().splitlines()
    def rename_func(self, sid, task, run, suffix=".npy"):
        if task in ["movieDM", "movieTP"]:
            assert run==1, "movie tasks only have one run"
            basename=f"sub-{sid}*_task-{task}*{suffix}"
        else:
            basename=f"sub-{sid}*_task-{task}*run-{run:d}{suffix}"
        return basename
