base_path = "/Volumes/T7 4/datasets/meural/LIBERO/libero/datasets"

import os
import h5py
for dir_name in os.listdir(base_path):
    # check if directory is libero
    if dir_name.startswith("libero_"):
        # open the hdf5 file. We're going to take a subslice of the demonstrations. Only take first 5 demonstrations.
        hdf5_path = os.path.join(base_path, dir_name, "demo.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            print(f.keys())