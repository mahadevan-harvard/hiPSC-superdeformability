import numpy as np, h5py, pathlib

def npz_to_h5(npz_path, h5_path=None, *, compression="gzip", level=4, attrs=None):
	npz_path = pathlib.Path(npz_path)
	h5_path = pathlib.Path(h5_path) if h5_path else npz_path.with_suffix(".h5")
	with np.load(npz_path, allow_pickle=False) as data, h5py.File(h5_path, "w") as h5:
		# optional file-level metadata
		if attrs:
			for k, v in attrs.items():
				h5.attrs[k] = v
		# write each npz array as a dataset
		for key in data.files:
			arr = np.asarray(data[key])  # ensure ndarray
			h5.create_dataset(
				name=key,
				data=arr,
				compression=compression,
				compression_opts=level,
				shuffle=True,  # improves gzip compression
				chunks=True    # let h5py pick chunking
			)
	return str(h5_path)

from glob import glob
for p in glob("./*.npz"):
	npz_to_h5(p, attrs={"source": "original .npz", "creator": "Justin Tauber"})