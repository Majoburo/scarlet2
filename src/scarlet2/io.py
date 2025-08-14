"""Methods to save and load scenes"""

import os
import pickle
import json
import h5py
import numpy as np


def model_to_h5(model, filename, id=0, path=".", overwrite=False):
    """Save the scene model to a HDF5 file

    Parameters
    ----------
    filename : str
        Name of the HDF5 file to create
    model : :py:class:`~scarlet2.Module`
        Scene to be stored
    id : int
        HDF5 group to store this `model` under
    path: str, optional
        Explicit path for `filename`. If not set, uses local directory
    overwrite : bool, optional
        Whether to overwrite an existing file with the same path and filename

    Returns
    -------
    None

    Notes
    -----
    This is not a pure function hence cannot be utilized within a JAX JIT compilation.
    """
    # create directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # first serialize the model into a pytree
    model_group = str(id)
    save_h5_path = os.path.join(path, filename)

    f = h5py.File(save_h5_path, "a")
    # create a group for the scene
    if model_group in f:
        if overwrite:
            del f[model_group]
        else:
            raise ValueError("ID already exists. Set overwrite=True to overwrite the ID.")

    # save the binary to HDF5
    group = f.create_group(model_group)
    model = pickle.dumps(model)
    group.attrs["model"] = np.void(model)
    f.close()


def model_from_h5(filename, id=0, path="."):
    """
    Load scene model from a HDF5 file

    Parameters
    ----------
    filename : str
        Name of the HDF5 file to load from
    id : int
        HDF5 group to identify the scene by
    path: str, optional
        Explicit path for `filename`. If not set, uses local directory

    Returns
    -------
    :py:class:`~scarlet2.Scene`
    """

    filename = os.path.join(path, filename)
    f = h5py.File(filename, "r")
    model_group = str(id)
    if model_group not in f:
        raise ValueError(f"ID {id} not found in the file.")

    group = f.get(model_group)
    out = group.attrs["model"]
    binary_blob = out.tobytes()
    scene = pickle.loads(binary_blob)
    f.close()

    return scene


def save_session_h5(filename, scene, obs, mcmc, id=0, path=".", spectra = None, centers= None, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path)
    save_h5_path = os.path.join(path, filename)
    group_name = str(id)

    samples = mcmc.get_samples(group_by_chain=False)

    with h5py.File(save_h5_path, "a") as f:
        if group_name in f:
            if overwrite:
                del f[group_name]
            else:
                raise ValueError(f"ID {id} already exists. Set overwrite=True to replace it.")
            
        g = f.create_group(group_name)

        scene_blob = pickle.dumps(scene, protocol=pickle.HIGHEST_PROTOCOL)
        obs_blob   = pickle.dumps(obs,   protocol=pickle.HIGHEST_PROTOCOL)
        g.create_dataset(
            "scene_pickle",
            data=np.frombuffer(scene_blob, dtype="uint8"),
            compression="gzip", compression_opts=4, shuffle=True, fletcher32=True
        )
        g.create_dataset(
            "obs_pickle",
            data=np.frombuffer(obs_blob, dtype="uint8"),
            compression="gzip", compression_opts=4, shuffle=True, fletcher32=True
        )

        gs = g.create_group("samples")
        for k, v in samples.items():
            dsname = str(k).replace("/", "_")
            gs.create_dataset(
                dsname, data=np.asarray(v),
                compression="gzip", compression_opts=4, shuffle=True, fletcher32=True
            )
            
        g.attrs["meta"] = json.dumps({
            "centers": centers,
            "spectra": spectra,
            "channels": getattr(scene.frame, "channels", None),
        })
        
def load_session_h5(filename, id=0, path="."):
    load_h5_path = os.path.join(path, filename)
    group_name = str(id)
    with h5py.File(load_h5_path, "r") as f:
        if group_name not in f:
            raise ValueError(f"ID {id} not found in file.")
        g = f[group_name]

        # --- READ FROM DATASETS ---
        scene = pickle.loads(bytes(g["scene_pickle"][...]))
        obs   = pickle.loads(bytes(g["obs_pickle"][...]))

        samples = {}
        if "samples" in g:
            for k in g["samples"].keys():
                samples[k] = np.array(g["samples"][k])
        meta = {}
        if "meta" in g.attrs:
            meta = json.loads(g.attrs["meta"])
        
        return scene, obs, samples, meta
