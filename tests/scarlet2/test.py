from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astropy.io import fits
from collections import defaultdict
import re
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d
from functools import partial
import numpyro.distributions as dist
from numpyro.distributions import (
    TransformedDistribution,
    MultivariateNormal,
    constraints,
    transforms
)
from scarlet2.morphology import GaussianMorphology
from scarlet2 import Observation, ArrayPSF, Frame, Scene, Source, init, Parameter, relative_step, GaussianPSF
from scarlet2.io import save_session_h5

# load in the model you wish to use
from galaxygrad import get_prior
from scarlet2.nn import ScorePrior

# instantiate the prior class
temp = 2e-2  # values in the range of [1e-3, 1e-1] produce good results
prior32 = get_prior("hsc32")
prior64 = get_prior("hsc64")
prior32 = ScorePrior(prior32, prior32.shape(), t=temp)
prior64 = ScorePrior(prior64, prior64.shape(), t=temp)



def create_radial_taper(shape, sigma=50.0):
    """Create a radial Gaussian taper."""
    if isinstance(shape, int):
        shape = (shape, shape)
    y, x = jnp.indices(shape)
    center_y, center_x = [(s - 1) / 2 for s in shape]
    r = (x - center_x)**2 + (y - center_y)**2
    morph = jnp.exp(-0.5 * r/sigma**2)
    morph /= jnp.sum(morph)
    return morph

def extract_filter(filename):
    # Match both NIR and VIS formats
    match = re.search(r"MOSAIC-(NIR|VIS)(?:-([YJH]))?", filename)
    if match:
        instr, filt = match.groups()
        return filt if filt else "VIS"  # VIS doesn't have sub-bands
    return None

def center_pad_to_shape(arr, target_shape):
    
    """
    Center-pad a 2D array `arr` to `target_shape` = (ty, tx).
    Pads equally (or as close as possible) on all sides with zeros.
    """
    ny, nx = arr.shape
    ty, tx = target_shape
    if ty < ny or tx < nx:
        raise ValueError("target_shape must be >= arr.shape")
    pad_y = ty - ny
    pad_x = tx - nx

    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left

    return jnp.pad(arr,
                  pad_width=[(pad_top, pad_bottom), (pad_left, pad_right)],
                  mode='constant', constant_values=0)

def main():
    files = list(Path(".").glob('TILE*/*.fits.gz'))
    channels = ['H', 'J', 'Y', 'VIS']

    # Map from OBJECTID to dict of {channel: filename}
    objid_to_channel_files = defaultdict(dict)

    for file in files:
        with fits.open(file) as hdul:
            objid = hdul[1].header['OBJECTID']
            channel = extract_filter(file.name)
            if channel in channels:
                objid_to_channel_files[objid][channel] = file

    # Sort OBJECTIDs
    sorted_objids = sorted(objid_to_channel_files.keys())[1]
    print(sorted_objids)

    # For each OBJECTID, stack the 2D arrays in channel order for data, weights, and psfs
    objid_to_data = {}
    objid_to_weights = {}
    objid_to_psfs = {}

    for objid in [sorted_objids]:
        data_arrays = []
        weights_arrays = []
        psf_arrays = []
        for i,channel in enumerate(channels):
            file = objid_to_channel_files[objid].get(channel)
            with fits.open(file) as hdul:
                data = jnp.asarray(hdul[1].data, dtype=np.float32)
                weights = 1 / jnp.asarray(hdul[2].data, dtype=np.float32) ** 2
                psf = GaussianPSF(i+1)()
                data_arrays.append(data)
                weights_arrays.append(weights)
                psf_arrays.append(center_pad_to_shape(psf, (41,41)))
        objid_to_data[objid] = jnp.stack(data_arrays, axis=0)
        objid_to_weights[objid] = jnp.stack(weights_arrays, axis=0)
        objid_to_psfs[objid] = jnp.stack(psf_arrays,axis=0)

        obs = Observation(
            objid_to_data[objid],
            objid_to_weights[objid],
            psf=ArrayPSF(objid_to_psfs[objid]),
            channels=channels,
        )
        #obs.data =      gal = GaussianMorphology(size=float(30), ellipticity=[0.1,0.2],shape=hdul[1].data.shape)
        #centers = jnp.array([(97.5,97.1),(98.5,99.5),(96.5,95.5),(98.2,90.1),(112.4,71.1)]) # (y,x)
        #spectra = [[3000,2000,1000,100],[2500,1000,500,100],[1000,1500,600,50],[600,500,400,10],[2000,1300,600,20]]


        model_frame = Frame.from_observations(obs)#, model_psf=ArrayPSF(objid_to_psfs[objid][0]))
        centers = jnp.array([(97.5,97.1),(98.5,99.5),(96.5,95.5),(98.2,90.1),(112.4,71.1)]) # (y,x)
        centers = jnp.array([(97.0,97.0),(98.0,99.0),(96.0,95.0),(98.0,90.0),(112.0,71.0)]) # (y,x)

        centers = jnp.array([(98.0,99.0)])#,(98.0,90.0)])
        box_sizes = [[61],[23],[23],[23],[23]]
        #box_sizes = [[23],[23],[23],[23]]

        with Scene(model_frame) as scene:
            for i, center in enumerate(centers):
                if False:
                    spectrum, morph = init.from_gaussian_moments(obs, center, box_sizes=box_sizes[i])#min_snr=10, min_corr=0.99)
                    plt.plot(spectrum)#morph = GaussianMorphology.from_image(morph)
                else:
                    spectrum = init.pixel_spectrum(obs, center)
                    plt.plot(spectrum)
                    morph = init.compact_morphology()
                Source(center, spectrum, morph)
        plt.show()
        #scene_array = scene()  # evaluate the model
        
        

        obs = Observation(
            obs.render(scene()),
            objid_to_weights[objid],
            psf=ArrayPSF(objid_to_psfs[objid]),
            channels=channels,
        )
        model_frame = Frame.from_observations(obs)#, model_psf=ArrayPSF(objid_to_psfs[objid][-1]))

        with Scene(model_frame) as scene:
            for i, center in enumerate(centers):
                if False:
                    spectrum, morph = init.from_gaussian_moments(obs, center, box_sizes=box_sizes[i])#min_snr=10, min_corr=0.99)
                    spectrum = init.pixel_spectrum(obs, center)
                    plt.plot(spectrum)#morph = GaussianMorphology.from_image(morph)
                else:
                    spectrum = init.pixel_spectrum(obs, center)
                    plt.plot(spectrum)
                    morph = init.compact_morphology()
                Source(center, spectrum, morph)
        plt.show()
        scene_array = scene()  # evaluate the model
        print("Initial likelihood:", obs.log_likelihood(scene_array))

        spec_step = partial(relative_step, factor=0.05)
        morph_step = partial(relative_step, factor=1e-3)
        parameters = scene.make_parameters()
        C = len(channels)
        for i in range(len(scene.sources)):
            parameters += Parameter(
                scene.sources[i].spectrum,
                name=f"spectrum:{i}",
                prior=dist.Uniform(low=jnp.zeros(C), high=1000*np.max(objid_to_data[objid]) * jnp.ones(C)),
                stepsize=spec_step,
                )

            if False:   
                parameters += Parameter(
                        scene.sources[i].morphology,
                        name=f"morph:{i}",
                        #prior= dist.HalfNormal(create_radial_taper(scene.sources[i].morphology.shape,box_sizes[i][0]/3)),
                        #prior=dist.Uniform(low=0, high=1).expand(scene.sources[i].morphology.shape),
                        prior=prior64,
                        stepsize = morph_step
                    )
            else:
                parameters += Parameter(scene.sources[i].center,
                        name=f"center:{i}",
                        stepsize=0.1,
                        prior=dist.MultivariateNormal(scene.sources[i].center, covariance_matrix=jnp.diag(jnp.array([10,10]))),
                        )
            if False:
                if i!=0:
                    parameters += Parameter(
                            scene.sources[i].morphology.ellipticity,
                            name=f"ellipticity:{i}",
                            prior = ReparameterizableEllipticityPrior(sigma=0.1),
                            stepsize = morph_step,
                        )
                    parameters += Parameter(
                        scene.sources[i].morphology.size,
                        name=f"size:{i}",
                        prior = dist.Uniform(low=0.1, high=box_sizes[i][0]),
                        stepsize = morph_step
                    )

        maxiter = 600
        scene.set_spectra_to_match(obs, parameters)
        mcmc = scene.sample(
            obs,
            parameters,
            num_warmup=400,
            num_samples=maxiter,
            progress_bar=True,
        )

        save_session_h5("obj.h5", scene, obs, mcmc, id=0, path="runs", overwrite=True)

main()