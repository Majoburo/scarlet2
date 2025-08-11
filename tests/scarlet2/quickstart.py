from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astropy.io import fits
from collections import defaultdict
import re
import numpy as np
import corner
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

from scarlet2 import Observation, ArrayPSF, Frame, Scene, Source, init, Parameter, relative_step
from scarlet2.io import save_session_h5

def corner_centers_spectra(mcmc, scene, sources=None, channels="all", thin=1, max_samples=5000):
    """
    Corner plot of centers and spectra for selected sources.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        The MCMC object returned by scene.sample(...).
    scene : scarlet2.Scene
        Scene (for channel names and source count).
    sources : list[int] or None
        Which sources to include. None = all.
    channels : "all" | list[int]
        Which spectral channels to include. "all" uses all.
    thin : int
        Take every `thin`-th sample.
    max_samples : int
        Randomly subsample to at most this many rows (after thinning).
    """
    samples = mcmc.get_samples(group_by_chain=False)
    nsrc = len(scene.sources)
    if sources is None:
        sources = list(range(nsrc))
    if channels == "all":
        C = scene.frame.C
        chan_idx = list(range(C))
    else:
        chan_idx = list(channels)

    cols = []
    labels = []

    # Build columns
    for i in sources:
        # centers
        key_c = f"center:{i}"
        if key_c in samples:
            Ci = np.asarray(samples[key_c])[::thin]
            cols.append(Ci[:, 0]); labels.append(f"S{i}: center_dy")
            cols.append(Ci[:, 1]); labels.append(f"S{i}: center_dx")
        # spectra
        key_s = f"spectrum:{i}"
        if key_s in samples:
            Si = np.asarray(samples[key_s])[::thin]
            # choose channels
            for k in chan_idx:
                if k < Si.shape[1]:
                    lab_ch = scene.frame.channels[k] if scene.frame.channels else k
                    cols.append(Si[:, k]); labels.append(f"S{i}: spec[{lab_ch}]")

    if not cols:
        raise ValueError("No center: or spectrum: samples found for requested sources.")

    # Align lengths + stack
    # (some arrays may differ in length if keys missing for some sources)
    min_len = min(len(c) for c in cols)
    cols = [c[:min_len] for c in cols]
    X = np.vstack(cols).T  # (Nsamples, D)

    # Subsample if huge
    if len(X) > max_samples:
        idx = np.random.choice(len(X), size=max_samples, replace=False)
        X = X[idx]

    fig = corner.corner(
        X,
        labels=labels,
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
        smooth=0.9,
        bins=30,
        label_kwargs={"fontsize": 9},
        title_kwargs={"fontsize": 9},
    )
    plt.show()
    return fig
    
class UnitDiskTransform(transforms.Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    event_dim = 1

    def __call__(self, x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scale = jnp.tanh(norm) / (norm + 1e-7)
        return scale * x

    def _inverse(self, y):
        norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
        scale = jnp.arctanh(norm) / (norm + 1e-7)
        return scale * y

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r = jnp.linalg.norm(x, axis=-1)        
        r_safe = jnp.where(r==0, 1e-6, r)
        s = jnp.tanh(r_safe)/r_safe                   
        ds = (r_safe*(1 - jnp.tanh(r_safe)**2) - jnp.tanh(r_safe)) / (r_safe**2)
        logdet = jnp.log(s) + jnp.log(s + r_safe * ds) 
        return logdet

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

def ReparameterizableEllipticityPrior(sigma=0.3):
    base = MultivariateNormal(loc=jnp.zeros(2), covariance_matrix=jnp.eye(2) * sigma**2)
    return TransformedDistribution(base, UnitDiskTransform())

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

def extend_psf_blended(
    psf,
    new_shape,
    center=None,
    n_bins=100,
    transition_frac=0.7,
    transition_width=10,
    decay_scale=None,
):
    """
    Extend a PSF to a larger shape smoothly, enforcing monotonic radial decrease.

    Parameters:
        psf : 2D numpy.ndarray
            Original PSF image array.
        new_shape : tuple of int
            Target shape (ny, nx) for the extended PSF array.
        center : tuple (float, float), optional
            Center (y, x) of the PSF. Defaults to PSF center.
        n_bins : int, optional
            Number of bins for radial profile calculation.
        transition_frac : float, optional
            Fraction of original radius where blending starts.
        transition_width : float, optional
            Width (pixels) over which blending transitions from core to tail.
        decay_scale : float, optional
            If provided, PSF tail exponentially decays beyond original radius with given scale.

    Returns:
        2D numpy.ndarray
            Smoothly extended PSF array of shape `new_shape`.
    """
    ny, nx = psf.shape
    Ny, Nx = new_shape

    # Determine original PSF center
    if center is None:
        y0, x0 = (ny - 1) / 2., (nx - 1) / 2.
    else:
        y0, x0 = center

    # Radial coordinates of original PSF
    y, x = np.indices(psf.shape)
    r = np.hypot(x - x0, y - y0).ravel()
    psf_flat = psf.ravel()

    # Compute radial profile via binning
    r_max = r.max()
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    radial_means = np.array([
        psf_flat[(r >= bin_edges[i]) & (r < bin_edges[i+1])].mean() if np.any((r >= bin_edges[i]) & (r < bin_edges[i+1])) else 0
        for i in range(n_bins)
    ])

    # Ensure monotonic decrease using isotonic regression
    iso_reg = IsotonicRegression(increasing=False)
    mono_profile = iso_reg.fit_transform(bin_centers, radial_means)

    # Smooth interpolation of radial profile
    interp_profile = interp1d(
        bin_centers, mono_profile,
        kind='cubic', fill_value='extrapolate', assume_sorted=True
    )

    # New grid and radial distances
    y_new, x_new = (Ny - 1) / 2., (Nx - 1) / 2.
    Y, X = np.indices((Ny, Nx))
    r_new = np.hypot(X - x_new, Y - y_new)

    # Compute smooth tail with optional exponential decay
    tail = interp_profile(np.minimum(r_new, r_max))
    if decay_scale:
        tail *= np.exp(-(r_new - r_max).clip(min=0) / decay_scale)

    # Embed original PSF centrally
    extended_psf = np.zeros((Ny, Nx), dtype=float)
    y_start, x_start = int(round(y_new - y0)), int(round(x_new - x0))
    extended_psf[y_start:y_start+ny, x_start:x_start+nx] = psf

    # Smooth blending weight function (sigmoid)
    transition_radius = transition_frac * r_max
    weight = 1 / (1 + np.exp((r_new - transition_radius) / (transition_width / 5)))

    # Combine original core and tail smoothly
    combined = weight * extended_psf + (1 - weight) * tail

    # Remove possible negative artifacts and ensure positivity
    combined = np.clip(combined, a_min=0, a_max=None)

    # Normalize flux to match original PSF flux
    combined *= psf.sum() / combined.sum()

    return combined

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

# initialise the spectrum
import operator
from functools import reduce
from scarlet2 import Scenery
def _sort_spectra(spectra, channels):
    try:
        frame = Scenery.scene.frame
    except AttributeError:
        print("Multi-observation initialization can only be created within the context of a Scene")
        print("Use 'with Scene(frame) as scene: ...")
        raise

    spectrum = []
    for channel in frame.channels:
        try:
            idx = channels.index(channel)
            spectrum.append(spectra[idx])
        except ValueError:
            msg = f"Channel '{channel}' not found in observations. Setting amplitude to 0."
            print(msg)
            spectrum.append(0)
    spectrum = jnp.array(spectrum)
    return spectrum
def subpixel_spectrum(obs, pos, correct_psf=False):
    """Get the spectrum at a given position in the observation(s).

    Yields the spectrum of a single-subpixel source with flux 1 in every channel,
    concatenated for all observations.

    Parameters
    ----------
    obs: `:py:class:`~scarlet2.Observation` or list
        Observation(s) to extract pixel SED from
    pos: tuple
        Position in the observation. Needs to be in sky coordinates if multiple
        observations have different locations or pixel scales.
    correct_psf: bool, optional
        Whether PSF shape variations in the observations should be corrected.
        If `True`, this method homogenizes the PSFs of the observations, which
        yields the correct spectrum for a flux=1 point source.

    Returns
    -------
    array or list
        If `obs` is a list, the method returns the associate list of spectra.
    """

    # for multiple observations, get spectrum from each observation and then
    # combine channels in order of model frame
    if isinstance(obs, (list, tuple)):
        # flat lists of spectra and channels in order of observations
        spectra = jnp.concatenate([subpixel_spectrum(obs_, pos, correct_psf=correct_psf) for obs_ in obs])
        channels = reduce(operator.add, [obs_.frame.channels for obs_ in obs])
        spectrum = _sort_spectra(spectra, channels)

        return spectrum

    assert isinstance(obs, Observation)

    pixel = obs.frame.get_pixel(pos).astype(int)

    if not obs.frame.bbox.spatial.contains(pixel):
        raise ValueError(f"Pixel coordinate expected, got {pixel}")

    spectrum = obs.data[:, pixel[0], pixel[1]].copy()

    if correct_psf and obs.frame.psf is not None:
        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Adaptive morphology can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise
        if frame.psf is None:
            raise AttributeError("Adaptive morphology can only be create with a PSF in the model frame")

        # correct spectrum for PSF-induced change in peak pixel intensity
        psf_model = obs.frame.psf()
        psf_peak = psf_model.max(axis=(-2, -1))

        psf0_model = frame.psf()
        psf0_peak = psf0_model.max(axis=(-2, -1))

        spectrum /= psf_peak / psf0_peak

    if jnp.any(spectrum <= 0):
        # If the flux in all channels is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = f"Zero or negative spectrum {spectrum} at {pos}"
        if jnp.all(spectrum <= 0):
            print("Zero or negative spectrum in all channels: Setting spectrum to 1")
            spectrum = jnp.ones_like(spectrum)
        print(msg)

    return spectrum

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
                data = jnp.asarray(hdul[1].data, dtype=jnp.float32)
                weights = 1 / np.asarray(hdul[2].data, dtype=np.float32) ** 2
                psf = jnp.asarray(hdul[4].data, dtype=jnp.float32)
                data_arrays.append(data)
                weights_arrays.append(weights)
                if i == 1:
                    weights[102:113, 31:45] = np.exp(-73.4) # apparently this is '0' in the existing weights
                    weights[104:114, 107:112] = np.exp(-73.4)
                weights = jnp.asarray(weights, dtype=jnp.float32)

                psf_arrays.append(center_pad_to_shape(psf, (data.shape[0]//5,data.shape[1]//5)))


        objid_to_data[objid] = jnp.stack(data_arrays, axis=0)
        objid_to_weights[objid] = jnp.stack(weights_arrays, axis=0)
        objid_to_psfs[objid] = jnp.stack(psf_arrays, axis=0)
        obs = Observation(
            objid_to_data[objid],
            objid_to_weights[objid],
            psf=ArrayPSF(objid_to_psfs[objid]),
            channels=channels,
        )

        model_frame = Frame.from_observations(obs)#, model_psf=ArrayPSF(objid_to_psfs[objid][-]))
        centers = jnp.array([(97.5,97.1),(98.5,99.5),(96.5,95.5),(98.2,90.1),(112.4,71.1)]) # (y,x)

        box_sizes = [[65],[23],[23],[23],[23]]

        with Scene(model_frame) as scene:
            for i, center in enumerate(centers):
                if i==0:
                    spectrum, morph = init.from_gaussian_moments(obs, center, box_sizes=box_sizes[i])#min_snr=10, min_corr=0.99)
                    #morph = GaussianMorphology.from_image(morph)
                else:
                    spectrum = init.pixel_spectrum(obs, center)
                    morph = init.compact_morphology()
                Source(center, spectrum, morph)
        
        scene_array = scene()  # evaluate the model

        print("Initial likelihood:", obs.log_likelihood(scene_array))

        spec_step = partial(relative_step, factor=0.05)
        morph_step = partial(relative_step, factor=1e-2)
        parameters = scene.make_parameters()
        C = len(channels)
        for i in range(len(scene.sources)):
            parameters += Parameter(
                scene.sources[i].spectrum,
                name=f"spectrum:{i}",
                prior=dist.Uniform(low=jnp.zeros(C), high=1000*np.max(objid_to_data[objid]) * jnp.ones(C)),
                stepsize=spec_step,
                )

            if i == 0:   
                parameters += Parameter(
                        scene.sources[i].morphology,
                        name=f"morph:{i}",
                        prior= dist.HalfNormal(create_radial_taper(scene.sources[i].morphology.shape,box_sizes[i][0]/3)),
                        stepsize = morph_step
                    )
            else:
                parameters += Parameter(scene.sources[i].center,
                        name=f"center:{i}",
                        stepsize=0.1,
                        prior=dist.MultivariateNormal(scene.sources[i].center, covariance_matrix=jnp.diag(jnp.array([1,1]))),
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

        maxiter = 10000
        #scene.set_spectra_to_match(obs, parameters)
        mcmc = scene.sample(
            obs,
            parameters,
            num_warmup=10,
            num_samples=maxiter,
            progress_bar=True,
        )

        save_session_h5("obj.h5", scene, obs, mcmc, id=0, path="runs", overwrite=True)

main()