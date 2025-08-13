import jax
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as dist
import equinox as eqx

from scarlet2 import Observation, ArrayPSF, Frame, Scene, Source, Parameter, Box, relative_step, GaussianPSF, init
from scarlet2.morphology import GaussianMorphology
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


def center_pad_to_shape(arr, target_shape):
    """Center-pad a 2D array `arr` to `target_shape` = (ty, tx) with zeros."""
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
    return jnp.pad(arr, [(pad_top, pad_bottom), (pad_left, pad_right)], mode='constant', constant_values=0)


def main(seed=1701):
    # ----- configuration -----
    C, H, W = 4, 50, 50
    channels = [f"ch{i+1}" for i in range(C)]

    # True source (model frame)
    true_center = jnp.array([25.3, 24.7])             # subpixel (y, x)
    true_spectrum = jnp.array([887.7139, 7800.6488, 4630.8484, 200.988321])

    # Model-frame PSF: small so PSF transfer is handled by rendering
    model_psf = GaussianPSF(0.30)
    obs_sigmas = [(i + 1) for i in range(C)]  # 1, 2, 3, 4 px
    kernels = []
    for s in obs_sigmas:
        k = center_pad_to_shape(GaussianPSF(s)(), (41, 41))
        k = k / jnp.sum(k)
        kernels.append(k)
    obs_psf = ArrayPSF(jnp.stack(kernels, 0))

    # ----- frames -----
    model_frame = Frame(Box((C, H, W)), channels=channels, psf=model_psf)

    # ----- SIMULATION SCENE (truth): point source + big Gaussian background -----
    true_bkg_center = jnp.array([20, 20])                    # blended with the point source
    true_bkg_size   = 8.0                            # pixels (broad background blob)
    true_bkg_ell    = jnp.array([0.05, -0.02])      # mild ellipticity (e1,e2)
    true_bkg_spec   = jnp.array([120.0, 900.0, 5000.0, 20005.0])  # per-channel flux

    with Scene(model_frame) as sim_scene:
        point_morph = model_frame.psf.morphology     # supports subpixel shift via delta_center
        Source(true_center, true_spectrum, point_morph)
        # Big Gaussian background morphology (unit area not enforced)
        bkg_morph_true = GaussianMorphology(size=true_bkg_size,
                                            ellipticity=true_bkg_ell,
                                            shape=(H, W))
        Source(true_bkg_center, true_bkg_spec, bkg_morph_true)

    # ----- make an observation object -----

    obs = Observation(
        data=jnp.zeros((C, H, W), dtype=jnp.float32),
        weights=jnp.ones((C, H, W), dtype=jnp.float32),
        psf=obs_psf,
        channels=channels
    )

    # ----- render noiseless model into observation frame -----
    obs = obs.match(sim_scene.frame)        # build renderer chain (PSF transfer, etc.)
    model_obs = obs.render(sim_scene())     # (C, H, W)

    # ----- add Gaussian noise from weights -----
    read_noise_e = 2
    var = model_obs + (float(read_noise_e) ** 2)
    key = jax.random.PRNGKey(seed)
    data = (model_obs + jax.random.normal(key, shape=model_obs.shape) * jnp.sqrt(var)).astype(jnp.float32)
    weights = 1.0 / jnp.clip(var, 1e-6)


    obs = eqx.tree_at(lambda o: (o.data,o.weights), obs, (data , weights))
    print("Initial mean χ²/pixel (truth scene):", float(obs.goodness_of_fit(sim_scene())))

    # ----- FIT SCENE (initialize from helpers; will sample morphology of background) -----
    with Scene(model_frame) as scene:
        # Source 0 (point): pixel-spectrum + compact morphology
        spec0_init = init.pixel_spectrum(obs, true_center, correct_psf=True)
        morph0_init = init.compact_morphology()
        Source(true_center, spec0_init, morph0_init)

        # Source 1 (background): initialize spectrum + morphology from Gaussian moments
        # Center is fixed (we will not sample it)
        spec1_init, morph1_init = init.from_gaussian_moments(
            obs,
            true_bkg_center,
            box_sizes=[41, 51, 61],
            min_snr=10,
            min_corr=0.95,
        )
        Source(true_bkg_center, spec1_init, morph1_init)

    # ----- set up sampling over parameters -----
    parameters = scene.make_parameters()

    # Source 0 (point): center + spectrum
    parameters += Parameter(
        scene.sources[0].center,
        name="center:0",
        prior=dist.MultivariateNormal(true_center, covariance_matrix=jnp.diag(jnp.array([2.0, 2.0])**2)),
        stepsize=0.10,
    )
    hi0 = 10.0 * float(jnp.max(true_spectrum))
    parameters += Parameter(
        scene.sources[0].spectrum,
        name="spectrum:0",
        prior=dist.Uniform(low=jnp.zeros(C), high=hi0 * jnp.ones(C)),
        stepsize=partial(relative_step, factor=0.05),
    )

    # Source 1 (background): spectrum + morphology (size, ellipticity); center is fixed
    # Spectrum prior: wide and positive
    hi1 = 10.0 * float(jnp.max(true_bkg_spec))
    parameters += Parameter(
        scene.sources[1].spectrum,
        name="spectrum:1",
        prior=dist.Uniform(low=jnp.zeros(C), high=hi1 * jnp.ones(C)),
        stepsize=partial(relative_step, factor=0.05),
    )

    morph_step = partial(relative_step, factor=1e-3)
    parameters += Parameter(
        scene.sources[1].morphology,
        name=f"morph:1",
        prior=prior64,
        stepsize = morph_step
    )

    # ----- run sampler -----
    mcmc = scene.sample(
        obs,
        parameters,
        num_warmup=100,
        num_samples=1000,
        progress_bar=True,
    )
    save_session_h5("obj_neural.h5", scene, obs, mcmc, id=0, path="runs", overwrite=True)

if __name__ == "__main__":
    main()