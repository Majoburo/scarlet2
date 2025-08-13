import jax
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as dist
import equinox as eqx

from scarlet2 import Observation, ArrayPSF, Frame, Scene, Source, Parameter, Box, relative_step, GaussianPSF, init
from scarlet2.morphology import GaussianMorphology
from scarlet2.io import save_session_h5

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

    true_center = jnp.array([25.3, 24.7])             # subpixel (y, x)
    true_spectrum = jnp.array([887.7139, 7800.6488, 4630.8484, 200.988321])

    model_psf = GaussianPSF(0.3)
    obs_sigmas = [(i + 1) for i in range(C)]  # 1, 2, 3, 4 px
    kernels = []
    for s in obs_sigmas:
        k = center_pad_to_shape(GaussianPSF(s)(), (41, 41))
        k = k / jnp.sum(k)
        kernels.append(k)
    obs_psf = ArrayPSF(jnp.stack(kernels, 0))

    model_frame = Frame(Box((C, H, W)), channels=channels, psf=model_psf)

    true_bkg_center = jnp.array([20, 20])
    true_bkg_size   = 8.0
    true_bkg_ell    = jnp.array([0.1,0.1])
    true_bkg_spec   = jnp.array([120.0, 900.0, 5000.0, 20005.0])

    with Scene(model_frame) as sim_scene:
        point_morph = model_frame.psf.morphology
        Source(true_center, true_spectrum, point_morph)
        bkg_morph_true = GaussianMorphology(size=true_bkg_size,
                                            #ellipticity=true_bkg_ell,
                                            shape=(H, W))
        Source(true_bkg_center, true_bkg_spec, bkg_morph_true)

    obs = Observation(
        data=jnp.zeros((C, H, W), dtype=jnp.float32),
        weights=jnp.ones((C, H, W), dtype=jnp.float32),
        psf=obs_psf,
        channels=channels
    )

    obs = obs.match(sim_scene.frame)        
    model_obs = obs.render(sim_scene())
    read_noise_e = 2
    var = model_obs + (float(read_noise_e) ** 2)
    key = jax.random.PRNGKey(seed)
    data = (model_obs + jax.random.normal(key, shape=model_obs.shape) * jnp.sqrt(var)).astype(jnp.float32)
    weights = 1.0 / jnp.clip(var, 1e-6)


    obs = eqx.tree_at(lambda o: (o.data,o.weights), obs, (data , weights))
    print("Initial mean χ²/pixel (truth scene):", float(obs.goodness_of_fit(sim_scene())))

    # ----- FIT SCENE (initialize from helpers; will sample morphology of background) -----
    with Scene(model_frame) as scene:
        spec0_init = init.pixel_spectrum(obs, true_center, correct_psf=True)
        morph0_init = init.compact_morphology()
        Source(true_center, spec0_init, morph0_init)

        spec1_init, morph1_init = init.from_gaussian_moments(
            obs,
            true_bkg_center,
            box_sizes=[21, 31, 41, 51, 61],
            min_snr=10,
            min_corr=0.95,
        )
        morph1_init = GaussianMorphology.from_image(morph1_init)
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

    # Morphology priors for background: "neural" if available, else moments-based fallback
    init_size = getattr(scene.sources[1].morphology, "size", jnp.array(8.0))
    init_ell  = getattr(scene.sources[1].morphology, "ellipticity", jnp.array([0.0, 0.0]))

    prior_size = dist.LogNormal(loc=jnp.log(jnp.asarray(init_size)), scale=0.3)
    prior_ell  = dist.MultivariateNormal(loc=jnp.asarray(init_ell), covariance_matrix=jnp.diag(jnp.array([0.2, 0.2])**2))

    parameters += Parameter(
        scene.sources[1].morphology.size,
        name="size:1",
        prior=prior_size,
        stepsize=partial(relative_step, factor=0.05),
    )
    parameters += Parameter(
        scene.sources[1].morphology.ellipticity,
        name="ellipticity:1",
        prior=prior_ell,
        stepsize=0.02,
    )

    # ----- run sampler -----
    mcmc = scene.sample(
        obs,
        parameters,
        num_warmup=2000,
        num_samples=10000,
        progress_bar=True,
    )
    save_session_h5("obj_blend.h5", scene, obs, mcmc, id=0, path="runs", 
                    spectra=jnp.stack((true_spectrum,true_bkg_spec)),
                    centers=jnp.stack((true_center,true_bkg_center)),
                    overwrite=True)


if __name__ == "__main__":
    main()