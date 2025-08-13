import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import numpyro.distributions as dist
import equinox as eqx
from scarlet2.io import save_session_h5
from scarlet2 import Observation, ArrayPSF, Frame, Scene, Source, Parameter, Box, relative_step, GaussianPSF

# ----------------------------
# helpers
# ----------------------------
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

def summarize(samples, key):
    q16, q50, q84 = np.percentile(np.asarray(samples[key]), [16, 50, 84], axis=0)
    return q50, (q84 - q16) / 2, q16, q84

def main(seed=1701):
    # ----- configuration -----
    C, H, W = 4, 50, 50
    channels = [f"ch{i+1}" for i in range(C)]

    # True source (model frame)
    true_center = jnp.array([25.3, 24.7])             # subpixel (y, x)
    true_spectrum = jnp.array([887.7139, 7800.6488, 4630.8484, 200.988321])

    # Model-frame PSF: small so PSF transfer is handled by rendering
    model_psf = GaussianPSF(0.30)
    obs_sigmas = [(i + 1) for i in range(C)]  # 0.1, 0.2, 0.3, 0.4 px
    kernels = []
    for s in obs_sigmas:
        k = center_pad_to_shape(GaussianPSF(s)(), (41, 41))
        k = k / jnp.sum(k)
        kernels.append(k)
    obs_psf = ArrayPSF(jnp.stack(kernels, 0))

    # ----- frames -----
    model_frame = Frame(Box((C, H, W)), channels=channels, psf=model_psf)

    # ----- build scene with a point-like morphology -----
    with Scene(model_frame) as scene:
        point_morph = model_frame.psf.morphology   # supports subpixel shift via delta_center
        Source(true_center, true_spectrum, point_morph)

    # ----- make an observation object -----
    obs = Observation(
        data=jnp.zeros((C, H, W), dtype=jnp.float32),
        weights=jnp.ones((C, H, W), dtype=jnp.float32),
        psf=obs_psf,
        channels=channels
    )

    # ----- render noiseless model into observation frame -----
    obs = obs.match(scene.frame)        # build renderer chain (PSF transfer, etc.)
    model_obs = obs.render(scene())     # (C, H, W)

    # ----- add Gaussian noise from weights -----
    read_noise_e = 2
    var = model_obs + (float(read_noise_e) ** 2)
    key = jax.random.PRNGKey(seed)
    data = (model_obs + jax.random.normal(key, shape=model_obs.shape) * jnp.sqrt(var)).astype(jnp.float32)
    weights = 1.0 / jnp.clip(var, 1e-6)


    obs = eqx.tree_at(lambda o: (o.data,o.weights), obs, (data , weights))
    print("Initial mean χ²/pixel:", float(obs.goodness_of_fit(scene())))

    # ----- set up sampling over center and spectrum -----
    parameters = scene.make_parameters()
    # Center prior: fairly loose around truth (pixels)
    parameters += Parameter(
        scene.sources[0].center,
        name="center:0",
        prior=dist.MultivariateNormal(true_center, covariance_matrix=jnp.diag(jnp.array([2.0, 2.0])**2)),
        stepsize=0.10,
    )
    # Spectrum prior: positive & wide
    hi = 10.0 * float(jnp.max(true_spectrum))
    parameters += Parameter(
        scene.sources[0].spectrum,
        name="spectrum:0",
        prior=dist.Uniform(low=jnp.zeros(C), high=hi * jnp.ones(C)),
        stepsize=partial(relative_step, factor=0.05),
    )

    # ----- run sampler -----
    mcmc = scene.sample(
        obs,
        parameters,
        num_warmup=2000,
        num_samples=10000,
        progress_bar=True,
    )
    save_session_h5("obj.h5", scene, obs, mcmc, id=0, path="runs", overwrite=True)

    samples = mcmc.get_samples()

    c50, cerr, c16, c84 = summarize(samples, "center:0")
    s50, serr, s16, s84 = summarize(samples, "spectrum:0")

    print("\n=== Recovery ===")
    print(f"True center       : {np.array(true_center)}")
    print(f"Posterior center  : {c50}  (± {cerr})  68%: [{c16}, {c84}]")
    print(f"\nTrue spectrum     : {np.array(true_spectrum)}")
    print(f"Posterior spectrum: {s50}\n  68%: [{s16}, {s84}]")
    plt.plot(true_spectrum)
    plt.plot(s50)
    plt.show()

    # ----- quick diagnostic figure -----
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    axes[0].imshow(jnp.mean(obs.data, axis=0), origin="lower");     axes[0].set_title("Data (mean over C)")
    axes[1].imshow(jnp.mean(model_obs, axis=0), origin="lower");    axes[1].set_title("Model (mean over C)")
    resid = (obs.data - model_obs)
    w = obs.weights
    z = jnp.where(w.sum(axis=0) > 0, (w * resid).sum(axis=0) / jnp.sqrt(w.sum(axis=0)), 0.0)
    axes[2].imshow(z, origin="lower", cmap="RdBu_r");              axes[2].set_title("Standardized residual z")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.show()

if __name__ == "__main__":
    main()