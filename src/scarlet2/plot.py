# To be removed as part of issue #169
# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103

"""Plotting functions"""

from abc import ABC, abstractmethod
import copy

import jax
import jax.numpy as jnp
import jax.random as random
from jax import grad, jit, jvp
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle
import corner

from . import measure
from .bbox import Box
from .renderer import ChannelRenderer
from .morphology import GaussianMorphology

def channels_to_rgb(channels):
    """Get the linear mapping of multiple channels to RGB channels
    The mapping created here assumes the the channels are ordered in wavelength
    direction, starting with the shortest wavelength. The mapping seeks to produce
    a relatively even weights for across all channels. It does not consider e.g.
    signal-to-noise variations across channels or human perception.

    Parameters
    ----------
    channels: int in range(0,7)
        Number of channels

    Returns
    -------
    array
     (3, channels) to map onto RGB
    """
    assert channels in range(0, 8), f"No mapping has been implemented for more than {channels} channels"

    channel_map = np.zeros((3, channels))
    if channels == 1:
        channel_map[0, 0] = channel_map[1, 0] = channel_map[2, 0] = 1
    elif channels == 2:
        channel_map[0, 1] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[1, 0] = 0.333
        channel_map[2, 0] = 0.667
        channel_map /= 0.667
    elif channels == 3:
        channel_map[0, 2] = 1
        channel_map[1, 1] = 1
        channel_map[2, 0] = 1
    elif channels == 4:
        channel_map[0, 3] = 1
        channel_map[0, 2] = 0.333
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.667
        channel_map[2, 1] = 0.333
        channel_map[2, 0] = 1
        channel_map /= 1.333
    elif channels == 5:
        channel_map[0, 4] = 1
        channel_map[0, 3] = 0.667
        channel_map[1, 3] = 0.333
        channel_map[1, 2] = 1
        channel_map[1, 1] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 1.667
    elif channels == 6:
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    elif channels == 7:
        channel_map[:, 6] = 2 / 3.0
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    return channel_map


class Norm(ABC):
    """Base class to normalize the color values of RGB images"""

    def __init__(self):
        self._uint8Max = float(np.iinfo(np.uint8).max)

    def get_intensity(self, im):
        """Compute total intensity image"""
        return jnp.maximum(0, im).sum(axis=0)

    def clip(self, im, min_value, max_value):
        """Clip image between min_value and max_value"""
        return jnp.maximum(0, jnp.minimum(im - min_value, max_value - min_value))

    def convert_to_uint8(self, im):
        """Convert three-channel image to RGB image with uint8 dtype"""
        im_clipped = self.clip(im, 0, 1)
        uint_im = (im_clipped * self._uint8Max).astype("uint8")
        im_flipped = uint_im.transpose().swapaxes(0, 1)  # 3 x Ny x Nx -> Ny x Nx x 3
        return im_flipped

    def make_rgb_image(self, *im):
        """Compute RGB image from three-channel image"""
        # backwards compatible to astropy Mapping Call
        return self.convert_to_uint8(self.__call__(jnp.stack(im, axis=0)))

    @abstractmethod
    def __call__(self, im):
        """Compute normalized three-channel image"""
        pass


class LinearNorm(Norm):
    def __init__(self, minimum, maximum):
        """Linear norm, mapping the interval [`minimum`, `maximum`] to [0,1]

        Parameters
        ----------
        minimum: float
            Value that will be mapped to 0
        maximum: float
            Value that will be mapped to 1
        """
        self.min_value, self.max_value = minimum, maximum
        super().__init__()

    def __call__(self, im):
        return self.clip(im, self.min_value, self.max_value) / (self.max_value - self.min_value)


class LinearPercentileNorm(LinearNorm):
    def __init__(self, img, percentiles=(1, 99)):
        """Norm that is linear between the two elements of `percentiles` of `img`

        Parameters
        ----------
        img: array
            Image to normalize
        percentiles: array-like, optional
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated. Default is (1, 99)
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        super().__init__(minimum=vmin, maximum=vmax)


class AsinhNorm(Norm):
    """AsinhNorm class"""

    def __init__(self, min_value, max_value, beta):
        """Norm that scales as arcsinh(I / beta) between `m` and `M`

        See Lupton+(2004) https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L

        Parameters
        ----------
        min_value: float
            Minimum value to consider
        max_value: float
            Maximum value to consider
        beta: float
            Turnover point of arcsinh. Below it norm behaves linear, above
            it norm approximates ln(2*I)
        """
        self.min_value, self.max_value, self.beta = min_value, max_value, beta
        self._rgb_max = 1
        super().__init__()

    def set_rgb_max(self, img, vibrance=0.15):
        """Set maximum value of normalized image

        Parameters
        ----------
        img: array
            Three-channel image
        vibrance: float
            Allowance to exceed normalization of three-channel image.
            Makes images more vibrant but causes slight color shifts towards white in the highlights.
        """
        rgb = self.__call__(img)
        self._rgb_max = rgb[np.isfinite(rgb)].max() / (1 + vibrance)

    def __call__(self, img):
        """Compute Asinh normalized image"""
        min_value = self.min_value
        max_value = self.max_value
        intensity = self.get_intensity(img)
        with np.errstate(invalid="ignore", divide="ignore"):  # n.b. np.where can't and doesn't short-circuit
            # clip between m and M
            i_ = self.clip(intensity - min_value, 0, max_value - min_value)

            # arcsinh scaling from Lupton+(2004)
            f = np.arcsinh(i_ / self.beta)  # no need to normalize, done below
            rgb = img / (intensity / f)[None, :, :]

            # keep rgb between 0 and 1 (with an allowance of self.vibrance)
            rgb = rgb / self._rgb_max

            return rgb


class AsinhPercentileNorm(AsinhNorm):
    """AsinhPercentileNorm class"""

    def __init__(self, img, percentiles=(45, 50, 99), vibrance=0.15):
        """Norm that scales as arcsinh(I / beta) between bottom and top percentile

        Uses the middle percentile to define the turnover `beta`. The defaults
        are chosen such that the median (percentile 50) tries to catch emission
        slightly above the sky level, while the minimum is aiming for the sky
        intensity itself.

        Parameters
        ----------
        img: array_like
            Image to normalize
        percentiles: array_like
            Lower, middle, and upper percentile to consider. Pixel values below will be
            set to zero, above to one. Asinh turnover is given by middle percentile.
            Default is (45,50,99)
        vibrance: float
            Allowance to exceed normalization of three-channel image.
            Makes images more vibrant but causes slight color shifts in the highlights.
        """
        assert len(percentiles) == 3
        min_value, beta, max_value = np.percentile(img, percentiles)
        super().__init__(min_value, max_value, beta)
        super().set_rgb_max(img, vibrance=vibrance)


class AsinhAutomaticNorm(AsinhNorm):
    def __init__(
        self,
        observation,
        channel_map=None,
        minimum=0,
        upper_percentile=99.5,
        noise_level=1,
        vibrance=0.15,
    ):
        """Norm that scales as arcsinh(I / beta) with parameters chosen automatically

        The turnover `beta` is taken from the at `noise_level` * RMS, where RMS is the
        total variance of the observations. This norm should automatically create an
        image scaling that picks out low-surface brightness features and highlights.

        Parameters
        ----------
        observation: py:class:`~scarlet2.Observation`
            Observation object with weights
        channel_map: array
            Linear mapping from channels to RGB, dimensions (3, channels)
        minimum: float
            Minimum value to consider.
        upper_percentile: float
            Upper percentile: Pixel values above will be saturated.
        noise_level: float
            Factor to be multiplied to the total noise RMS to define the turnover point
        vibrance: float
            Allowance to exceed normalization of three-channel image.
            Makes images more vibrant but causes slight color shifts in the highlights.
        """
        if channel_map is None:
            channel_map = channels_to_rgb(observation.frame.C)

        im3 = img_to_3channel(observation.data, channel_map=channel_map)
        var3 = 1 / observation.weights
        var3 = np.where(np.isfinite(var3), var3, 0)
        var3 = img_to_3channel(1 / observation.weights, channel_map=channel_map)

        # total intensity and variance images
        i = self.get_intensity(im3)
        v = self.get_intensity(var3)

        # find upper clipping point
        (max_value,) = np.percentile(i.flatten(), [upper_percentile])
        min_value = minimum

        # find a good turnover point for arcsinh: ~noise level
        rms = np.median(np.sqrt(v))
        beta = rms * noise_level

        super().__init__(min_value, max_value, beta)
        super().set_rgb_max(im3, vibrance=vibrance)


def img_to_3channel(img, channel_map=None):
    """Convert multi-band image cube into 3 RGB channels

    Parameters
    ----------
    img: array
        This should be an array with dimensions (channels, height, width).
    channel_map: array
        Linear mapping from channels to RGB, dimensions (3, channels)

    Returns
    -------
    array
        Dimensions (3, height, width), type float
    """
    # expand single img into cube

    assert img.ndim in [2, 3]
    if len(img.shape) == 2:
        ny, nx = img.shape
        img_ = img.reshape(1, ny, nx)
    elif len(img.shape) == 3:
        img_ = img
    num_channels = len(img_)

    # filterWeights: channel x band
    if channel_map is None:
        channel_map = channels_to_rgb(num_channels)
    else:
        assert channel_map.shape == (3, len(img))

    # map channels onto RGB channels
    _, ny, nx = img_.shape
    rgb = jnp.dot(channel_map, img_.reshape(num_channels, -1)).reshape(3, ny, nx)

    rgb = jnp.where(np.isfinite(rgb), rgb, 0)

    return rgb


def img_to_rgb(img, channel_map=None, fill_value=0, norm=None, mask=None):
    """Convert images to normalized RGB.

    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.

    Parameters
    ----------
    img: array
        This should be an array with dimensions (channels, height, width).
    channel_map: array
        Linear mapping from channels to RGB, dimensions (3, channels)
    fill_value: float, optional
        Value to use for any masked pixels.
    norm: Norm, optional
        Norm to use for mapping in the allowed range [0..255]. If `norm=None`,
        `scarlet.display.LinearPercentileNorm` will be used.
    mask: array_like, optional
        A [0,1] binary mask to apply over the top of the image,
        where pixels with mask==1 are masked out.

    Returns
    -------
    array
        Dimensions (3, height, width), type float
    """
    im3 = img_to_3channel(img, channel_map=channel_map)
    if norm is None:
        norm = LinearPercentileNorm(im3)
    rgb = norm.make_rgb_image(*im3)
    if mask is not None:
        rgb = jnp.dstack([rgb, ~mask * 255])
    return rgb


panel_size = 4.0


def observation(
    observation,
    norm=None,
    channel_map=None,
    sky_coords=None,
    show_psf=False,
    add_labels=True,
    split_channels=False,
    fig_kwargs=None,
    title_kwargs=None,
    label_kwargs=None,
):
    """Plot observation

    Show entire content of `observation`, optionally with list of sources given
    by `sky_coords` or a PSF image.

    Parameters
    ----------
    observation: :py:class:`~scarlet2.Observation`
        The observation object to plot
    norm: Norm, optional
        Norm to scale the intensity of `observation` into RGB 0..256
    channel_map: array, optional
        Linear mapping from channels to RGB, dimensions (3, channels)
    sky_coords: list, optional
        2D coordinates (in pixel coordinates or sky coordinates).
        If in sky coordinates, the Frame of `observation` needs to have a valid WCS.
    show_psf: bool, optional
        Whether to plot a panel with the PSF model of `observation` centered in
        the middle
    add_labels: bool, optional
        Whether to plot a text label with the running number for each of the
        sources in `sky_coords`
    split_channels: bool, optional
        Whether to split the observation into separate channels
    fig_kwargs: dict, optional
        Additional arguments for `mpl.subplots`
    title_kwargs: dict, optional
        Additional arguments for `mpl.set_title`
    label_kwargs: dict, optional
        Additional arguments for `mpl.text`. Default is None and will be set to
        `{"color": "w", "ha": "center", "va": "center"}`

    Returns
    -------
    mpl.Figure
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    if title_kwargs is None:
        title_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {"color": "w", "ha": "center", "va": "center"}

    if show_psf:
        assert observation.frame.psf is not None, "show_psf requires observation.frame.psf to be set"
        psf_model = observation.frame.psf()

    rows = len(observation.frame.channels) if split_channels else 1
    panels = 1 if show_psf is False else 2
    figsize = fig_kwargs.pop("figsize", None)
    if figsize is None:
        figsize = (panel_size * panels, panel_size * rows)
    fig, ax = plt.subplots(rows, panels, figsize=figsize, squeeze=False, **fig_kwargs)
    if not hasattr(ax, "__iter__"):
        ax = (ax,)

    extent = observation.frame.bbox.get_extent()

    for row in range(rows):
        if split_channels:
            data = observation.data[row]
            mask = observation.weights[row] == 0
            name = observation.frame.channels[row]
            if show_psf:
                psf = psf_model[row]
                # make PSF as bright as the brightest pixel of the observation
                psf *= data.max() / psf.max()
        else:
            data = observation.data
            # Mask any pixels with zero weight in all channels
            mask = np.sum(observation.weights, axis=0) == 0
            name = ""
            if show_psf:
                psf = psf_model
                # make PSF as bright as the brightest pixel of the observation
                psf *= observation.data.mean(axis=0).max() / psf_model.mean(axis=0).max()

        # if there are no masked pixels, do not use a mask
        if np.all(mask == 0):
            mask = None

        panel = 0
        ax[row, panel].imshow(
            img_to_rgb(data, norm=norm, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[row, panel].set_title(f"Observation {name}", **title_kwargs)

        if add_labels and sky_coords is not None:
            for k, center in enumerate(sky_coords):
                center_ = observation.frame.get_pixel(center)
                ax[row, panel].text(*center_[::-1], k, **label_kwargs)

        if show_psf:
            panel = 1
            psf_image = np.zeros(data.shape)
            # insert into middle of "blank" observation
            full_box = Box(psf_image.shape)
            shift = tuple(psf_image.shape[d] // 2 - psf.shape[d] // 2 for d in range(full_box.D))
            model_box = Box(psf.shape) + shift
            model_box.insert_into(psf_image, psf)
            # slices = scarlet.box.overlapped_slices
            ax[row, panel].imshow(img_to_rgb(psf_image, norm=norm), origin="lower")
            ax[row, panel].set_title("PSF", **title_kwargs)

    fig.tight_layout()
    return fig


# ------------------------------------------------------ #
# include a routine to calculate the hallucination score #
#  ----------------------------------------------------- #
def cut_square_box(arr, center, size):
    """
    Cut out a square box from a 2D array based on the center and size.

    Parameters:
    arr: numpy.ndarray
        The input 2D array.
    center: tuple
        The center of the box in the format (row_center, col_center).
    size: int
        The size of the square box (side length).

    Returns:
        numpy.ndarray: The square box extracted from the input array.
    """

    # get the dimensions of the data
    obs_dim = arr.ndim

    row_center, col_center = center
    # col_center, row_center = center
    half_size = size // 2

    # Calculate the indices for slicing
    start_row = row_center - half_size
    end_row = start_row + size
    start_col = col_center - half_size
    end_col = start_col + size

    # Ensure the indices are within the array bounds
    start_row = max(0, start_row)
    start_col = max(0, start_col)
    if obs_dim == 2:
        end_row = min(arr.shape[0], end_row)
        end_col = min(arr.shape[1], end_col)
    else:
        end_row = min(arr.shape[1], end_row)
        end_col = min(arr.shape[2], end_col)

    # Cut out the square box
    if obs_dim == 2:
        square_box = arr[start_row:end_row, start_col:end_col]
    else:
        square_box = arr[:, start_row:end_row, start_col:end_col]

    # pad array up if needed (ie box outside array bounds)
    pad = False
    if obs_dim == 2:
        if square_box.shape[0] < size or square_box.shape[1] < size:
            pad_low = size - square_box.shape[0]
            pad_high = size - square_box.shape[1]
            pad = True
    else:
        if square_box.shape[1] < size or square_box.shape[2] < size:
            pad_low = size - square_box.shape[1]
            pad_high = size - square_box.shape[2]
            pad = True

    # perform the padding
    if pad:
        # If the square box is not the correct size, pad it with zeros
        if pad_low < 0:
            pad_low = 0
        if pad_high < 0:
            pad_high = 0
        if obs_dim <= 2:
            square_box = np.pad(square_box, ((pad_low, 0), (pad_high, 0)), mode="constant", constant_values=0)
        else:
            # Get the original array shape
            original_height, original_width, num_channels = square_box.shape

            # Create a new zero-padded array
            padded_rgb_array = np.zeros(
                (original_height + 2 * pad_high, original_width + 2 * pad_low, num_channels),
                dtype=square_box.dtype,
            )

            # Place the original RGB array in the center of the padded array
            padded_rgb_array[
                pad_high : pad_high + original_height, pad_low : pad_low + original_width, :
            ] = square_box

    return square_box


@jax.grad
def neural_grad(galaxy, src):
    parameters = src.get_parameters(return_info=True)
    prior = 2 * sum(
        info["prior"].log_prob(galaxy) for name, (p, info) in parameters.items() if info["prior"] is not None
    )
    return prior


def log_like(morph, spectrum, data, weights):
    model = morph[None, :, :] * spectrum[:, None, None]
    d = jnp.prod(jnp.asarray(data.shape)) - jnp.sum(weights == 0)
    log_norm = d / 2 * jnp.log(2 * jnp.pi)
    log_like = -jnp.sum(weights * (model - data) ** 2) / 2
    return log_like - log_norm


# --------------------- #
# Hessian approximation #
# --------------------- #
# https://arxiv.org/pdf/2006.00719.pdf


# for regular functions f
def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]


# for score functions
def hvp_grad(grad_f, primals, tangents):
    return jvp(grad_f, primals, tangents)[1]


# diagonals of Hessian from HVPs
def hvp_rad(hvp, shape):
    """Approximate the diagonal of the Hessian"""
    max_iters = 100  # maximum number of iterations
    h = jnp.zeros(shape, dtype=jnp.float32)
    h_ = jnp.zeros(shape, dtype=jnp.float32)
    for i in range(max_iters):
        key = random.PRNGKey(i)
        z = random.rademacher(key, shape, dtype=jnp.float32)
        h += jnp.multiply(z, hvp(z))
        if i > 0:
            norm = jnp.linalg.norm(h / (i + 1) - h_ / i, ord=2)
            if norm < 1e-6 * jnp.linalg.norm(h / (i + 1), ord=2):  # gets reasonable results with 1e-2
                break
        h_ = h
    return h / (i + 1)


# TODO: fix the jit compilation errors here
def hallucination_score(scene, obs, src_num):
    """Calculate the hallucination score of a source in `scene` based on `obs`"""
    src = scene.sources[src_num]
    center = np.array(src.morphology.bbox.center)[::-1]
    morph = src.morphology.data
    f = lambda morph: neural_grad(morph, src)
    jit_hvp_x2 = jit(lambda z: hvp_grad(f, (morph,), (z,)))
    hvp_nn = hvp_rad(jit_hvp_x2, morph.shape)
    hvp_nn = np.array(hvp_nn)

    model_scene = scene()
    morph = model_scene[
        src_num
    ]  # FIXME: this must be wrong because that is a channel image, not a source image
    spectrum = jnp.array((1,))
    data = obs.data
    weights = obs.weights

    # jit the HVP for this loss and this morph model
    f = lambda morph: log_like(morph, spectrum, data, weights)  # noqa: E371
    jit_hvp_x = jit(lambda z: hvp(f, (morph,), (z,)))
    hvp_ll = hvp_rad(jit_hvp_x, morph.shape)

    box_size = hvp_nn.shape[1]
    # Cut out the square box
    hvp_ll_cut = cut_square_box(hvp_ll, center, box_size)
    hallucination = -hvp_nn + hvp_ll_cut

    return -hallucination * src.morphology(), jnp.sum(-hallucination * src.morphology())


def confidence(scene, observation):
    """The confidence of each source in `scene` based on the hallucination score"""
    sources = scene.sources
    n_sources = len(sources)
    metrics = np.zeros(n_sources)
    for k, _ in enumerate(sources):
        _, metric = hallucination_score(scene, observation, k)
        metrics[k] = metric
    return metrics


def sources(
    scene,
    observation=None,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_spectrum=True,
    model_mask=None,
    add_markers=True,
    add_boxes=False,
    fig_kwargs=None,
    title_kwargs=None,
    marker_kwargs=None,
    box_kwargs=None,
):
    """Plot all sources in `scene`

    Creates one figure, with each source in `scene` occupying one row. Depending
    on the chosen options, multiple panels per source will be created.

    Parameters
    ----------
    scene: :py:class:`~scarlet2.Scene`
        The scene object containing the sources and their models
    observation: :py:class:`~scarlet2.Observation`, optional
        The observation to render the sources for, or to show the data of.
        Only needed when `show_observed` or `show_rendered` is True.
    norm: Norm, optional
        Norm to scale the intensity of `observation` into RGB 0..256
    channel_map: array, optional
        Linear mapping from channels to RGB, dimensions (3, channels)
    show_model: bool, optional
        Whether to show the internal model of each source
    show_observed: bool, optional
        Whether to show the observations in the same region as the source
    show_rendered: bool, optional
        Whether to show the model of each source rendered into the frame of `observation`
    show_spectrum: bool, optional
        Whether to show the spectrum of each source
    model_mask: array, optional
        A mask to apply to the model. If not given, no mask is applied
    add_markers: bool, optional
        Whether to plot a marker at the center for each source
        Requires the source to have a `center` attribute.
    add_boxes: bool, optional
        Whether to plot the bounding box of each source
    fig_kwargs: dict, optional
        Additional arguments for `mpl.subplots`
    title_kwargs: dict, optional
        Additional arguments for `mpl.set_title`
    marker_kwargs: dict, optional
        Additional arguments for `mpl.plot` of the source centers. Defaults to
        {"color": "w", "marker": "x", "mew": 1, "ms": 10}
    box_kwargs: dict, optional
        Additional arguments for `mpl.Polygon`.
        Defaults to {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    Returns
    -------
    mpl.Figure
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    if title_kwargs is None:
        title_kwargs = {}
    if marker_kwargs is None:
        marker_kwargs = {"color": "w", "marker": "x", "mew": 1, "ms": 10}
    if box_kwargs is None:
        box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    sources = scene.sources
    n_sources = len(sources)
    panels = sum((show_model, show_observed, show_rendered, show_spectrum))

    figsize = fig_kwargs.pop("figsize", None)
    if figsize is None:
        figsize = (panel_size * panels, panel_size * n_sources)

    fig, ax = plt.subplots(n_sources, panels, figsize=figsize, squeeze=False, **fig_kwargs)

    for k, src in enumerate(sources):
        center = np.array(src.center)[::-1]
        start, stop = src.bbox.start[-2:][::-1], src.bbox.stop[-2:][::-1]
        box_coords = (start, (start[0], stop[1]), stop, (stop[0], start[1]))

        # model in its bbox
        panel = 0
        model = src()
        if show_model:
            # Show the unrendered model in it's bbox
            extent = src.bbox.get_extent()
            ax[k][panel].imshow(
                img_to_rgb(model, norm=norm, channel_map=channel_map, mask=model_mask),
                extent=extent,
                origin="lower",
            )
            ax[k][panel].set_title(f"Model Source {k}", **title_kwargs)
            if center is not None and add_markers:
                ax[k][panel].plot(*center, **marker_kwargs)
            panel += 1

        if show_rendered or show_observed:
            assert observation is not None, "show_rendered or show_observed requires observation"
            extent = observation.frame.bbox.get_extent()

        # model in observation frame
        if show_rendered:
            model = scene.evaluate_source(src)
            model_ = observation.render(model)

            ax[k][panel].imshow(
                img_to_rgb(model_, norm=norm, channel_map=channel_map, mask=model_mask),
                extent=extent,
                origin="lower",
            )
            ax[k][panel].set_title(f"Model Source {k} Rendered", **title_kwargs)
            if add_markers:
                ax[k][panel].plot(*center, **marker_kwargs)
            if add_boxes:
                poly = Polygon(box_coords, closed=True, **box_kwargs)
                ax[k][panel].add_artist(poly)
            panel += 1

        if show_observed:
            # Center the observation on the source and display it
            ax[k][panel].imshow(
                img_to_rgb(observation.data, norm=norm, channel_map=channel_map),
                extent=extent,
                origin="lower",
            )
            ax[k][panel].set_title("Observation".format(), **title_kwargs)
            if add_markers:
                ax[k][panel].plot(*center, **marker_kwargs)
            if add_boxes:
                poly = Polygon(box_coords, closed=True, **box_kwargs)
                ax[k][panel].add_artist(poly)
            panel += 1

        if show_spectrum:
            # needs to be evaluated in the source box to prevent truncation
            spectra = [
                measure.flux(src),
            ] + [measure.flux(component) for component in src.components]

            for spectrum in spectra:
                ax[k][panel].plot(spectrum)
            ax[k][panel].set_xticks(range(len(spectrum)))
            if scene.frame.channels is not None:
                ax[k][panel].set_xticklabels(scene.frame.channels)
            ax[k][panel].set_title("Spectrum", **title_kwargs)
            ax[k][panel].set_xlabel("Channel")
            ax[k][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig


def scene(
    scene,
    observation=None,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_residual=False,
    add_labels=True,
    add_boxes=False,
    split_channels=False,
    fig_kwargs=None,
    title_kwargs=None,
    label_kwargs=None,
    box_kwargs=None,
):
    """Plot all sources to recreate the scene.
    The functions provide a fast way of evaluating the quality of the entire model,
    i.e. the combination of all scenes that seek to fit the observation.

    Parameters
    ----------
    scene: :py:class:`~scarlet2.Scene`
        The scene object containing the sources and their models
    observation: :py:class:`~scarlet2.Observation`, optional
        The observation containing the data
    norm: Norm
        Norm to scale the intensity of `observation` into RGB 0..256
    channel_map: array_like
        Linear mapping from channels to RGB, dimensions (3, channels)
    show_model: bool
        Whether the internal model is shown in the model frame
    show_observed: bool
        Whether the observation is shown
    show_rendered: bool
        Whether the model, rendered to match the observation, is shown
    show_residual: bool
        Whether the residuals between rendered model and observation is shown
    add_labels: bool
        Whether each source is labeled with its numerical index in the source list
    add_boxes: bool
        Whether each source box is shown
    split_channels: bool
        Whether to split the observation into separate channels
    fig_kwargs: dict
        kwargs for plt.figure()
    title_kwargs: dict
        kwargs for plt.title()
    label_kwargs: dict
        kwargs for source labels, default {"color": "w", "ha": "center", "va": "center"}
    box_kwargs: dict
        kwargs for source boxes, default {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    Returns
    -------
    mpl.Figure
    """

    if fig_kwargs is None:
        fig_kwargs = {}
    if title_kwargs is None:
        title_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {"color": "w", "ha": "center", "va": "center"}
    if box_kwargs is None:
        box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    # for animations with multiple scenes
    if hasattr(scene, "__iter__"):
        scenes = scene
        scene = scenes[0]

    if show_observed or show_rendered or show_residual:
        assert observation is not None, "Provide matched observation to show observed frame"

    rows = len(observation.frame.channels) if split_channels else 1
    panels = sum((show_model, show_observed, show_rendered, show_residual))
    figsize = fig_kwargs.pop("figsize", None)
    if figsize is None:
        figsize = (panel_size * panels, panel_size * rows)
    fig, ax = plt.subplots(rows, panels, figsize=figsize, squeeze=False, **fig_kwargs)

    model = scene()
    if show_rendered or show_residual:
        model_rendered = observation.render(model)
    if show_model and observation is not None:
        c = ChannelRenderer(scene.frame, observation.frame)
        model = c(model)
    if show_observed or show_residual:
        data = observation.data
        mask = observation.weights == 0

    for row in range(rows):
        if split_channels:
            sel = row
            name = observation.frame.channels[row]
            channel_map = None
        else:
            sel = slice(None)
            name = ""

        panel = 0
        if show_model:
            extent = scene.frame.bbox.get_extent()
            # if scene.frame.wcs is not None:
            #     extent = scene.frame.get_pixel(
            #         scene.frame.get_sky_coord(np.array([[extent[0], extent[1]], [extent[2], extent[3]]]))
            #     ).flatten()

            model_img = ax[row, panel].imshow(
                img_to_rgb(model[sel], norm=norm, channel_map=channel_map),
                extent=extent,
                origin="lower",
            )
            ax[row, panel].set_title(f"Model {name}", **title_kwargs)
            panel += 1

        if show_rendered:
            rendered_img = ax[row, panel].imshow(
                img_to_rgb(model_rendered[sel], norm=norm, channel_map=channel_map),
                origin="lower",
            )
            ax[row, panel].set_title("Model Rendered", **title_kwargs)
            panel += 1

        if show_observed or show_rendered:
            if split_channels:  # noqa: SIM108
                mask_ = mask[sel]
            else:
                # Mask any pixels with zero weight in all channels
                mask_ = np.sum(mask, axis=0) > 0
            if np.all(mask_ == 0):
                mask_ = None

        if show_observed:
            _ = ax[row, panel].imshow(
                img_to_rgb(data[sel], norm=norm, channel_map=channel_map, mask=mask_),
                origin="lower",
            )
            ax[row, panel].set_title("Observation", **title_kwargs)
            panel += 1

        if show_residual:
            residual = data[sel] - model_rendered[sel]
            norm_ = LinearPercentileNorm(residual)
            residual_img = ax[row, panel].imshow(
                img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask_),
                origin="lower",
            )
            ax[row, panel].set_title("Data - Model", **title_kwargs)
            panel += 1

        for k, src in enumerate(scene.sources):
            start, stop = src.bbox.spatial.start[::-1], src.bbox.spatial.stop[::-1]

            if add_boxes:
                panel = 0
                if show_model:
                    extent = [start[0], stop[0], start[1], stop[1]]
                    rect = Rectangle(
                        (extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2], **box_kwargs
                    )
                    ax[row, panel].add_artist(rect)
                    panel = 1
                if observation is not None:
                    start = observation.frame.get_pixel(scene.frame.get_sky_coord(np.array(start))).flatten()
                    stop = observation.frame.get_pixel(scene.frame.get_sky_coord(np.array(stop))).flatten()
                    box_coords = (start, (start[0], stop[1]), stop, (stop[0], start[1]))
                    for panel in range(panel, panels):  # noqa: B020
                        poly = Polygon(box_coords, closed=True, **box_kwargs)
                        ax[row, panel].add_artist(poly)

            if add_labels:
                center = np.array(src.center)[::-1]
                panel = 0
                if show_model:
                    ax[row, panel].text(*center, k, **label_kwargs)
                    panel = 1
                if observation is not None:
                    center = observation.frame.get_pixel(scene.frame.get_sky_coord(center)).flatten()
                    for panel in range(panel, panels):  # noqa: B020
                        ax[row, panel].text(*center, k, **label_kwargs)

    fig.tight_layout()

    try:
        # animate multiple scenes
        n_frames = len(scenes)

        # update only images dependent on the current state of scene
        def update(i):
            updated = []
            scene = scenes[i]
            model = scene()
            if show_model:
                model_img.set_data(img_to_rgb(model, norm=norm, channel_map=channel_map))
                updated.append(model_img)

            if show_rendered or show_residual:
                model = observation.render(model)

            if show_rendered:
                rendered_img.set_data(img_to_rgb(model, norm=norm, channel_map=channel_map, mask=mask))
                updated.append(rendered_img)

            if show_residual:
                residual = observation.data - model
                norm_ = LinearPercentileNorm(residual)
                residual_img.set_data(img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask))
                updated.append(residual_img)
            return updated

        ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=30)
        return ani

    except NameError:
        return fig


def mcmc_scene(obs, scene, samples, vmin=None, vmax=None, figsize=(15,5)):
    """
    Plot data, posterior‐mean model, and residual for one channel,
    overlaying source centers from the MCMC chains.

    Parameters
    ----------
    obs : scarlet2.Observation
        Your observation object (contains data & PSF).
    scene : scarlet2.Scene
        The original Scene you passed to `scene.sample`.
    samples : numpyro.infer.MCMC
        The MCMC samples.
    channel : int
        Which band/channel to display.
    vmin, vmax : floats, optional
        Color limits for data & model.
    figsize : tuple
        Figure size.
    """

    # 1) Copy the scene so we don't clobber your originals
    post_scene = copy.deepcopy(scene)

    # 2) For each source, assign the posterior‐mean parameters
    for i, src in enumerate(post_scene.sources):
        # center
        key_c = f"center:{i}"
        if key_c in samples:
            mean_c = jnp.mean(samples[key_c], axis=0)
            src.center = tuple(map(float, mean_c))
        # spectrum
        key_s = f"spectrum:{i}"
        if key_s in samples:
            mean_s = jnp.mean(samples[key_s], axis=0)
            src.spectrum = mean_s
        # morphology *array* (if you sampled it)
        key_m = f"morph:{i}"
        if key_m in samples:
            mean_m = jnp.mean(samples[key_m], axis=0)
            src.morphology = mean_m
        # ellipticity & size (for parametric morphologies)
        key_e = f"ellipticity:{i}"
        if key_e in samples:
            mean_e = jnp.mean(samples[key_e], axis=0)
            src.morphology.ellipticity = mean_e
        key_sz = f"size:{i}"
        if key_sz in samples:
            mean_sz = float(jnp.mean(samples[key_sz], axis=0))
            src.morphology.size = mean_sz

    # 3) Render the model
    model = obs.render(post_scene())
    # use only unmasked pixels
    n = jnp.prod(jnp.asarray(obs.data.shape)) - jnp.sum(obs.weights == 0)
    resid = (obs.weights * (obs.data - model) ** 2).sum(axis=0) / n

    # 4) Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    im0 = axes[0].imshow(jnp.mean(obs.data,axis=0), origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Data")
    im1 = axes[1].imshow(jnp.mean(model,axis=0), origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Model (posterior mean)")
    im2 = axes[2].imshow(resid, origin="lower", cmap="RdBu_r")
    axes[2].set_title("Residual fraction")

    # Colorbars
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    # Link zoom
    for r in range(1,3):
        axes[r].sharex(axes[0]); axes[r].sharey(axes[0])

    plt.tight_layout()
    plt.show()

def _central_crop(arr2d, out_shape):
    """Center-crop arr2d to out_shape=(H,W)."""
    H, W = arr2d.shape
    h, w = out_shape
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return arr2d[y0:y0+h, x0:x0+w]

def _shift2d_bilinear(img, dy, dx):
    """Subpixel shift 2D image (bilinear). Positive dy/dx shift the image down/right."""
    import numpy as _np
    H, W = img.shape
    y = _np.arange(H)[:, None]
    x = _np.arange(W)[None, :]
    y0 = _np.floor(y - dy).astype(int)
    x0 = _np.floor(x - dx).astype(int)
    wy = (y - dy) - y0
    wx = (x - dx) - x0

    def safe_get(a, yy, xx):
        yy = _np.clip(yy, 0, H - 1)
        xx = _np.clip(xx, 0, W - 1)
        return a[yy, xx]

    I00 = safe_get(img, y0,     x0)
    I01 = safe_get(img, y0,     x0 + 1)
    I10 = safe_get(img, y0 + 1, x0)
    I11 = safe_get(img, y0 + 1, x0 + 1)

    return (1 - wy) * ((1 - wx) * I00 + wx * I01) + wy * ((1 - wx) * I10 + wx * I11)


# Helper to render a point-like source using the scene machinery
def _render_point_like_patch(obs, scene, src_prototype, stamp_size, center=None):
    """
    Render a single point-like source (copy of `src_prototype`) into the observation frame,
    using a unit spectrum, optionally at a different `center`, and return a cropped
    (H, W) patch matching the source bbox size `stamp_size`.

    This uses `scene.evaluate_source` + `obs.render` so PSF, pixel integration,
    and WCS/frame transforms are handled consistently with the likelihood.
    """
    import copy as _copy
    # copy source and set center / unit spectrum
    src = _copy.deepcopy(src_prototype)
    if center is not None:
        src.center = tuple(map(float, center))
    # unit spectrum so we're visualizing morphology-like shape
    src.spectrum = np.ones(obs.frame.C, dtype=float)

    # evaluate only this source to model frame, then render to obs frame
    model_single = scene.evaluate_source(src)           # (C, Hm, Wm)
    rendered     = obs.render(model_single)             # (C, Ho, Wo)
    patch2d      = np.asarray(rendered).mean(axis=0)    # (Ho, Wo)

    # crop around the source bbox of the **updated copy**; convert to obs-frame indices
    H, W = stamp_size

    # Use the bbox attached to the shifted copy (src), not the prototype
    # Scene plotting utilities treat bbox corners as (x, y) when converting through WCS
    start_xy_scene = np.array(src.bbox.spatial.start[::-1], dtype=float)  # (x0, y0) in scene frame
    stop_xy_scene  = np.array(src.bbox.spatial.stop[::-1],  dtype=float)  # (x1, y1) in scene frame

    # map to sky then to observation pixel coordinates (x, y)
    start_xy_sky = scene.frame.get_sky_coord(start_xy_scene).flatten()
    stop_xy_sky  = scene.frame.get_sky_coord(stop_xy_scene).flatten()
    start_xy_obs = obs.frame.get_pixel(start_xy_sky).flatten()  # (x0_obs, y0_obs)
    stop_xy_obs  = obs.frame.get_pixel(stop_xy_sky).flatten()   # (x1_obs, y1_obs)

    # Ensure increasing order and convert to integer slice indices; for numpy slicing we need (y, x)
    x0 = int(np.clip(np.floor(min(start_xy_obs[0], stop_xy_obs[0])), 0, patch2d.shape[1]))
    x1 = int(np.clip(np.ceil( max(start_xy_obs[0], stop_xy_obs[0])),  0, patch2d.shape[1]))
    y0 = int(np.clip(np.floor(min(start_xy_obs[1], stop_xy_obs[1])), 0, patch2d.shape[0]))
    y1 = int(np.clip(np.ceil( max(start_xy_obs[1], stop_xy_obs[1])),  0, patch2d.shape[0]))

    patch = patch2d[y0:y1, x0:x1]

    # If shapes don't match due to rounding, pad/crop centrally to the requested stamp
    if patch.shape != (H, W):
        # center-crop or pad to (H,W)
        ph, pw = patch.shape
        # pad if smaller
        pad_y = max(0, H - ph)
        pad_x = max(0, W - pw)
        if pad_y or pad_x:
            patch = np.pad(patch, ((pad_y // 2, pad_y - pad_y // 2),
                                   (pad_x // 2, pad_x - pad_x // 2)), mode="constant")
        # then center-crop if larger
        if patch.shape[0] > H or patch.shape[1] > W:
            patch = _central_crop(patch, (H, W))

    # Normalize to unit flux for morphology-like visualization
    s = patch.sum()
    if s > 0:
        patch = patch / s
    return patch

# ---- mcmc_diagnostics helpers (factored) ----
def _centroid(arr):
    """Return (cy, cx) intensity-weighted centroid; fallback to image center if non-positive sum."""
    arr = np.asarray(arr)
    arr = arr - arr.min()
    s = arr.sum()
    if s <= 0:
        h, w = arr.shape
        return (h / 2.0, w / 2.0)
    y_idx, x_idx = np.indices(arr.shape)
    cy = float((y_idx * arr).sum() / s)
    cx = float((x_idx * arr).sum() / s)
    return (cy, cx)

def _get_morph_stack(i, src, samples, obs, scene):
    """Build a morphology stack for source i from samples: pixel, Gaussian, or point-like."""
    key_m  = f"morph:{i}"
    key_sz = f"size:{i}"
    key_el = f"ellipticity:{i}"
    key_c  = f"center:{i}"
    H, W = src.bbox.shape[-2], src.bbox.shape[-1]
    stamp_size = (H, W)
    if key_m in samples:
        return np.asarray(samples[key_m])
    elif key_sz in samples and key_el in samples:
        sz  = np.asarray(samples[key_sz])
        ell = np.asarray(samples[key_el])
        ms = [np.asarray(GaussianMorphology(size=float(sz[j]), ellipticity=ell[j], shape=stamp_size)())
              for j in range(len(sz))]
        return np.stack(ms, axis=0)
    else:
        if key_c in samples and len(samples[key_c]) > 0:
            ctr = np.asarray(samples[key_c])
            idx = np.linspace(0, len(ctr) - 1, len(ctr), dtype=int)
            stack = []
            for j in idx:
                stack.append(_render_point_like_patch(obs, scene, src, stamp_size, center=ctr[j]))
            return np.stack(stack, axis=0)

def _align_morph_stack(morph_stack, i, samples):
    """Optionally align samples to a common reference center to reduce averaging blur."""
    if morph_stack.ndim != 3 or morph_stack.size == 0:
        return morph_stack
    key_c_this = f"center:{i}"
    if  key_c_this in samples and len(samples[key_c_this]) > 0:
        ref_c = np.asarray(samples[key_c_this]).mean(axis=0)  # (y,x)
        cs = np.asarray(samples[key_c_this])
        aligned = []
        for img, c in zip(morph_stack, cs):
            dy = float(ref_c[0] - c[0]); dx = float(ref_c[1] - c[1])
            aligned.append(_shift2d_bilinear(img, dy, dx))
        morph_stack = np.stack(aligned, axis=0)
    else:
        cents = np.array([_centroid(img) for img in morph_stack])
        ref_cy, ref_cx = cents.mean(axis=0)
        aligned = []
        for img, (cy, cx) in zip(morph_stack, cents):
            dy = float(ref_cy - cy); dx = float(ref_cx - cx)
            aligned.append(_shift2d_bilinear(img, dy, dx))
        morph_stack = np.stack(aligned, axis=0)
    return morph_stack

def _apply_uniform_crop(morph_stack, uniform_crop):
    """Center-crop each sample in the stack to uniform_crop = (h,w) or int."""
    if uniform_crop is None or morph_stack.ndim != 3:
        return morph_stack
    if isinstance(uniform_crop, int):
        target_shape = (uniform_crop, uniform_crop)
    else:
        target_shape = tuple(uniform_crop)
    return np.stack([_central_crop(img, target_shape) for img in morph_stack], axis=0)

def _posterior_residual_mean(obs, scene, samples):
    """Render posterior-mean scene and return mean residual over channels in obs frame."""
    import copy as _copy
    post_scene = _copy.deepcopy(scene)
    for j, s in enumerate(post_scene.sources):
        key_c = f"center:{j}"
        if key_c in samples:
            s.center = np.asarray(samples[key_c]).mean(axis=0)
        key_s = f"spectrum:{j}"
        if key_s in samples:
            s.spectrum = np.asarray(samples[key_s]).mean(axis=0)
        key_m = f"morph:{j}"
        key_sz = f"size:{j}"
        key_el = f"ellipticity:{j}"
        if key_m in samples:
            s.morphology = np.asarray(samples[key_m]).mean(axis=0)
        elif key_sz in samples and key_el in samples:
            mean_sz = float(np.asarray(samples[key_sz]).mean(axis=0))
            mean_el = np.asarray(samples[key_el]).mean(axis=0)
            try:
                s.morphology.size = mean_sz
                s.morphology.ellipticity = mean_el
            except Exception:
                s.morphology = GaussianMorphology(mean_sz, mean_el, shape=(s.bbox.shape[-2], s.bbox.shape[-1]))
    model_post = obs.render(post_scene())
    residual_cube = obs.data - model_post
    return residual_cube.mean(axis=0)

def _crop_residual_to_src(residual2d, src_obj, scene, obs):
    """Crop a 2D residual map (obs frame) to the bbox of src_obj, returning a (H,W) patch."""
    start_yx = np.array(src_obj.bbox.spatial.start, dtype=float)
    stop_yx  = np.array(src_obj.bbox.spatial.stop,  dtype=float)
    start_xy = start_yx[::-1]; stop_xy = stop_yx[::-1]
    start_sky = scene.frame.get_sky_coord(start_xy).flatten()
    stop_sky  = scene.frame.get_sky_coord(stop_xy).flatten()
    start_xy_obs = obs.frame.get_pixel(start_sky).flatten()
    stop_xy_obs  = obs.frame.get_pixel(stop_sky).flatten()
    x0 = int(np.clip(np.floor(min(start_xy_obs[0], stop_xy_obs[0])), 0, residual2d.shape[1]))
    x1 = int(np.clip(np.ceil( max(start_xy_obs[0], stop_xy_obs[0])),  0, residual2d.shape[1]))
    y0 = int(np.clip(np.floor(min(start_xy_obs[1], stop_xy_obs[1])), 0, residual2d.shape[0]))
    y1 = int(np.clip(np.ceil( max(start_xy_obs[1], stop_xy_obs[1])),  0, residual2d.shape[0]))
    patch = residual2d[y0:y1, x0:x1]
    H, W = src_obj.bbox.shape[-2], src_obj.bbox.shape[-1]
    if patch.shape != (H, W):
        ph, pw = patch.shape
        pad_y = max(0, H - ph); pad_x = max(0, W - pw)
        if pad_y or pad_x:
            patch = np.pad(patch, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2)), mode="constant")
        if patch.shape[0] > H or patch.shape[1] > W:
            patch = _central_crop(patch, (H, W))
    return patch

def _compute_norms(morph_means, morph_stds):
    """Build global Normalize objects from lists of mean/std images (robust percentiles)."""
    all_mean_pixels = np.concatenate([m.ravel() for m in morph_means]) if morph_means else np.array([0.0])
    all_std_pixels  = np.concatenate([s.ravel() for s in morph_stds])  if morph_stds  else np.array([0.0])
    mean_norm = Normalize(vmin=np.percentile(all_mean_pixels, 2), vmax=np.percentile(all_mean_pixels, 98))
    std_norm  = Normalize(vmin=0.0, vmax=np.percentile(all_std_pixels, 98))
    return mean_norm, std_norm

def _plot_source_row(ax_row, i, mean_img, std_img, spec_samples,
                     resid_patch, mean_norm, std_norm, channels, center_samples=None):
    """Plot one row: mean, std, spectrum, center trace, residual patch."""
    ax_mean, ax_std, ax_spec, ax_ctr, ax_resid = ax_row
    im0 = ax_mean.imshow(mean_img, origin='lower', norm=mean_norm)
    im1 = ax_std.imshow(std_img, origin='lower', cmap='magma', norm=std_norm)
    ax_mean.set_title(f"Source {i} mean", fontsize=10)
    ax_std.set_title(f"Source {i} std", fontsize=10)

    for ax in (ax_mean, ax_std):
        ax.set_aspect('equal'); ax.tick_params(labelsize=8, length=2)
        ax.set_xlabel("x [pix]", fontsize=8); ax.set_ylabel("y [pix]", fontsize=8)

    # Spectrum
    if spec_samples is not None and len(spec_samples) > 0:
        spec = np.asarray(spec_samples)
        mu, sig = spec.mean(axis=0), spec.std(axis=0)
        x = np.arange(mu.size)
        ax_spec.plot(x, mu, '-o', markersize=3)
        ax_spec.fill_between(x, mu - sig, mu + sig, alpha=0.25, linewidth=0)
        ymax = float(np.max(mu + sig)) if mu.size else 1.0
        ax_spec.set_ylim(0, 1.05*ymax)
        ax_spec.set_xticks(x)
        if channels is not None and len(channels) == mu.size:
            ax_spec.set_xticklabels(channels, rotation=0, fontsize=9)
    ax_spec.tick_params(labelsize=9); ax_spec.grid(alpha=0.2, linestyle=':', linewidth=0.7)
    ax_spec.set_title(f"Spectrum {i}", fontsize=10)
    # Center trace
    if center_samples is not None and len(center_samples) > 0:
        ctr = np.asarray(center_samples)
        ax_ctr.plot(ctr[:, 0], '.', ms=2, label='y');ax_ctr.plot(ctr[:, 1],  '.', ms=2, label='x')
        if i == 0:
            ax_ctr.legend(frameon=False, fontsize=9, loc='upper right')
        ax_ctr.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax_ctr.set_title(f"Center {i}", fontsize=10); ax_ctr.tick_params(labelsize=9)
    ax_ctr.grid(alpha=0.2, linestyle=':', linewidth=0.7)
    # Residual
    v = np.percentile(np.abs(resid_patch), 98) if np.isfinite(resid_patch).any() else 1.0
    ax_resid.imshow(resid_patch, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
    ax_resid.set_title(f"Residual ⟨data−model⟩ {i}", fontsize=10)

    ax_resid.set_aspect('equal'); ax_resid.tick_params(labelsize=8, length=2)
    ax_resid.set_xlabel("x [pix]", fontsize=8); ax_resid.set_ylabel("y [pix]", fontsize=8)

    return im0, im1


def mcmc_diagnostics(obs, scene, samples, centers, figsize=(10, 2.6),
                     recenter=True, 
                     uniform_crop=None):
    """
    For each source (row):
      [0] morphology mean   [1] morphology std   [2] spectrum (mean ± 1σ)   [3] center trace   [4] residual

    Parameters
    ----------
    obs : scarlet2.Observation
        The observation object.
    scene : scarlet2.Scene
        The scene object.
    mcmc : numpyro.infer.MCMC
        The MCMC samples.
    centers : list
        List of centers for each source.
    figsize : tuple, optional
        Figure size.
    recenter : bool, optional
        Whether to recenter morphologies.
    uniform_crop : tuple or int, optional
        Uniform crop size.
    """
    nsrc = len(scene.sources)
    ncols = 5

    # Build stacks → align → optional crop → collect mean/std
    morph_means, morph_stds = [], []
    for i, src in enumerate(scene.sources):
        stack = _get_morph_stack(i, src, samples, obs, scene)
        if recenter:
            stack = _align_morph_stack(stack, i, samples)
        stack = _apply_uniform_crop(stack, uniform_crop)
        morph_means.append(stack.mean(axis=0))
        morph_stds.append(stack.std(axis=0))

    # Posterior-mean residual map in obs frame
    residual_mean_global = _posterior_residual_mean(obs, scene, samples)

    # Global color norms
    mean_norm, std_norm = _compute_norms(morph_means, morph_stds)

    # Figure & axes
    fig, axes = plt.subplots(
        nsrc, ncols,
        figsize=(figsize[0]*ncols, figsize[1]*nsrc),
        constrained_layout=True,
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05,
                     'width_ratios': [1, 1, 1.2, 1.4, 1.2]}
    )
    if nsrc == 1:
        axes = axes[np.newaxis, :]

    im_mean_refs, im_std_refs = [], []
    for i in range(nsrc):
        if f"center:{i}" in samples.keys():
            center_samples=(np.asarray(samples.get(f"center:{i}", None)) - np.asarray(centers[i]))
        else:
            center_samples=(np.ones((len(samples),2)) - np.asarray(centers[i]))
        resid_patch = _crop_residual_to_src(residual_mean_global, scene.sources[i], scene, obs)
        im0, im1 = _plot_source_row(
            axes[i], i,
            morph_means[i], morph_stds[i],
            samples.get(f"spectrum:{i}", None),
            resid_patch, mean_norm, std_norm, scene.frame.channels,
            center_samples=center_samples,
        )
        im_mean_refs.append(im0); im_std_refs.append(im1)

    # Colorbars
    fig.colorbar(im_mean_refs[0], ax=axes[:, 0], fraction=1, pad=0.15)
    fig.colorbar(im_std_refs[0],  ax=axes[:, 1], fraction=1, pad=0.15)


    # Link zoom
    for r in range(nsrc):
        axes[r, 1].sharex(axes[r, 0]); axes[r, 1].sharey(axes[r, 0])

    plt.show()
    return fig, axes

def corner_centers_spectra(samples, scene, sources=None, channels="all", thin=1, max_samples=5000):
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