import numpy as jnp
import numpy.ma as ma

from .module import Module


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if isinstance(component, Module):
        model = component()
        origin = component.bbox.origin
    else:
        model = component
        origin = 0

    return tuple(jnp.array(jnp.unravel_index(jnp.argmax(model), model.shape)) + origin)


def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if isinstance(component, Module):
        model = component()
    else:
        model = component

    return model.sum(axis=(1, 2))


def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet2.Component` or array
        Component to analyze or its hyperspectral model
    """
    if isinstance(component, Module):
        model = component()
    else:
        model = component
    indices = jnp.indices(model.shape)
    centroid = jnp.array([jnp.sum(ind * model) for ind in indices]) / model.sum()
    return centroid


def snr(component, observations):
    """Determine SNR with morphology as weight function

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model

    observations: `scarlet.Observation` or list thereof
    """
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    if hasattr(component, "get_model"):
        frame = None
        if not prerender:
            frame = observations[0].model_frame
        model = component.get_model(frame=frame)
    else:
        model = component

    M = []
    W = []
    var = []
    # convolve model for every observation;
    # flatten in channel direction because it may not have all C channels; concatenate
    # do same thing for noise variance
    for obs in observations:
        noise_rms = 1 / jnp.sqrt(ma.masked_equal(obs.weights, 0))
        ma.set_fill_value(noise_rms, jnp.inf)
        model_ = obs.render(model)
        M.append(model_.reshape(-1))
        W.append((model_ / (model_.sum(axis=(-2, -1))[:, None, None])).reshape(-1))
        noise_var = noise_rms ** 2
        var.append(noise_var.reshape(-1))
    M = jnp.concatenate(M)
    W = jnp.concatenate(W)
    var = jnp.concatenate(var)

    # SNR from Erben (2001), eq. 16, extended to multiple bands
    # SNR = (I @ W) / sqrt(W @ Sigma^2 @ W)
    # with W = morph, Sigma^2 = diagonal variance matrix
    snr = (M * W).sum() / jnp.sqrt(((var * W) * W).sum())

    return snr


# adapted from https://github.com/pmelchior/shapelens/blob/master/src/Moments.cc
def moments(component, N=2, center=None, weight=None):
    """Determine SNR with morphology as weight function

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    N: int >=0
        Moment order
    center: array
        2D coordinate in frame of `component`
    weight: array
        weight function with same shape as `component`
    """
    if isinstance(component, Module):
        model = component()
    else:
        model = component

    if weight is None:
        weight = 1
    else:
        assert model.shape == weight.shape

    if center is None:
        center = centroid(model)

    grid_x, grid_y = jnp.indices(model.shape[-2:], dtype=jnp.float64)
    if len(model.shape) == 3:
        grid_y = grid_y[None, :, :]
        grid_x = grid_x[None, :, :]
    grid_y -= center[0]
    grid_x -= center[1]

    M = dict()
    for n in range(N + 1):
        for m in range(n + 1):
            # moments ordered by power in y, then x
            M[m, n - m] = (grid_y ** m * grid_x ** (n - m) * model * weight).sum(
                axis=(-2, -1)
            )
    return M


def size(moments):
    """Determine size from moments

    Parameters
    ----------
    moments: moment dictionary from moments()
    """
    flux = moments[(0, 0)]
    T = (moments[(0, 2)] / flux * moments[(2, 0)] / flux - (moments[(1, 1)] / flux) ** 2) ** (1 / 4)
    return T


def ellipticity(moments):
    """Determine complex ellipticity from moments.

    Parameters
    ----------
    moments: moment dictionary from moments()

    Returns:
    --------
    jnp.array
    """
    ellipticity = (moments[(2, 0)] - moments[(0, 2)] + 2j * moments[(1, 1)]) / (moments[(2, 0)] + moments[(0, 2)])
    return jnp.array((ellipticity.real, ellipticity.imag))


# adapted from  https://github.com/pmelchior/shapelens/blob/src/DEIMOS.cc
def binomial(n, k):
    if k == 0:
        return 1
    if k > n // 2:
        return binomial(n, n - k)
    result = 1
    for i in range(1, k + 1):
        result *= n - i + 1
        result //= i
    return result


# moments of the Gaussian
def deconvolve(g, p):
    """Deconvolve moments of a Gaussian from moments of a general distribution"""

    # Hard code as 2 for now
    Nmin = 2  # min(p.shape[0], g.shape[0])

    # use explicit relations for up to 2nd moments
    g[(0, 0)] /= p[(0, 0)]
    if Nmin >= 1:
        g[(0, 1)] -= g[(0, 0)] * p[(0, 1)]
        g[(1, 0)] -= g[(0, 0)] * p[(1, 0)]
        g[(0, 1)] /= p[(0, 0)]
        g[(1, 0)] /= p[(0, 0)]
        if Nmin >= 2:
            g[(0, 2)] -= g[(0, 0)] * p[(0, 2)] + 2 * g[(0, 1)] * p[(0, 1)]
            g[(1, 1)] -= g[(0, 0)] * p[(1, 1)] + g[(0, 1)] * p[(1, 0)] + g[(1, 0)] * p[(0, 1)]
            g[(2, 0)] -= g[(0, 0)] * p[(2, 0)] + 2 * g[(1, 0)] * p[(1, 0)]
            if Nmin >= 3:
                # use general formula
                for n in range(3, Nmin + 1):
                    for i in range(n + 1):
                        for j in range(n - i):
                            for k in range(i):
                                for l in range(j):
                                    g[(i, j)] -= (
                                        binomial(i, k)
                                        * binomial(j, l)
                                        * g[k, l]
                                        * p[i - k, j - l]
                                    )
                            for k in range(i):
                                g[(i, j)] -= binomial(i, k) * g[k, j] * p[i - k, 0]
                            for l in range(j):
                                g[(i, j)] -= binomial(j, l) * g[i, l] * p[0, j - l]
                    g[(i, j)] /= p[(0, 0)]

    return g
