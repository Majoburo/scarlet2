from scarlet2.io import load_session_h5
from scarlet2 import plot

scene2, obs2, samples, meta = load_session_h5("obj.h5", id=0, path="runs")

plot.mcmc_scene(obs2, scene2, samples)
plot.mcmc_diagnostics(obs2, scene2, samples, centers=meta["centers"],recenter=False)
plot.corner_centers_spectra(samples, scene2)
