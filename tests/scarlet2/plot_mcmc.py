from scarlet2.io import load_session_h5
from scarlet2 import plot
from matplotlib import pyplot as plt
import sys

scene2, obs2, samples, meta = load_session_h5(sys.argv[1], id=0, path="runs")

fig,ax = plot.mcmc_scene(obs2, scene2, samples)
plt.show()
fig,ax =plot.mcmc_diagnostics(obs2, scene2, samples, centers=meta["centers"], recenter=False)
plt.show()
fig,ax =plot.corner_centers_spectra(samples, scene2)
plt.show()
