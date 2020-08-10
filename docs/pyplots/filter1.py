from slab import Signal, Sound, Filter
from matplotlib import pyplot as plt
Signal.set_default_samplerate(44100)
# generate sound and filter it
sound = Sound.whitenoise()
filt = Filter.cutoff_filter(frequency=12000, kind='lp')
sound_filt = filt.apply(sound)
# plot the result
_, ax = plt.subplots(2, sharex=True, sharey=True)
sound.spectrum(show=False, axes=ax[0], color="blue", label="unfiltered")
sound_filt.spectrum(show=False, axes=ax[1], color="red", label="after lowpass")
ax[1].axvline(12000, color="black", linestyle="--")
ax[0].legend()
ax[1].legend()
ax[1].set(title=None, xlabel="Frequency [Hz]")
plt.show()
