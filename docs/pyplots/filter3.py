from slab import Signal, Sound, Filter
from matplotlib import pyplot as plt
import numpy
Signal.set_default_samplerate(44100)

sound = Sound.whitenoise()
# make filter bank with 16 bandpass-filters of width 100 Hz
start, stop, n = 500, 2000, 16
low_cutoff = numpy.linspace(start, stop, n)
high_cutoff = numpy.linspace(start, stop, n)+100
filters = []
for i in range(n):
    filters.append(Filter.cutoff_filter(
        frequency=(low_cutoff[i], high_cutoff[i]), kind='bp'))
fbank = Filter(filters)  # put the list into a single filter object
sound_filt = fbank.apply(sound)
_, ax = plt.subplots(1)
sound_filt.spectrum(axes=ax, show=False)
ax.set_xlim(100, 5000)
plt.show()
