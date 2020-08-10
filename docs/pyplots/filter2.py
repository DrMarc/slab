from slab import Signal, Sound, Filter
from matplotlib import pyplot as plt
Signal.set_default_samplerate(44100)
sound = Sound.whitenoise()
filt = Filter.cutoff_filter(frequency=12000, kind='lp')
sound_filt = filt.apply(sound)
_, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(sound.times, sound.data, color="blue", label="unfiltered")
ax[1].plot(sound_filt.times, sound_filt.data, color="red", label="after lowpass")
[a.set(xlabel="Time in Seconds", ylabel="Amplitude") for a in ax]
[a.legend() for a in ax]
plt.show()
