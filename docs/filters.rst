.. currentmodule:: slab

Filters
=======

The **Filter** class (inherits from :class:`slab.Signal`) can be used to generate, manipulate and save
filter banks and transfer functions. Two typical use cases are lowpass/highpass filtering and
loudspeaker equalization.


Cut-Off Filtering
-----------------

The aim of a Cut-Off filter is to suppress certain parts of a signals power spectrum. For example, if we
don't want our sound to contain power above 12 kHz (maybe our speakers can't go higher), we can generate
the sound and then apply a 12 kHz lowpass Filter. ::

  from slab import Signal, Sound, Filter
  from matplotlib import pyplot as plt
  Signal.set_default_samplerate(44100)

  sound = Sound.whitenoise(samplerate=44100)
  filt = Filter.cutoff_filter(frequency=12000, kind='lp')
  sound_filt = filt.apply(sound)

  _, ax = plt.subplots(1)
  sound.spectrum(show=False, axes=ax, color="blue", label="unfiltered")
  sound_filt.spectrum(show=False, axes=ax, color="red", label="after lowpass")
  ax.legend()
  plt.show()

After filtering the sound does not carry any power above 12 kHz. Since filtering can cause artifacts
in the time domain, it is recommended to always plot and inspect the filtered signal ::

  _, ax = plt.subplots(2, sharex=True, sharey=True)
  ax[0].plot(sound.times, sound.data, color="blue", label="unfiltered")
  ax[1].plot(sound_filt.times, sound_filt.data, color="red", label="after lowpass")
  [a.set(xlabel="Time in Seconds", ylabel="Amplitude") for a in ax]
  [a.legend() for a in ax]

While filtering did not cause any visible artifacts, it reduced the amplitude of the signal.
This is because by filtering we remove part of the signal and thus loose power. A naive approach would
be to correct this by setting the level of the filtered sound equal to that of the original. However, this
is not recommended because our perception of loudness is non-linear with respect to frequency.
This problem is adressed later

Applying Multiple Filters
-------------------------
We can apply multiple filters
