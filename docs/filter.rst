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
the sound and then apply a 12 kHz lowpass Filter.

.. plot::
  from slab import Signal, Sound, Filter
  from matplotlib import pyplot as plt
  Signal.set_default_samplerate(44100)
  # generate sound and filter it
  sound = Sound.whitenoise()
  filt = Filter.cutoff_filter(frequency=12000, kind='lp')
  sound_filt = filt.apply(sound)
  # plot the result
  _, ax = plt.subplots(2, sharex=True, sharey=True)
  sound.spectrum(show=False, axis=ax[0], color="blue", label="unfiltered")
  sound_filt.spectrum(show=False, axis=ax[1], color="red", label="after lowpass")
  ax[1].axvline(12000, color="black", linestyle="--")
  ax[0].legend()
  ax[1].legend()
  ax[1].set(title=None, xlabel="Frequency [Hz]")
  plt.show()


After filtering the sound does not carry any power above 12 kHz. Since filtering can cause artifacts
in the time domain, it is good practice to always plot and inspect the filtered signal

.. plot::
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


While filtering did not cause any visible artifacts, it reduced the amplitude of the signal.
This is to be expected because by filtering we remove part of the signal and thus loose power. A naive approach would
be to correct this by setting the level of the filtered sound equal to that of the original. However, this
is not recommended because our perception of loudness is non-linear with respect to frequency.
This problem will be addressed later

Applying Multiple Filters
-------------------------
Slab features multi-channel filtering - you can easily apply multiple filters to one signal,
one filer to multiple signals or a bunch of filters to a bunch of signals! The **apply** method will
choose what to do depending on the number of channels in the filter and signal.
If a multi-channel filter is applied to a one-channel signal, each filter channel is applied
to a copy of the signal so the resulting filtered signal has the same number of channels as the filter.
This can be used, for example, to create a set of filtered noise with different spectra

.. plot::
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
  sound_filt = fbank.apply(sound)  # apply each filter to a copy of sound
  # plot the spectra, each color represents one channel of the filtered sound
  _, ax = plt.subplots(1)
  sound_filt.spectrum(axes=ax, show=False)
  ax.set_xlim(100, 5000)
  plt.show()

If the a one-channel filter is applied to a multi-channel signal, the filter will be applied to each
channel individually. This can be used, for example, to easily pre-process a set of recordings (where
every recording is represented by a channel in the :class:`slab.Sound` object). If a multi-channel filter
is applied to a multi-channel signal with the same number of channels each filter channel is applied to
the corresponding signal channel. This is useful for e.g. equalization of a set of loudspeakers

Applying Multiple Filters
-------------------------

In Psychoacoustic experiments, we are often interested in the effect of a specific feature. One could,
for example, take the bandpass filtered sounds from the example above and investigate how well listeners
can discriminate them from a noisy background - a typical cocktail-party task. However, if the transfer
function of the loudspeakers or headphones used in the experiment is not flat, the finding will be biased.
Imagine that the headphones used were bad at transmitting frequencies below 1000 Hz. This would make a sound
with center frequency of 550 Hz harder to detect than one with a center frequency of 1550 Hz. We can prevent
this by inverting the headphones transfer function and using that as a filter. The inverse transfer function
filter and the actual transfer function will cancel each other out and the result will be an equalized sound.

.. plot::
