.. currentmodule:: slab

.. _filters:

Filters
=======

The :class:`Filter` class can be used to generate, manipulate and save filter banks and transfer functions. Filters are represented internally as :class:`Signal` and come in two flavours: finite impulse responses (FIR) and frequency bin amplitudes (FFT). The :attr:`fir` (True or False).

Simple Filters
--------------
Simple low-, high-, bandpass, and bandstop filters can be used to suppress selected frequency bands in a sound. For example, if you don't want the sound to contain power above 1 kHz, apply a 1 kHz lowpass filter:

.. plot::
    :include-source:

    from matplotlib import pyplot as plt
    sound = slab.Sound.whitenoise()
    filt = slab.Filter.cutoff_filter(frequency=1000, kind='lp')
    sound_filt = filt.apply(sound)
    _, [ax1, ax2] = plt.subplots(2, sharex=True)
    sound.spectrum(axis=ax1)
    sound_filt.spectrum(axis=ax2)

The :meth:`~Sound.filter` of the :class:`Sound` class wraps around :meth:`Filter.cutoff_filter` and :meth:`Filter.apply` so that you can use these filters conveniently from within the :class:`Sound` class.

Filter design is tricky and it is good practice to plot and inspect the transfer function of the filter:

.. plot:: pyplots/filter2.py
  :include-source:

While filtering did not cause any visible artifacts, it reduced the amplitude of the signal.
This is to be expected because by filtering we remove part of the signal and thus loose power. A naive approach would
be to correct this by setting the level of the filtered sound equal to that of the original. However, this
is not recommended because our perception of loudness is non-linear with respect to frequency.
This problem will be addressed later

Applying Multiple Filters
-------------------------
Slab features multi-channel filtering - you can easily apply multiple filters to one signal,
one filer to multiple signals or a bunch of filters to a bunch of signals! The **apply** method will
choose what to do depending on the number of channels in the filter and signal. If signal and filter
have the same number of channels each channel of the filter is applied to the corresponding signal.
If a multi-channel filter is applied to a one-channel signal, each filter channel is applied
to a copy of the signal so the resulting filtered signal has the same number of channels as the filter.
This can be used, for example, to create a set of filtered noise with different spectra

.. plot:: pyplots/filter3.py
  :include-source:
