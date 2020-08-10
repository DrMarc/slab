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

.. plot:: pyplots/filter1.py
  :include-source:

After filtering the sound does not carry any power above 12 kHz. Since filtering can cause artifacts
in the time domain, it is good practice to always plot and inspect the filtered signal

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
