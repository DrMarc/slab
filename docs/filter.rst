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
    :context:

    from matplotlib import pyplot as plt
    sound = slab.Sound.whitenoise()
    filt = slab.Filter.band(frequency=1000, kind='lp')
    sound_filt = filt.apply(sound)
    _, [ax1, ax2] = plt.subplots(2, sharex=True)
    sound.spectrum(axis=ax1, color="blue")
    sound_filt.spectrum(axis=ax2, color="red")

The :meth:`~Sound.filter` of the :class:`Sound` class wraps around :meth:`Filter.cutoff_filter` and :meth:`Filter.apply` so that you can use these filters conveniently from within the :class:`Sound` class.

Filter design is tricky and it is good practice to plot and inspect the transfer function of the filter:

.. plot::
    :include-source:
    :context: close-figs

    filt.tf()


Inspecting the waveform of the sound (using the :meth:`~slab.Sound.waveform` method) or the rms level (using the :attr:`slab.Sound/level` attribute) shows that the amplitude of the filtered signal is smaller than that of the original, because the filter has removed power. You might be tempted to correct this difference by increasing the level of the filtered sound, but this is not recommended because the perception of intensity (loudness) depends non-linearly on the frequency content of the sound.

Filter banks
------------
A :class:`Filter` objects can hold multiple channels, just like a :class:`Sound` object. The :meth:`.cos_filterbank` and :meth:`.equalizing_filterbank` methods construct multi-channel filter banks for you, but you can also initialize a :class:`Filter` object with a list of filters, just as you can make a multi-channel sound by initializing a :class:`Sound` object with a list of sounds (both classes inherit this feature from the parent :class:`Signal` class):

.. plot::
    :include-source:

    filters = []
    for freq in range(200, 3000, 300):
        filters.append(slab.Filter.band(frequency=(freq, freq+200), kind='bp'))
    fbank = slab.Filter(filters)
    fbank.tf()

If this multi-channel filter is applied to a one-channel signal, each filter channel is applied separately and the resulting signal has the same number of channels as the filter (subbands). You can modify these subbands and re-combine them using the :meth:`.combine_subbands` method. An example of this process is the vocoder implementation in the :class:`Sound` class, which uses these features of the :class:`Filter` class. The multi-channel filter is generated with :meth:`.cos_filterbank`, which produces cosine-shaped filters that divide the sound into small frequency bands which are spaced in a way that mimics the filters of the human auditory periphery (`equivalent rectangular bandwidth, ERB <https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth>`_). Here is an example of the transfer functions of this filter bank:

.. plot::
    :include-source:

    fbank = slab.Filter.cos_filterbank()
    fbank.tf()

A speech signal is filtered with this bank, and the envelopes of the subbands are computed using the :meth:`envelope` method of the :class:`Signal` class. The envelopes are filled with noise, and the subbands are collapsed back into one sound. This removes most spectral information but retains temporal information in a speech signal and sound a bit like whispered speech. Here is the code of the :meth:`~slab.Sound.vocode` method. Some arguments that make sure that all lengths and sample rates fit have been redacted for clarity::

    fbank = slab.Filter.cos_filterbank()
    subbands = fbank.apply(signal)
    envs = subbands.envelope()
    noise = slab.Sound.whitenoise()
    subbands_noise = fbank.apply(noise)
    subbands_noise *= envs  # fill envelopes with noise
    subbands_noise.level = subbands.level # keep subband level of original
    return Sound(slab Filter.collapse_subbands(subbands_noise, filter_bank=fbank))


Equalization
------------
Psychoacoustic experiments with stimuli that contain several frequencies require presentation hardware (headphones or loudspeakers) that can transmit the different frequencies equally well. You can measure the transfer function of your system by playing a wide-band sound, like a chirp, and recording it with a probe microphone (which itself must have a flat transfer function). From this recording, you can calculate the transfer function and construct an inverse filter. Apply the inverse filter to a sound before playing it through that system to compensate for the uneven transfer, because the inverse filter and the actual transfer function cancel each other. The :meth:`~slab.Filter.equalizing_filterbank` method does most of this work for you. For a demonstration, we simulate a (pretty bad) loudspeaker transfer function by applying a random filter:

.. plot::
    :include-source:
    :context:

    import random
    freqs = [f * 800 for f in range(5)]
    gain = [random.uniform(.3, 1) for _ in range(5)]
    tf = slab.Filter.band(frequency=freqs, gain=gain)
    sound = slab.Sound.whitenoise()
    recording = tf.apply(sound)
    recording.spectrum()

With the original sound and the simulated recording we can compute an inverse filter und pre-filter the sound (or in this case, just filter the recording) to achieve a nearly flat playback through our simulated bad loudspeaker:

.. plot::
    :include-source:
    :context: close-figs

    inverse = slab.Filter.equalizing_filterbank(target=sound, signal=recording)
    equalized = inverse.apply(recording)
    equalized.spectrum()

If there are multiple channels in your recording (assembled from recordings of the same white noise through several loudspeakers, for instance) then the :meth:`~slab.Filter.equalizing_filterbank` method returns a filter bank with one inverse filter for each signal channel, which you can :meth:`~slab.Filter.apply` just as in the example above.
