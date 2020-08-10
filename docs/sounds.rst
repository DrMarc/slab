
Sound
=====

Generating sounds
^^^^^^^^^^^^^^^^^
The **Sound** class provides methods for generating, manipulating, displaying, and analysing sound stimuli.
You can generate typical experimental stimuli with this class, including tones, noises, and click trains, and also more specialized stimuli, like equally-masking noises, Schroeder-phase harmonics, iterated ripple noise and synthetic vowels. For instance, let's make a 500ms long 500 Hz pure tone signal with a band-limited (one octave below and above the tone) pink noise background with a 10 dB signal-to-noise ratio: ::

    import slab
    tone = slab.Sound.tone(frequency=500, duration=0.5)
    tone.level = 80 # setting the intensity to 80 dB
    noise = slab.Sound.pinknoise(duration=0.5)
    noise.filter(frequency=(250, 1000), kind='bp') # bandpass .25 to 1 kHz
    noise.level = 70 # 10 dB lower than the tone
    stimulus = tone + noise # combine the two signals
    stimulus.ramp() # apply on- and offset ramps to avoid clicks
    stimulus.play()

:class:`slab.Sound` objects have many useful methods for manipulating (like :meth:`.ramp`, :meth:`.filter`, and :meth:`.pulse`) or inspecting them (like :meth:`.waveform`, :meth:`.spectrum`, and :meth:`.spectral_feature`). A complete list is in the :ref:`Reference` section, and the majority is also discussed here. If you use IPython, you can tap the `tab` key after typing ``slab.Sound.``, or the name of any Sound object followed by a full stop, to get an interactive list of the possibilities.

Sounds can also be created by recording them with :meth:`slab.Sound.record`. For instance ``recording = slab.Sound.record(duration=1.0, samplerate=44100)`` will record a 1-second sound at 44100 Hz from the default audio input (usually the microphone). The ``record`` method uses `SoundCard <https://github.com/bastibe/SoundCard>`_ if installed, or `SoX <http://sox.sourceforge.net>`_ (via a temporary file) otherwise. Both are cross-platform and easy to install. If neither tool is installed, you won't be able to record sounds.

Specifying durations
^^^^^^^^^^^^^^^^^^^^
Sometimes it is useful to specify the duration of a stimulus in **samples** rather than seconds. All functions to generate sounds have a duration argument that accepts floating point numbers or integers. Floating point numbers are interpreted as durations in seconds (``slab.Sound.tone(duration=1.0)`` results in a 1 second tone). Integers are interpreted as number of samples (``slab.Sound.tone(duration=1000)`` gives you 1000 samples of a tone).

Setting the sample rate
^^^^^^^^^^^^^^^^^^^^^^^
We did not specify a sample rate for any of the stimuli in the examples above. The default sample rate is 8000 Hz (for no particular reason other than that this is MATLAB's default), which is ok for stimuli well below 4 kHz. Instead of giving a sample rate separately for each Sound object (which is possible, most methods have a ``samplerate`` argument), you can also change the default at the start of your script or Python session. The default rate is saved in the variable ``_default_samplerate`` and can be set, for instance to the standard CD rate, with ``slab.Sound.set_default_samplerate(44100)``. You can access the variable directly as ``slab.signal._default_samplerate``.

Saving and loading sounds
^^^^^^^^^^^^^^^^^^^^^^^^^
You can save sounds to wav files by calling the object's :meth:`.Sound.write` method (``signal.write('signal.wav')``). By default, sounds are normalized to have a maximal amplitude of 1 to avoid clipping when writing the file. You should set ``signal.level`` to the intended level when loading a sound from file or disable normalization if you know what you are doing. You can load a wav file by initializing a Sound object with the filename: ``signal = slab.Sound('signal.wav')``.

Combining sounds
^^^^^^^^^^^^^^^^
Several functions allow you to string stimuli together. For instance, in a forward masking experiment [#f1]_ we need a masking noise followed by a target sound after a brief silent interval. An example implementation of a complete experiment is discussed in the :ref:`Psychoacoustics` section, but here, we will construct the stimulus: ::

    masker = slab.Sound.tone(frequency=550, duration=0.5) # a 0.5s 550 Hz tone
    masker.level = 80 # at 80 dB
    masker.ramp() # default 10 ms raised cosine ramps
    silence = slab.Sound.silence(duration=0.01) # 10 ms silence
    signal = slab.Sound.tone(duration=0.05) # using the default 500 Hz
    signal.level = 80 # let's start at the same intensity as the masker
    signal.ramp(duration=0.005) # short signal, we'll use 5 ms ramps
    stimulus = slab.Sound.sequence(masker, silence, signal)
    stimulus.play()

We can make a classic non-interactive demonstration of forward masking by playing these stimuli with decreasing signal level in a loop, once without the masker, and once with the masker. Count for how many steps you can hear the signal tone: ::

    import time # we need the sleep function
    for level in range(80, 10, -5): # down from 80 in steps of 5 dB
        signal.level = level
        signal.play()
        time.sleep(0.5)
    # now with the masker
    for level in range(80, 10, -5): # down from 80 in steps of 5 dB
        signal.level = level
        stimulus = slab.Sound.sequence(masker, silence, signal)
        stimulus.play()
        time.sleep(0.5)

I can hear all of the steps without the masker, but only the first 6 or 7 with the masker. This will depend on the intensity at which you play the demo (see :ref:`Calibrating the output<calibration>` below). The :meth:`.sequence` method is an example of list unpacking---you can provide any number of sounds to be concatenated. If you have a list of sounds, call the method like so: ``slab.Sound.sequence(*[list_of_sound_objects])`` to unpack the list into function arguments.

Another method to put sounds together is :meth:`.crossfade`, which applies a crossfading between two sounds with a specified ``overlap`` in seconds. An interesting experimental use is in adaptation designs, in which one longer stimulus is played to adapt neuronal responses to its sound features, and then a new stimulus feature is introduced (but nothing else changes). Responses (measured for instance with EEG) at that point will be mostly due to that feature. A classical example is the pitch onset response, which is evoked when the temporal fine structure of a continuous noise is regularized to produce a pitch percept without altering the sound spectrum (see `Krumbholz et al. (2003) <https://pubmed.ncbi.nlm.nih.gov/12816892/>`_). It is easy to generate the main stimulus of that study, a noise transitioning to an iterates ripple noise after two seconds, with 5 ms crossfade overlap, then filtered between 0.8 and 3.2 kHz: ::

    slab.Sound.set_default_samplerate(16000) # we need a higher sample rate
    adapter = slab.Sound.whitenoise(duration=2.0)
    adapter.level = 80
    irn = slab.Sound.irn(frequency=125, niter=2, duration=1.0) # pitched sound
    irn.level = 80 # set to the same level
    stimulus = slab.Sound.crossfade(adapter, irn, overlap=0.005) # crossfade
    stimulus.filter(frequency=[800, 3200], kind='bp') # filter
    stimulus.ramp(duration=0.005) # 5 ms on- and offset ramps
    stimulus.spectrogram() # note that there is no change at the transition
    stimulus.play() # but you can hear the onset of the regularity (pitch)

.. _calibration:

Calibrating the output
^^^^^^^^^^^^^^^^^^^^^^
Setting the **level** property of a stimulus changes the root-mean-square of the waveform and relative changes are correct (reducing the level attribute by 10 dB will reduce the sound output by the same amount), but the *absolute* intensity is only correct if you calibrate your output. The recommended procedure it to set your system volume to maximum, connect the listening hardware (headphone or loudspeaker) and set up a sound level meter. Then call :meth:`slab.calibrate`. The ``calibrate`` method will play a 1 kHz tone for 5 seconds. Note the recorded intensity on the meter and enter it when requested. The difference between the tone's level attribute and the recorded level is saved in the class variable ``_calibration_intensity``. It is applied to all level calculations so that a sound's level attribute now roughly corresponds to the actual output intensity in dB SPL---'roughly' because your output hardware may not have a flat frequency transfer function (some frequencies play louder than others). See :ref:`Filters` for methods to equalize transfer functions. Experiments sometimes require you to play different stimuli at comparable loudness. Loudness is the perception of sound intensity and it is difficult to calculate. You can use the :meth:`Sound.aweight` method of a sound to filter it so that frequencies are weighted according to the typical human hearing thresholds. This will increase the correspondence between the rms intensity measure returned by the ``level`` attribute and the perceived loudness. However, in most cases, controlling relative intensities is sufficient. If you do not have a sound level meter, then you can present in dB HL (hearing level). For that, measure the hearing threshold of the listener at the frequency or frequencies that are presented in your experiment and play you stimuli at a set level above that threshold. You can measure the hearing threshold at one frequency (or for any broadband sound, in fact) with the few lines of code shown at the start of the :ref:`introduction<audiogram>`.

Plotting and analysis
^^^^^^^^^^^^^^^^^^^^^
You can inspect sounds by plotting the :meth:`.waveform`, :meth:`.spectrum`, or :meth:`.spectrogram`: ::

    a = slab.Sound.vowel(vowel='a')
    e = slab.Sound.vowel(vowel='e')
    i = slab.Sound.vowel(vowel='i')
    signal = slab.Sound.sequence(a,e,i)
    signal.waveform()
    signal.waveform(end=0.05) # first 50ms, you can see the glottal pulses
    signal.spectrum()
    signal.spectrogram()

Instead of plotting, :meth:`.spectrum` and :meth:`.spectrogram` will return the time frequency bins and spectral power values for further analysis if you set the ``plot`` argument to False.

You can also extract common features from sounds, such as the :meth:`.crest_factor` (a measure of how 'peaky' the waveform is), or the average :meth:`.onset_slope` (a measure of how fast the on-ramps in the sound are---important for sound localization). Features of the spectral content are bundled in the :meth:`.spectral_feature` method. It can compute spectral centroid, flux, flattness, and roll-off. When working with environmental sounds or other recorded stimuli, one often needs to compute relevant features for collections of recordings in different experimental conditions. The slab module contains a function :func:`slab.apply_to_path`, which applies a function to all wav files in a given folder and returns a dictionary of file names and computed features. In fact, you can also use that function to modify (for instance ramp and filter) all files in a folder.

For other time-frequency processing, the :meth:`.frames` provides an easy way to step through the signal in short windowed frames and compute some values from it. For instance, you could detect on- and offsets in the signal by computing the crest factor in each frame: ::

    signal.pulse() # apply a 4 Hz pulse to the 3 vowels from above
    signal.waveform() # note the pulses
    crest = [] # the short-term crest factor will show on- and offsets
    frames = signal.frames(duration=64)
    for f in frames:
        crest.append(f.crest_factor())
    times = signal.frametimes(duration=64) # frame center times
    import matplotlib.pyplot as plt
    plt.plot(times, crest) # peaks in the crest factor mark intensity ramps

Binaural sounds
---------------
For experiments in spatial hearing, or any other situation that requires differential manipulation of the left and right channel of a sound, you can use the :class:`Binaural` class. It inherits all methods from :class:`Sound` and provides additional methods for generating and manipulating binaural sounds, including advanced interaural time and intensity manipulation.

Generating binaural sounds
^^^^^^^^^^^^^^^^^^^^^^^^^^
Binaural sounds support all sound generating functions with a ``nchannels`` attribute of the :class:`Sound` class, but automatically set ``nchannels`` to 2. Noises support an additional ``kind`` argument, which can be set to 'diotic' (identical noise in both channels) or 'dichotic' (uncorrelated noise). Other methods just return 2-channel versions of the stimuli. You can recast any Sound object as Binaural sound, which duplicates the first channel if ``nchannels`` is 1 or greater than 2: ::

    monaural = slab.Sound.tone()
    monaural.nchannels
    out: 1
    binaural = slab.Binaural(monaural)
    binaural.nchannels
    out: 2
    binaural.left # access to the left channel
    binaural.right # access to the right channel

Loading a wav file with ``slab.Binaural('file.wav')`` returns a Binaural sound object with two channels (even if the wav file contains only one channel).

The easiest manipulation of a binaural parameter may be to change the interaural level difference (ILD). This can be achieved by setting the ``level`` attributes of both channels: ::

    noise = slab.Binaural.pinknoise()
    noise.left.level = 75
    noise.right.level = 85
    noise.level
    out: array([75., 85.])

The :meth:`.ild` makes this easier and keeps the overall level constant: ``noise.ild(10)`` adds a 10dB level difference (positive dB values attenuate the left channel (virtual sound source moves to the right). The pink noise in the example is a broadband signal, and the ILD is frequency dependent and should not be the same for all frequencies. A frequency-dependent level difference can be computed and applied with :meth:`.interaural_level_spectrum`. The level spectrum is computed from a head-related transfer function (HRTF) and can be customised for individual listeners. See :ref:`HRTF` for how to handle these functions. The default level spectrum is computed form the HRTF of the KEMAR binaural recording mannequin (as measured by `Gardener and Martin (1994) <https://sound.media.mit.edu/resources/KEMAR.html>`_ at the MIT Media Lab).

.. plot::
    :include-source:

    signal = slab.Binaural.pinknoise(kind='dichotic')
    signal.waveform()


::

    right_lateralized = sig.itd(duration=600e-6) # add an interaural time difference of 600 microsec, right channel leading
    # apply a linearly increasing or decreasing interaural time difference.
    # This is achieved by sinc interpolation of one channel with a dynamic delay:
    moving = sig.itd_ramp(from_itd=-0.001, to_itd=0.01)
    lateralized = sig.at_azimuth(azimuth=-45) # add frequency- and headsize-dependent ITD and ILD corresponding to a sound at 45 deg
	external = lateralized.externalize() # add a low resolution HRTF filter that results in the percept of an externalized source (i.e. outside of the head), defaults to the KEMAR HRTF recordings, but any HRTF can be supplied


    Signals
    -------
    Sounds inherit from the **Signal** class, which provides a generic signal object with properties duration, number of samples, sample times, number of channels. Keeps the data in a 'data' property and implements slicing, arithmetic operations, and conversion between sample points and time points.::

        sig = slab.Sound.pinknoise(nchannels=2) # make a pink noise
        sig.duration
    	out: 1.0
    	sig.nsamples
    	out: 8000
    	sig2 = sig.resample(samplerate=4000) # resample to 4 kHz
    	env = sig2.envelope() # returns a new signal containing the lowpass Hilbert envelopes of both channels
    	sig.delay(duration=0.0006, channel=0) # delay the first channel by 0.6 ms


.. rubric:: Footnotes

.. [#f1] Forward masking occurs when a signal cannot be heard due to a preceding masking sound. Typically, three intervals are presented to the listener, two contain only the masker and one contains the masker followed by the signal. The listener has to identify the interval with the signal. The level of the masker is fixed and the signal level is varied adaptively to obtain the masked threshold.
