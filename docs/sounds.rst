
Sound
=====
**Sound**: Inherits from Signal and provides methods for generating, manipulating, displaying, and analysing sound stimuli. Can compute descriptive sound features and apply manipulations to all sounds in a folder.::

    vowel = slab.Sound.vowel(vowel='a', duration=.5) # make a 0.5-second synthetic vowel sound
    vowel.ramp() # apply default raised-cosine onset and offset ramps
    vowel.filter(kind='bp', f=[50, 3000]) # apply bandpass filter between 50 and 3000 Hz
    vowel.spectrogram() # plot the spectrogram
    vowel.spectrum(low=100, high=4000, log_power=True) # plot a band-limited spectrum
    vowel.waveform(start=0, end=.1) # plot the waveform
	vowel.write('vowel.wav') # save the sound to a WAV file
	vocoded_vowel = vowel.vocode() # run a vocoding algorithm
	vowel.spectral_feature(feature='centroid') # compute the spectral centroid of the sound in Hz

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

Binaural sounds
---------------
**Binaural**: Inherits from Sound and provides methods for generating and manipulating binaural sounds, including advanced interaural time and intensity manipulation. Binaural sounds have left and a right channel properties.::

    sig = slab.Binaural.pinknoise()
	sig.pulse() # make a 2-channel pulsed pink noise
    sig.nchannels
    out: 2
    right_lateralized = sig.itd(duration=600e-6) # add an interaural time difference of 600 microsec, right channel leading
    # apply a linearly increasing or decreasing interaural time difference.
    # This is achieved by sinc interpolation of one channel with a dynamic delay:
    moving = sig.itd_ramp(from_itd=-0.001, to_itd=0.01)
    lateralized = sig.at_azimuth(azimuth=-45) # add frequency- and headsize-dependent ITD and ILD corresponding to a sound at 45 deg
	external = lateralized.externalize() # add a low resolution HRTF filter that results in the percept of an externalized source (i.e. outside of the head), defaults to the KEMAR HRTF recordings, but any HRTF can be supplied
