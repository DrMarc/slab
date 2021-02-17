import time
import pathlib
import tempfile
import numpy
import copy

try:
    import soundfile
except ImportError:
    soundfile = False
try:
    import soundcard
except ImportError:
    soundcard = False
try:
    import scipy.signal
except ImportError:
    scipy = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib, plt = False, False

from slab.signal import Signal
from slab.filter import Filter
from slab import DATAPATH

# get a temporary directory for writing intermediate files
_tmpdir = pathlib.Path(tempfile.gettempdir())

try:  # try getting a previously set calibration intensity from file
    _calibration_intensity = numpy.load(DATAPATH + 'calibration_intensity.npy')
except FileNotFoundError:
    _calibration_intensity = 0  #: Difference between rms intensity and measured output intensity in dB


class Sound(Signal):
    """ Class for working with sounds, including loading/saving, manipulating and playing. Inherits from the base class
    `slab.Signal`. Instances of Sound can be created by either loading a file, passing an array of values and a
    samplerate or by using one of the sound-generating methods of the class (all of the @staticmethods).
    Arguments:
        data(numpy.ndarray | str | pathlib.Path): given an array, an instance of `Sound` with the value of data as
            `data` and the value of samplerate as `samplerate`. A string or Path must point to a .wav. The data and
            samplerate is loaded from the file and used to create an instance of `Sound`.
        samplerate(int | float): must only be defined when creating a `Sound` from an array.
    Attributes:
        .data: the data-array of the Sound object which has the shape `n_samples` x `n_channels`.
        .n_channels: the number of channels in `data`.
        .n_samples: the number of samples in `data`. Equals `duration` * `samplerate`.
        .duration: the duration of the sound in seconds. Equals `n_samples` / `samplerate`.
    Examples:
        import slab, numpy
        # generate a Sound object from an array of random floats:
        sig = slab.Sound(data=numpy.random.randn(10000), samplerate=41000)
        # generate a Sound object using one of the modules methods, like `tone`:
        sig = slab.Sound.tone()  # generate a tone
        sig.level = 80  # set the level to 80 dB
        sig = sig.ramp(duration=0.05)  # add a 50 millisecond ramp
        sig.spectrum(log_power=True)  # plot the spectrum
        sig. waveform()  # plot the time course """

    def _get_level(self):
        """ Calculate level in dB SPL (RMS) assuming array is in Pascals.
        Returns:
            (float | numpy.ndarray): In the case of multi-channel sounds, returns an array of levels (one per channel),
                otherwise returns a float. """
        if self.n_channels == 1:
            rms_value = numpy.sqrt(numpy.mean(numpy.square(self.data - numpy.mean(self.data))))
            if rms_value == 0:
                rms_decibel = 0
            else:
                rms_decibel = 20.0 * numpy.log10(rms_value / 2e-5)
            return rms_decibel + _calibration_intensity
        channels = self.channels()
        levels = [c.level for c in channels]
        return numpy.array(levels)

    def _set_level(self, level):
        # TODO: display a warning if the setup is not level-calibrated
        """ Sets level in dB SPL (RMS) assuming array is in Pascals.
        Arguments:
            level (float | int | numpy.ndarray): the level in dB. Given a single float or int, all channels will be
                set to this level.
                should be a value in dB, or an array of levels, one for each channel. """
        rms_decibel = self._get_level()
        if self.n_channels > 1:
            level = numpy.array(level)
            if level.size == 1:
                level = level.repeat(self.n_channels)
            level = numpy.reshape(level, (1, self.n_channels))
            rms_decibel = numpy.reshape(rms_decibel, (1, self.n_channels))
        gain = 10 ** ((level - rms_decibel) / 20.)
        self.data *= gain

    level = property(fget=_get_level, fset=_set_level, doc="""
    Can be used to get or set the rms level of a sound, which should be in dB.
    For single channel sounds a value in dB is used, for multiple channel
    sounds a value in dB can be used for setting the level (all channels
    will be set to the same level), or a list/tuple/array of levels. Use
    :meth:`slab.Sound.calibrate` to make the computed level reflect output intensity.
    """)

    def __init__(self, data, samplerate=None):
        if isinstance(data, pathlib.Path):  # Sound initialization from a file name (pathlib object)
            data = str(data)
        if isinstance(data, str):  # Sound initialization from a file name (string)
            if samplerate is not None:
                raise ValueError('Cannot specify samplerate when initialising Sound from a file.')
            _ = Sound.read(data)
            self.data = _.data
            self.samplerate = _.samplerate
        else:
            # delegate to the baseclass init
            super().__init__(data, samplerate)

    # static methods (creating sounds)
    @staticmethod
    def read(filename):
        """ Load a wav file and create an instance of `Sound`.
        Arguments:
            filename (str): the full path to a (.wav) file.
        Returns:
            (slab.Sound): the sound generated with the `data` and `samplerate` from the file. """
        if soundfile is False:
            raise ImportError(
                'Reading wav files requires SoundFile (pip install git+https://github.com/bastibe/SoundFile.git')
        data, samplerate = soundfile.read(filename)
        return Sound(data, samplerate=samplerate)

    @staticmethod
    def tone(frequency=500, duration=1., phase=0, samplerate=None, n_channels=1):
        """ Generate a pure tone.
        Arguments:
            frequency (int | float | list): frequency of the tone.  Given a list of length `n_channels`, one
                element of the list is used as frequency for each channel.
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            phase (int | float | list): phase of the sinusoid, defaults to 0. Given a list of length `n_channels`, one
                element of the list is used as phase for each channel.
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            n_channels (int): number of channels, defaults to one.
        Returns:
            (slab.Sound): the tone generated from the parameters. """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        frequency = numpy.array(frequency)
        phase = numpy.array(phase)
        if frequency.size > n_channels == 1:
            n_channels = frequency.size
        if phase.size > n_channels == 1:
            n_channels = phase.size
        if frequency.size == n_channels:
            frequency.shape = (1, n_channels)
        if phase.size == n_channels:
            phase.shape = (1, n_channels)
        t = numpy.arange(0, duration, 1) / samplerate
        t.shape = (t.size, 1)  # ensures C-order
        x = numpy.sin(phase + 2 * numpy.pi * frequency * numpy.tile(t, (1, n_channels)))
        return Sound(x, samplerate)

    @staticmethod
    def harmoniccomplex(f0=500, duration=1., amplitude=0, phase=0, samplerate=None, n_channels=1):
        # TODO: in tone() the phase argument refers to the channels, here it refers to the harmonics --> rename?
        """ Generate a harmonic complex tone composed of pure tones at integer multiples of the fundamental frequency.
        Arguments:
            f0 (int): the fundamental frequency. Harmonics will be generated at integer multiples of this value.
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            amplitude (int | float | list): Amplitude in dB, relative to the full scale (i.e. 0 corresponds to maximum
                intensity, -30 would be 30 dB softer). Given a single int or float, all harmonics are set to the same
                amplitude and harmonics up to 1/5th of of the samplerate are generated. Given a list of values,
                the number of harmonics generated is equal to the length of the list with each element of the list
                setting the amplitude for one harmonic.
            phase (int | float | string | list): phase of the sinusoid, defaults to 0. Given a list (with the same
                length as the one given for the amplitude argument) every element will be used as the phase of one
                harmonic. Given a string, its value must be schroeder', in which case the harmonics are in
                Schroeder phase, producing a complex tone with minimal peak-to-peak amplitudes (Schroeder 1970).
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            n_channels (int): number of channels, defaults to one.
        Returns:
            (slab.Sound): the harmonic complex generated from the parameters.
        Examples:
            sig = slab.Sound.harmoniccomplex(f0=200, amplitude=[0,-10,-20,-30])  # generate the harmonic complex tone
            _ = sig.spectrum()  # plot it's spectrum """

        samplerate = Sound.get_samplerate(samplerate)
        phases = numpy.array(phase).flatten()
        amplitudes = numpy.array(amplitude).flatten()
        if len(phases) > 1 or len(amplitudes) > 1:
            if (len(phases) > 1 and len(amplitudes) > 1) and (len(phases) != len(amplitudes)):
                raise ValueError('Please specify the same number of phases and amplitudes')
            n_harmonics = max(len(phases), len(amplitudes))
        else:
            n_harmonics = int(numpy.floor(samplerate / (5 * f0)))
        if len(phases) == 1:
            phases = numpy.tile(phase, n_harmonics)
        if len(amplitudes) == 1:
            amplitudes = numpy.tile(amplitude, n_harmonics)
        freqs = numpy.linspace(f0, n_harmonics * f0, n_harmonics, endpoint=True)
        if isinstance(phase, str) and phase == 'schroeder':
            n = numpy.linspace(1, n_harmonics, n_harmonics, endpoint=True)
            phases = numpy.pi * n * (n + 1) / n_harmonics
        out = Sound.tone(f0, duration, phase=phases[0], samplerate=samplerate, nchannels=n_channels)
        lvl = out.level
        out.level += amplitudes[0]
        for i in range(1, n_harmonics):
            tmp = Sound.tone(frequency=freqs[i], duration=duration,
                             phase=phases[i], samplerate=samplerate, nchannels=n_channels)
            tmp.level = lvl + amplitudes[i]
            out += tmp
        return out

    @staticmethod
    def whitenoise(duration=1.0, samplerate=None, n_channels=1, normalise=True):
        """ Generate white noise.
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            n_channels (int): number of channels, defaults to one. If channels > 1, a new white noise signal will
                be generated and they will be uncorrelated.
            normalise (bool): if True, normalise the signal. # TODO: how is this kind of normalization called ?
        Returns:
            (slab.Sound): the white noise generated from the parameters.
        Examples:
            noise = slab.Sound.whitenoise(1.0, n_channels=2).  # generate a 1 second white noise with two channels """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        x = numpy.random.randn(duration, n_channels)
        if normalise:
            for i in range(n_channels):
                x[:, i] = ((x[:, i] - numpy.amin(x[:, i])) /
                           (numpy.amax(x[:, i]) - numpy.amin(x[:, i])) - 0.5) * 2
        return Sound(x, samplerate)

    @staticmethod
    def powerlawnoise(duration=1.0, alpha=1, samplerate=None, n_channels=1, normalise=True):
        """ Generate a power-law noise with a spectral density per unit of bandwidth scales as 1/(f**alpha).
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            alpha (int) : power law exponent.
            samplerate: output samplerate
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            n_channels (int): number of channels, defaults to one. If channels > 1, a new white noise signal will
                be generated and they will be uncorrelated.
            normalise (bool): if True, normalise the signal. # TODO: how is this kind of normalization called ?
        Returns:
            (slab.Sound): the power law noise generated from the parameters.
        Examples:
            # Generate and plot power law noise with three different exponents
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()
            for alpha in [1, 2, 3]:
                noise = slab.Sound.powerlawnoise(0.2, alpha, samplerate=8000)
                noise.spectrum(axis=ax, show=False)
            plt.show() """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        n = duration
        n2 = int(n / 2)
        f = numpy.array(numpy.fft.fftfreq(n, d=1.0 / samplerate), dtype=complex)
        f.shape = (len(f), 1)
        f = numpy.tile(f, (1, n_channels))
        if n % 2 == 1:
            z = (numpy.random.randn(n2, n_channels) + 1j * numpy.random.randn(n2, n_channels))
            a2 = 1.0 / (f[1:(n2 + 1), :] ** (alpha / 2.0))
        else:
            z = (numpy.random.randn(n2 - 1, n_channels) + 1j * numpy.random.randn(n2 - 1, n_channels))
            a2 = 1.0 / (f[1:n2, :] ** (alpha / 2.0))
        a2 *= z
        if n % 2 == 1:
            d = numpy.vstack((numpy.ones((1, n_channels)), a2,
                              numpy.flipud(numpy.conj(a2))))
        else:
            d = numpy.vstack((numpy.ones((1, n_channels)), a2,
                              1.0 / (numpy.abs(f[n2]) ** (alpha / 2.0)) *
                              numpy.random.randn(1, n_channels),
                              numpy.flipud(numpy.conj(a2))))
        x = numpy.real(numpy.fft.ifft(d.flatten()))
        x.shape = (n, n_channels)
        if normalise:
            for i in range(n_channels):
                x[:, i] = ((x[:, i] - numpy.amin(x[:, i])) /
                           (numpy.amax(x[:, i]) - numpy.amin(x[:, i])) - 0.5) * 2
        return Sound(x, samplerate)

    @staticmethod
    def pinknoise(duration=1.0, samplerate=None, n_channels=1, normalise=True):
        """ Generate pink noise (power law noise with exponent alpha==1. This is simply a wrapper for calling
        the `powerlawnoise` method.
        Arguments:
            see `slab.Sound.powerlawnoise`
        Returns:
            (slab.Sound): power law noise generated from the parameters with exponent alpha==1.
        """
        return Sound.powerlawnoise(duration, 1.0, samplerate=samplerate,
                                   nchannels=n_channels, normalise=normalise)

    @staticmethod
    def irn(frequency=100, gain=1, n_iter=4, duration=1.0, samplerate=None):
        """
        Generate iterated ripple noise (IRN). IRN is a broadband noise with temporal regularities,
        which can give rise to a perceptible pitch. Since the perceptual pitch to noise
        ratio of these stimuli can be altered without substantially altering their spectral
        content, they have been useful in exploring the role of temporal processing in pitch
        perception [Yost 1996, JASA]. The noise is obtained by adding attenuated and delayed
        versions of a white noise in the frequency domain.  # TODO: maybe this is a bit too much info for a docstring?
        Arguments:
            frequency (int | float): the frequency of the signals perceived pitch in Hz.
            gain (int | float) : multiplicative factor of the repeated additions. Smaller values reduce the
                temporal regularities in the resulting IRN.
            n_iter (int): number of iterations of additions. Higher values increase pitch saliency.
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
        Returns:
            (slab.Sound): ripple noise that has a perceived pitch at the given frequency.
        """
        samplerate = Sound.get_samplerate(samplerate)
        delay = 1 / frequency
        noise = Sound.whitenoise(duration, samplerate=samplerate)
        x = numpy.array(noise.data.T)[0]
        irn_add = numpy.fft.fft(x)
        n_samples, sample_dur = len(irn_add), float(1 / samplerate)
        w = 2 * numpy.pi * numpy.fft.fftfreq(n_samples, sample_dur)
        d = float(delay)
        for k in range(1, n_iter + 1):
            irn_add += (gain ** k) * irn_add * numpy.exp(-1j * w * k * d)
        irn_add = numpy.fft.ifft(irn_add)
        x = numpy.real(irn_add)
        return Sound(x, samplerate)

    @staticmethod
    def click(duration=0.0001, samplerate=None, n_channels=1):
        """Generate a click (a sequence of ones).
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            n_channels (int): number of channels, defaults to one.
        Returns:
            (slab.Sound): click generated from the given parameters.
            """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        return Sound(numpy.ones((duration, n_channels)), samplerate)

    @staticmethod
    def clicktrain(duration=1.0, frequency=500, clickduration=0.0001, samplerate=None):
        """Generate a series of n clicks (by calling the `click` method) with a perceived pitch at the given frequency.
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            frequency (float | int): the frequency of the signals perceived pitch in Hz.
            clickduration: (float | int): duration of a single click in seconds (given a float) or in
                samples (given an int). The number of clicks in the train is given by `duration` / `clickduration`.
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
        Returns:
            (slab.Sound): click train generated from the given parameters.
    """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        clickduration = Sound.in_samples(clickduration, samplerate)
        interval = int(numpy.rint(1 / frequency * samplerate))
        n = numpy.rint(duration / interval)
        oneclick = Sound.click(clickduration, samplerate=samplerate)
        oneclick.resize(interval)
        oneclick.repeat(n)
        return oneclick

    @staticmethod
    def chirp(duration=1.0, from_frequency=100, to_frequency=None, samplerate=None, kind='quadratic'):
        """ Returns a pure tone with in- or decreasing frequency using the function `scipy.signal.chirp`.
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            from_frequency (float | int): the frequency of tone in Hz at the start of the signal.
            to_frequency (float | int | None): the frequency of tone in Hz at the end of the signal. If None, the
                nyquist frequency (`samplerate` / 2) will be used.
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            kind (str): determines the type of ramp (see :func:`scipy.signal.chirp` for options).
        Returns:
            (slab.Sound): chirp generated from the given parameters.
        """

        if scipy is False:
            raise ImportError('Generating chirps requires Scipy.')
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        t = numpy.arange(0, duration, 1) / samplerate  # generate a time vector
        t.shape = (t.size, 1)  # ensures C-order
        if not to_frequency:
            to_frequency = samplerate / 2
        chirp = scipy.signal.chirp(
            t, from_frequency, t[-1], to_frequency, method=kind, vertex_zero=True)
        return Sound(chirp, samplerate=samplerate)

    @staticmethod
    def silence(duration=1.0, samplerate=None, n_channels=1):
        """ Generate silence (all samples equal zero).
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            n_channels (int): number of channels, defaults to one.
        Returns:
            (slab.Sound): silence generated from the given parameters. """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        return Sound(numpy.zeros((duration, n_channels)), samplerate)

    @staticmethod
    def vowel(vowel='a', gender=None, glottal_pulse_time=12, formant_multiplier=1,
              duration=1., samplerate=None, n_channels=1):
        """ Generate a sound resembling the human vocalization of a vowel.
        Arguments:
            vowel (str | None): kind of vowel to generate can be: 'a', 'e', 'i', 'o', 'u', 'ae', 'oe', or 'ue'. For
                these vowels, the function haa pre-set format frequencies. If None, a vowel will be generated from
                random formant frequencies in the range of the existing vowel formants.
            gender (str | None): Setting the gender ('male', 'female') is a shortcut for setting  the arguments
                `glottal_pulse_time` and `formant_multiplier`.
            glottal_pulse_time (int | float) : the distance between glottal pulses in
                milliseconds (determines vocal trakt length).
            formant_multiplier (int | float): multiplier for the pre-set formant frequencies (scales the voice pitch).
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            n_channels (int): number of channels, defaults to one.
        Returns:
            (slab.Sound): vowel generated from the given parameters. """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        formant_freqs = {'a': (0.73, 1.09, 2.44), 'e': (0.36, 2.25, 3.0), 'i': (0.27, 2.29, 3.01),
                         'o': (0.35, 0.5, 2.6), 'u': (0.3, 0.87, 2.24), 'ae': (0.86, 2.05, 2.85),
                         'oe': (0.4, 1.66, 1.96), 'ue': (0.25, 1.67, 2.05)}
        if vowel is None:
            BW = 0.3
            formants = (0.22 / (1 - BW) + (0.86 / (1 + BW) - 0.22 / (1 - BW)) * numpy.random.rand(),
                        0.5 / (1 - BW) + (2.29 / (1 + BW) - 0.5 / (1 - BW)) * numpy.random.rand(),
                        1.96 / (1 - BW) + (3.01 / (1 + BW) - 1.96 / (1 - BW)) * numpy.random.rand())
        else:
            if vowel not in formant_freqs:
                raise ValueError(f'Unknown vowel: {vowel}')
            formants = formant_freqs[vowel]
        if gender == 'male':
            glottal_pulse_time = 12
        elif gender == 'female':
            glottal_pulse_time = 6
            formant_multiplier = 1.2  # raise formant frequencies by 20%
        formants = [formant_multiplier * f for f in formants]  # scale each formant
        ST = 1000 / samplerate
        times = ST * numpy.arange(duration)
        T05 = 2.5  # decay half-time for glottal pulses
        env = numpy.exp(-numpy.log(2) / T05 * numpy.mod(times, glottal_pulse_time))
        env = numpy.mod(times, glottal_pulse_time) ** 0.25 * env
        min_env = numpy.min(env[(times >= glottal_pulse_time / 2) & (times <= glottal_pulse_time - ST)])
        env = numpy.maximum(env, min_env)
        out = numpy.zeros(len(times))
        for f in formants:
            A = numpy.min((0, -6 * numpy.log2(f)))
            out = out + 10 ** (A / 20) * env * numpy.sin(2 * numpy.pi *
                                                         f * numpy.mod(times, glottal_pulse_time))
        if n_channels > 1:
            out = numpy.tile(out, (n_channels, 1))
        vowel = Sound(data=out, samplerate=samplerate)
        vowel.filter(frequency=0.75 * samplerate / 2, kind='lp')
        return vowel

    @staticmethod
    def multitone_masker(duration=1.0, low_cutoff=125, high_cutoff=4000, bandwidth=1 / 3, samplerate=None):
        """ Generate noise made of ERB-spaced random-phase pure tones. This noise does not have random amplitude
        variations and is useful for testing CI patients [Oxenham 2014, Trends Hear].
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            low_cutoff (int | float): the lower frequency limit of the noise in Hz
            high_cutoff (int | float): the upper frequency limit of the noise in Hz
            bandwidth (float):  # TODO: what's a concise description of bandwidth?
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
        Returns:
            (slab.Sound): multi tone masker noise, generated from the given parameters.
        Examples:
            sig = Sound.multitone_masker()
            sig = sig.ramp()
            sig.spectrum()
            # Generate and plot power law noise with three different exponents
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()
            for alpha in [1, 2, 3]:
                noise = slab.Sound.powerlawnoise(0.2, alpha, samplerate=8000)
                noise.spectrum(axis=ax, show=False)
            plt.show() """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        freqs, _, _ = Filter._center_freqs(  # get center_freqs
            low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth)
        rand_phases = numpy.random.rand(len(freqs)) * 2 * numpy.pi
        sig = Sound.tone(frequency=freqs, duration=duration,
                         phase=rand_phases, samplerate=samplerate)
        data = numpy.sum(sig.data, axis=1) / len(freqs)  # collapse across channels
        return Sound(data, samplerate=samplerate)

    @staticmethod
    # TODO: would be more consistent if the function would be called equally_masking_noise?"
    def erb_noise(duration=1.0, low_cutoff=125, high_cutoff=4000, samplerate=None):
        """ Generate an equally-masking noise (ERB noise) within a given frequency band.
        Arguments:
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            low_cutoff (int | float): the lower frequency limit of the noise in Hz
            high_cutoff (int | float): the upper frequency limit of the noise in Hz
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
        Returns:
            (slab.Sound): equally masking noise noise, generated from the given parameters.
        Examples:
            sig = Sound.erb_noise()
            sig.spectrum()
        """
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        n = 2 ** (duration - 1).bit_length()  # next power of 2
        st = 1 / samplerate
        df = 1 / (st * n)
        frq = df * numpy.arange(n / 2)
        frq[0] = 1  # avoid DC = 0
        lev = -10 * numpy.log10(24.7 * (4.37 * frq))
        filt = 10. ** (lev / 20)
        noise = numpy.random.randn(n)
        noise = numpy.real(numpy.fft.ifft(numpy.concatenate(
            (filt, filt[::-1])) * numpy.fft.fft(noise)))
        noise = noise / numpy.sqrt(numpy.mean(noise ** 2))
        band = numpy.zeros(len(lev))
        band[round(low_cutoff / df):round(high_cutoff / df)] = 1
        fnoise = numpy.real(numpy.fft.ifft(numpy.concatenate(
            (band, band[::-1])) * numpy.fft.fft(noise)))
        fnoise = fnoise[:duration]
        return Sound(data=fnoise, samplerate=samplerate)

    @staticmethod
    def sequence(*sounds):
        """ Join sounds into a new sound object.
        Arguments
            *sounds (slab.Sound): two or more sounds to combine.
        Returns:
            (slab.Sound): the input sounds combined in a single object. """
        samplerate = sounds[0].samplerate
        for sound in sounds:
            if sound.samplerate != samplerate:
                raise ValueError('All sounds must have the same sample rate.')
        sounds = tuple(s.data for s in sounds)
        x = numpy.vstack(sounds)
        return Sound(x, samplerate)

    # instance methods
    def write(self, filename, normalise=True, fmt='WAV'):
        """ Save the sound as a WAV.
        Arguments:
            filename (str | pathlib.Path): path, the file is written to.
            normalise (bool): if True, the maximal amplitude of the sound is normalised to 1.
            fmt (str): data format to write. See soundfile.available_formats(). """
        if soundfile is False:
            raise ImportError(
                'Writing wav files requires SoundFile (pip install SoundFile).')
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        if self.samplerate % 1:
            self = self.resample(int(self.samplerate))
            print('Sampling rate rounded to nearest integer for writing!')
        if normalise:
            soundfile.write(filename, self.data / numpy.amax(numpy.abs(self.data)), self.samplerate, format=fmt)
        else:
            soundfile.write(filename, self.data, self.samplerate, format=fmt)

    def ramp(self, when='both', duration=0.01, envelope=None):
        """ Adds an on and/or off ramp to the sound.
        Arguments:
            when (str): can take values 'onset', 'offset' or 'both'
            duration (float | int): duration of the signal in seconds (given a float) or in samples (given an int).
            envelope():  # TODO: what does envelope do?
        Returns:
            (slab.Sound): copy of the sound with the added ramp(s) """
        sound = copy.deepcopy(self)
        when = when.lower().strip()
        if envelope is None:
            envelope = lambda t: numpy.sin(numpy.pi * t / 2) ** 2  # squared sine window
        sz = Sound.in_samples(duration, sound.samplerate)
        multiplier = envelope(numpy.reshape(numpy.linspace(0.0, 1.0, sz), (sz, 1)))
        if when in ('onset', 'both'):
            sound.data[:sz, :] *= multiplier
        if when in ('offset', 'both'):
            sound.data[sound.n_samples - sz:, :] *= multiplier[::-1]
        return sound

    def repeat(self, n):
        """ Repeat the sound n times.
        Arguments:
            n (int): the number of repetitions.
        Returns:
            (slab.Sound): copy of the sound repeated n times. """
        sound = copy.deepcopy(self)
        sound.data = numpy.vstack((sound.data,) * int(n))
        return sound

    @staticmethod
    def crossfade(*sounds, overlap=0.01):
        # TODO: write a new test for this
        """ Crossfade two sounds.
        Arguments:
            sound1 and sound2 (slab.Sound): sounds to crossfade
            overlap (float | int): duration of the overlap between the cross-faded sounds in seconds (given a float)
                or in samples (given an int).
        Returns:
            (slab.Sound):
        Examples:
            noise = Sound.whitenoise(duration=1.0)
            vowel = Sound.vowel()
            noise2vowel = Sound.crossfade(vowel, noise, vowel, overlap=0.4)
            noise2vowel.play() """
        sounds = list(sounds)
        if len(set([sound.n_channels for sound in sounds])) != 1:
            raise ValueError('Cannot crossfade sounds with unequal numbers of channels.')
        if len(set([sound.samplerate for sound in sounds])) != 1:
            raise ValueError('Cannot crossfade sounds with unequal samplerates.')
        overlap = Sound.in_samples(overlap, samplerate=sounds[0].samplerate)
        n_total = sum([sound.n_samples for sound in sounds]) - overlap * (len(sounds)-1)
        # give each sound an offset and onset ramp and add silence to them. The length of the silence added to the
        # beginning and end of the sound is equal to the length of the sounds that come before or after minus overlaps
        n_previous = 0
        for i, sound in enumerate(sounds):
            n_samples = sound.n_samples
            if i == 0:
                sound = sound.ramp(duration=overlap, when="offset")  # for the first sound only add offset ramp
                sounds[i] = sound.resize(n_total)
            else:
                if i == len(sounds)-1:
                    sound = sound.ramp(duration=overlap, when="onset")  # for the first sound only add onset ramp
                else:
                    sound = sound.ramp(duration=overlap, when="both")  # for all other sounds add both
                n_silence_before = n_previous - overlap * i
                n_silence_after = n_total - n_silence_before - sound.n_samples
                sounds[i] = Sound.sequence(  # TODO: BUg when one of the sounds is empty
                    Sound.silence(n_silence_before, samplerate=sound.samplerate, n_channels=sound.n_channels),
                    sound,
                    Sound.silence(n_silence_after, samplerate=sound.samplerate, n_channels=sound.n_channels))
            n_previous += n_samples
        sound = sum(sounds)
        return sound

    def pulse(self, pulse_frequency=4, duty=0.75, rf_time=0.05):
        """
        Apply a pulse envelope to the sound with a `pulse_frequency` and `duty` cycle (in place).
        Args:
            pulse_frequency (int): description
            duty (float, int): duty cycle in s
            rf(float): rise/fall time of the pulse in milliseconds
        Returns:
            slab.Sound: pulsed copy of the instance
        """
        sound = copy.deepcopy(self)
        pulse_period = 1 / pulse_frequency
        n_pulses = round(sound.duration / pulse_period)  # number of pulses in the stimulus
        pulse_period = sound.duration / n_pulses  # period in s, fits into stimulus duration
        pulse_samples = Sound.in_samples(pulse_period * duty, sound.samplerate)
        fall_samples = Sound.in_samples(rf_time, sound.samplerate)  # 5ms rise/fall time
        fall = numpy.cos(numpy.pi * numpy.arange(fall_samples) / (2 * (fall_samples))) ** 2
        pulse = numpy.concatenate((1 - fall, numpy.ones(pulse_samples - 2 * fall_samples), fall))
        pulse = numpy.concatenate(
            (pulse, numpy.zeros(Sound.in_samples(pulse_period, sound.samplerate) - len(pulse))))
        envelope = numpy.tile(pulse, n_pulses)
        envelope = envelope[:, None]  # add an empty axis to get to the same shape as sound.data
        # if data is 2D (>1 channel) broadcase the envelope to fit
        sound.data *= numpy.broadcast_to(envelope, sound.data.shape)
        return sound

    def am(self, frequency=10, depth=1, phase=0):
        """
        Apply an amplitude modulation to the sound by multplication with a sine function.
        Args:
            frequency (int): frequency of the modulating sine function in Hz
            depth (int, float): modulation depth/index of the modulating sine function
            phase (int, float): initial phase of the modulating sine function
        Returns:
            slab.Sound: amplitude modulated copy of the instance
        """
        sound = copy.deepcopy(self)
        envelope = (1 + depth * numpy.sin(2 * numpy.pi * frequency * sound.times + phase))
        envelope = envelope[:, None]
        sound.data *= numpy.broadcast_to(envelope, sound.data.shape)
        return sound

    def filter(self, frequency=100, kind='hp'):
        """
        Convenient wrapper for the Filter class for a standard low-, high-, bandpass,
        and bandstop filter.
        Args:
            frequency (int, tuple): cutoff frequency in Hz. Integer for low- and highpass filters,
                                    tuple eith lowe cutoff and upper cutoff for bandpass and -stop.
            kind (str): type of filter, can be "lp" (lowpass), "hp" (highpass)
                        "bp" (bandpass) or "bs" (bandstop)
        Returns:
            slab.Sound: filtered copy of the instance

        """
        sound = copy.deepcopy(self)
        n = min(1000, self.n_samples)
        filt = Filter.band(
            frequency=frequency, kind=kind, samplerate=self.samplerate, length=n)
        sound.data = filt.apply(self).data
        return sound

    def aweight(self):
        '''
        Returns A-weighted sound. A-weighting is applied to instrument-recorded sounds
        to account for the relative loudness of different frequencies perceived by the
        human ear. See: https://en.wikipedia.org/wiki/A-weighting'''
        if not have_scipy:
            raise ImportError('Applying a-weighting requires Scipy.')
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        A1000 = 1.9997
        numerators = [(2 * numpy.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
        denominators = numpy.convolve(
            [1, 4 * numpy.pi * f4, (2 * numpy.pi * f4) ** 2], [1, 4 * numpy.pi * f1, (2 * numpy.pi * f1) ** 2])
        denominators = numpy.convolve(numpy.convolve(
            denominators, [1, 2 * numpy.pi * f3]), [1, 2 * numpy.pi * f2])
        b, a = scipy.signal.filter_design.bilinear(numerators, denominators, self.samplerate)
        data_chans = []
        for chan in self.channels():
            data = scipy.signal.lfilter(b, a, chan.data.flatten())
            data_chans.append(data)  # concatenate channel data
        return Sound(data_chans, self.samplerate)

    @staticmethod
    def record(duration=1.0, samplerate=None):
        '''Record from inbuilt microphone. Note that most soundcards can only record at 44100 Hz samplerate.
        Uses SoundCard module if installed [recommended], otherwise uses SoX (duration must be in sec in this case).
        '''
        if have_soundcard:
            samplerate = Sound.get_samplerate(samplerate)
            duration = Sound.in_samples(duration, samplerate)
            mic = soundcard.default_microphone()
            data = mic.record(samplerate=samplerate, numframes=duration, channels=1)
            out = Sound(data, samplerate=samplerate)
        else:  # use sox
            import subprocess
            try:
                subprocess.call(
                    ['sox', '-d', '-r', str(samplerate), str(_tmpdir / 'tmp.wav'), 'trim', '0', str(duration)])
            except:
                raise ImportError(
                    'Recording whithout SoundCard module requires SoX. Install: sudo apt-get install sox libsox-fmt-all OR pip install SoundCard.')
            time.sleep(duration)
            out = Sound('tmp.wav')
        return out

    def play(self, sleep=False):
        'Plays the sound through the default device.'
        if have_soundcard:
            soundcard.default_speaker().play(self.data, samplerate=self.samplerate)
        else:
            self.write(_tmpdir / 'tmp.wav', normalise=False)
            Sound.play_file(_tmpdir / 'tmp.wav')
        if sleep:  # all current play methods are blocking, there is no reason to sleep!
            time.sleep(self.duration)

    @staticmethod
    def play_file(fname):
        fname = str(fname)  # in case it is a pathlib.Path object, get the name string
        from platform import system
        system = system()
        if system == 'Windows':
            import winsound
            winsound.PlaySound(fname, winsound.SND_FILENAME)
        elif system == 'Darwin':  # MacOS
            import subprocess
            subprocess.call(['afplay', fname])
        else:  # Linux
            import subprocess
            try:
                subprocess.call(['sox', fname, '-d'])
            except:
                raise NotImplementedError(
                    'Playing from files on Linux without SoundCard module requires SoX. Install: sudo apt-get install sox libsox-fmt-all or pip install SoundCard')

    def waveform(self, start=0, end=None, show=True, axis=None, **kwargs):
        '''
        Plots the waveform of the sound.

        Arguments:
            start, end: time or sample limits; if unspecified, shows the full waveform
        '''
        if not have_pyplot:
            raise ImportError('Plotting waveforms requires matplotlib.')
        start = self.in_samples(start, self.samplerate)
        if end is None:
            end = self.n_samples
        end = self.in_samples(end, self.samplerate)
        if axis is None:
            _, axis = plt.subplots()
        if self.n_channels == 1:
            axis.plot(self.times[start:end], self.channel(0)[start:end], **kwargs)
        elif self.n_channels == 2:
            axis.plot(self.times[start:end], self.channel(0)[start:end], label='left', **kwargs)
            axis.plot(self.times[start:end], self.channel(1)[start:end], label='right', **kwargs)
            axis.legend()
        else:
            for i in range(self.n_channels):
                axis.plot(self.times[start:end], self.channel(i)[start:end], label=f'channel {i}', **kwargs)
            plt.legend()
        axis.set(title='Waveform', xlabel='Time [sec]', ylabel='Amplitude')
        if show:
            plt.show()

    def spectrogram(self, window_dur=0.005, dyn_range=120, upper_frequency=None, other=None, show=True, axis=None,
                    **kwargs):
        '''
        Plots a spectrogram of the sound.

        Arguments:
            window_dur: Duration of time window for short-term FFT (*0.005sec*)
            dyn_range: Dynamic range in dB to plot (*120*)
            other: If a sound object is given, subtract the waveform and plot the difference spectrogram.
        If plot is False, returns the values returned by :func:`scipy.signal.spectrogram`, namely
        freqs, times, power where power is a 2D array of powers, freqs are the corresponding frequencies,
        and times are the time bins.
        '''
        if not have_scipy:
            raise ImportError('Computing spectrograms requires Scipy.')
        if self.n_channels > 1:
            raise ValueError('Can only compute spectrograms for mono sounds.')
        if other is not None:
            x = self.data.flatten() - other.data.flatten()
        else:
            x = self.data.flatten()
        # set default for step_dur optimal for Gaussian windows.
        step_dur = window_dur / numpy.sqrt(numpy.pi) / 8
        # convert window & step durations from seconds to numbers of samples
        window_nsamp = Sound.in_samples(window_dur, self.samplerate) * 2
        step_nsamp = Sound.in_samples(step_dur, self.samplerate)
        # make the window. A Gaussian filter needs a minimum of 6σ - 1 samples, so working
        # backward from window_nsamp we can calculate σ.
        window_sigma = (window_nsamp + 1) / 6
        window = scipy.signal.windows.gaussian(window_nsamp, window_sigma)
        # convert step size into number of overlapping samples in adjacent analysis frames
        noverlap = window_nsamp - step_nsamp
        # compute the power spectral density
        freqs, times, power = scipy.signal.spectrogram(
            x, mode='psd', fs=self.samplerate, scaling='density', noverlap=noverlap, window=window,
            nperseg=window_nsamp)
        if show or (axis is not None):
            if not have_pyplot:
                raise ImportError('Ploting spectrograms requires matplotlib.')
            p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
            power = 10 * numpy.log10(power / (p_ref ** 2))  # logarithmic power for plotting
            # set lower bound of colormap (vmin) from dynamic range.
            dB_max = power.max()
            vmin = dB_max - dyn_range
            cmap = matplotlib.cm.get_cmap('Greys')
            extent = (times.min(), times.max(), freqs.min(), upper_frequency or freqs.max())
            if axis is None:
                _, axis = plt.subplots()
            axis.imshow(power, origin='lower', aspect='auto',
                        cmap=cmap, extent=extent, vmin=vmin, vmax=None, **kwargs)
            axis.set(title='Spectrogram', xlabel='Time [sec]', ylabel='Frequency [Hz]')
        if show:
            plt.show()
        else:
            return freqs, times, power

    def cochleagram(self, bandwidth=1 / 5, show=True, axis=None, **kwargs):
        '''
        Computes a cochleagram of the sound by filtering with a bank of cosine-shaped filters with given bandwidth
        (*1/5* th octave) and applying a cube-root compression to the resulting envelopes.
        If show is False, returns the envelopes.
        '''
        fbank = Filter.cos_filterbank(bandwidth=bandwidth, low_cutoff=20,
                                      high_cutoff=None, samplerate=self.samplerate)
        freqs = fbank.filter_bank_center_freqs()
        subbands = fbank.apply(self.channel(0))
        envs = subbands.envelope()
        envs.data[envs.data < 1e-9] = 0  # remove small values that cause waring with numpy.power
        envs = envs.data ** (1 / 3)  # apply non-linearity (cube-root compression)
        if show or (axis is not None):
            if not have_pyplot:
                raise ImportError('Plotting cochleagrams requires matplotlib.')
            cmap = matplotlib.cm.get_cmap('Greys')
            if axis is None:
                _, axis = plt.subplots()
            axis.imshow(envs.T, origin='lower', aspect='auto', cmap=cmap)
            labels = list(freqs.astype(int))
            axis.yaxis.set_major_formatter(matplotlib.ticker.IndexFormatter(
                labels))  # centre frequencies as ticks
            axis.set_xlim([0, self.duration])
            axis.set(title='Cochleagram', xlabel='Time [sec]', ylabel='Frequency [Hz]')
            if show:
                plt.show()
        else:
            return envs

    def spectrum(self, low_cutoff=16, high_cutoff=None, log_power=True, axis=None, show=True, **kwargs):
        '''
        Returns the spectrum of the sound and optionally plots it.

        Arguments:
            low_cutoff/high_cutoff: If these are left unspecified, it shows the full spectrum, otherwise it shows
                only between `low` and `high` in Hz.
            log_power: If True it returns the log of the power.
            show: Whether to plot the output.
                If show=False, returns `Z, freqs`, where `Z` is a 1D array of powers
                and `freqs` are the corresponding frequencies.
        '''
        freqs = numpy.fft.rfftfreq(self.n_samples, d=1 / self.samplerate)
        sig_rfft = numpy.zeros((len(freqs), self.n_channels))
        for chan in range(self.n_channels):
            sig_rfft[:, chan] = numpy.abs(numpy.fft.rfft(self.data[:, chan], axis=0))
        # scale by the number of points so that the magnitude does not depend on the length of the signal
        pxx = sig_rfft / len(freqs)
        pxx = pxx ** 2  # square to get the power
        if low_cutoff is not None or high_cutoff is not None:
            if low_cutoff is None:
                low_cutoff = 0
            if high_cutoff is None:
                high_cutoff = numpy.amax(freqs)
            I = numpy.logical_and(low_cutoff <= freqs, freqs <= high_cutoff)
            I2 = numpy.where(I)[0]
            Z = pxx[I2, :]
            freqs = freqs[I2]
        else:
            Z = pxx
        if log_power:
            Z[Z < 1e-20] = 1e-20  # no zeros because we take logs
            Z = 10 * numpy.log10(Z)
        if show or (axis is not None):
            if not have_pyplot:
                raise ImportError('Plotting spectra requires matplotlib.')
            if axis is None:
                _, axis = plt.subplots()
            axis.semilogx(freqs, Z, **kwargs)
            ticks_freqs = numpy.round(32000 * 2 **
                                      (numpy.arange(12, dtype=float) * -1))
            axis.set_xticks(ticks_freqs)
            axis.set_xticklabels(map(str, ticks_freqs.astype(int)))
            axis.grid()
            axis.set_xlim((freqs[1], freqs[-1]))
            axis.set_ylabel('Power [dB/Hz]') if log_power else plt.ylabel('Power')
            axis.set_title('Spectrum')
            if show:
                plt.show()
        else:
            return Z, freqs

    def spectral_feature(self, feature='centroid', mean='rms', frame_duration=None, rolloff=0.85):
        '''
        Computes one of several features of the spectrogram of a sound and returns either a
        new Signal with the feature value at each sample, or the average (*rms* or mean) feature value over all samples.
        Available features:
        `centroid` is the centre of mass of the short-term spectrum, and 'fwhm' is the width of a Gaussian of the same variance as the spectrum around the centroid.

        >>> sig = Sound.tone(frequency=500, n_channels=2)
        >>> round(sig.spectral_feature(feature='centroid')[0])
        500.0

        `flux` is a measure of how quickly the power spectrum of a signal is changing, calculated by comparing the power spectrum for one frame against the power spectrum from the previous frame. Returns the root-mean-square over the entire stimulus of the change in power spectrum between adjacent time windows, measured as Euclidean distance.

        >>> sig = Sound.tone()
        >>> numpy.testing.assert_allclose(sig.spectral_feature(feature='flux'), desired=0, atol=1e-04)

        `flatness` measures how tone-like a sound is, as opposed to being noise-like.
        It is calculated by dividing the geometric mean of the power spectrum by the arithmetic mean. (Dubnov, Shlomo  "Generalization of spectral flatness measure for non-gaussian linear processes" IEEE Signal Processing Letters, 2004, Vol. 11.)

        `rolloff` is the frequency at which the spectrum rolles off and is typically used to find a suitable low-cutoff
        frequency that retains most of the signal power (given as fraction in `rolloff`).
        '''
        if not frame_duration:
            if mean is not None:
                frame_duration = int(self.n_samples / 2)  # long frames if not averaging
            else:
                frame_duration = 0.05  # 50ms frames by default
        out_all = []
        for chan in self.channels():
            freqs, times, power = chan.spectrogram(window_dur=frame_duration, show=False)
            norm = power / power.sum(axis=0, keepdims=True)  # normalize successive frames
            if feature == 'centroid':
                out = numpy.sum(freqs[:, numpy.newaxis] * norm, axis=0)
            elif feature == 'fwhm':
                cog = numpy.sum(freqs[:, numpy.newaxis] * norm, axis=0)
                sq_dist_from_cog = (freqs[:, numpy.newaxis] - cog[numpy.newaxis, :]) ** 2
                sigma = numpy.sqrt(numpy.sum(sq_dist_from_cog * norm, axis=0))
                out = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma
            elif feature == 'flux':
                norm = numpy.c_[norm[:, 0], norm]  # duplicate first frame to give 0 diff
                delta_p = numpy.diff(norm, axis=1)  # diff now has same shape as norm
                out = numpy.sqrt((delta_p ** 2).sum(axis=0)) / power.shape[0]
            elif feature == 'rolloff':
                cum = numpy.cumsum(norm, axis=0)
                rolloff_idx = numpy.argmax(cum >= rolloff, axis=0)
                out = freqs[rolloff_idx]  # convert from index to Hz
            elif feature == 'flatness':
                norm[norm == 0] = 1
                gmean = numpy.exp(numpy.log(power + 1e-20).mean(axis=0))
                amean = power.sum(axis=0) / power.shape[0]
                out = gmean / amean
            else:
                raise ValueError('Unknown feature name.')
            if mean is None:
                out = numpy.interp(self.times, times, out)  # interpolate to sound samples
            elif mean == 'rms':
                out = numpy.sqrt(numpy.mean(out ** 2))  # average feature time series
            elif mean == 'average':
                out = out.mean()
            out_all.append(out)  # concatenate channel data
        if mean is None:
            out_all = Signal(data=out_all, samplerate=self.samplerate)  # cast as Signal
        return out_all

    def vocode(self, bandwidth=1 / 3):
        '''
        Returns a noise vocoded version of the sound by computing the envelope in different frequency subbands,
        filling these envelopes with noise, and collapsing the subbands into one sound. This removes most spectral
        information but retains temporal information in a speech signal.

        Arguments:
            bandwidth: width of the subbands in octaves
        '''
        fbank = Filter.cos_filterbank(length=self.n_samples, bandwidth=bandwidth,
                                      low_cutoff=30, pass_bands=True, samplerate=self.samplerate)
        subbands = fbank.apply(self.channel(0))
        envs = subbands.envelope()
        envs.data[envs.data < 1e-9] = 0  # remove small values that cause waring with numpy.power
        noise = Sound.whitenoise(duration=self.n_samples,
                                 samplerate=self.samplerate)  # make white noise
        subbands_noise = fbank.apply(noise)  # divide into same subbands as signal
        subbands_noise *= envs  # apply envelopes
        subbands_noise.level = subbands.level
        return Sound(Filter.collapse_subbands(subbands=subbands_noise, filter_bank=fbank))

    def crest_factor(self):
        '''
        The crest factor is the ratio of the peak amplitude and the RMS value of a waveform
        and indicates how extreme the peaks in a waveform are. Returns the crest factor in dB.
        Numerically identical to the peak-to-average power ratio.
        '''
        jwd = self.data - numpy.mean(self.data)
        if numpy.any(jwd):  # if not all elements are zero
            crest = numpy.abs(jwd).max() / numpy.sqrt(numpy.mean(numpy.square(jwd)))
            return 20 * numpy.log10(crest)
        return numpy.nan

    def onset_slope(self):
        '''
        Returns the centroid of a histogram of onset slopes as a measure of how many
        quick intensity increases the sound has. These onset-like features make the
        sound easier to localize via envelope ITD.
        '''
        env = self.envelope(kind='dB')  # get envelope
        diffs = numpy.diff(env.data, axis=0) * self.samplerate  # compute db change per sec
        diffs[diffs < 0] = 0  # keep positive changes (onsets)
        if diffs.max() == 0:
            return 0
        # compute histogram of differences
        hist, bins = numpy.histogram(diffs, range=(1, diffs.max()), bins=1000)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        norm = hist / hist.sum()  # normalize histogram so that it summs to 1
        return numpy.sum(bin_centers * norm)  # compute centroid of histogram

    def frames(self, duration=1024):
        '''
        Returns a generator that steps through the sound in overlapping, windowed frames.
        Get the frame center times by calling `frametimes`. The frames have the same class as the object.

        Arguments:
            duration: half-length of the returned frames in samples or seconds

        >>> windows = sig.frames()
        >>> for w in windows:
        >>>		process(w) # process windowed frame here
        '''
        frame = copy.deepcopy(self)
        if not have_scipy:
            raise ImportError('Need scipy for time window processing.')
        window_nsamp = Sound.in_samples(duration, self.samplerate) * 2
        # step_dur optimal for Gaussian windows
        step_nsamp = numpy.floor(window_nsamp / numpy.sqrt(numpy.pi) / 8).astype(int)
        # make the window, Gaussian filter needs a minimum of 6σ - 1 samples.
        window_sigma = numpy.ceil((window_nsamp + 1) / 6)
        window = numpy.tile(scipy.signal.windows.gaussian(
            window_nsamp, window_sigma), (self.n_channels, 1)).T
        idx = 0
        while idx + window_nsamp / 2 < self.n_samples:  # loop through windows, yield each one
            frame.data = self.data[idx:min(self.n_samples, idx + window_nsamp), :]
            frame.resize(window_nsamp)  # in case the last window is too short
            frame *= window
            yield frame
            idx += step_nsamp

    def frametimes(self, duration=1024):
        'Returns the time points at the frame centers constructed by the `frames` method.'
        window_nsamp = Sound.in_samples(duration, self.samplerate) * 2
        step_nsamp = numpy.floor(window_nsamp / numpy.sqrt(numpy.pi) / 8).astype(int)
        samplepoints = []
        idx = 0
        while idx + window_nsamp / 2 < self.n_samples:
            samplepoints.append(min(idx + window_nsamp / 2, self.n_samples))
            idx += step_nsamp
        return numpy.array(samplepoints) / self.samplerate  # convert to array of time points


def calibrate(intensity=None, make_permanent=False):
    '''
    Calibrate the presentation intensity of a setup. Enter the calibration intensity, if you know it.
    If None, plays a 1kHz tone. Please measure actual intensity with a sound level meter and appropriate
    coupler. Set make_permanent to True to save a calibration file in slab.DATAPATH that is loaded on import.
    '''
    global _calibration_intensity
    if intensity is None:
        tone = Sound.tone(duration=5.0, frequency=1000)  # make 1kHz tone
        print('Playing 1kHz test tone for 5 seconds. Please measure intensity.')
        tone.play()  # play it
        intensity = input('Enter measured intensity in dB: ')  # ask for measured intesnity
        intensity = intensity - tone.level  # subtract measured from rms intensity
    # set and save
    _calibration_intensity = intensity
    if make_permanent:
        numpy.save(DATAPATH + 'calibration_intensity.npy', _calibration_intensity)


def apply_to_path(path='.', method=None, kwargs={}, out_path=None):
    '''
    Apply a function to all wav files in a given directory.

    Arguments:
        path: input path (str or pathlib.Path) from which wav files are collected for processing
        method: callable function to be applied to each file
        kwargs: dictionary of keyword arguments and values passed to the function.
        out_path: if is supplied, sounds are saved with their original file name in this directory

    >>> slab.apply_to_path('.', slab.Sound.spectral_feature, {'feature':'fwhm'})
    >>> slab.apply_to_path('.', slab.Sound.ramp, out_path='./modified')
    >>> slab.apply_to_path('.', slab.Sound.ramp, kwargs={'duration':0.3}, out_path='./test')
    '''
    if not callable(method):
        raise ValueError('Method must be callable.')
    if isinstance(path, str):
        path = pathlib.Path(path)
    if isinstance(out_path, str):
        out_path = pathlib.Path(out_path)
    files = sorted(path.glob('*.wav'))
    results = dict()
    for file in files:
        sig = Sound(file)
        res = method(sig, **kwargs)
        if out_path:
            if hasattr(res, 'write'):  # if objects with write methods were returned, write them to out_path
                res.write(out_path.joinpath(file.name))
            else:  # otherwise assume the modification was in-place and write sig to out_path
                sig.write(out_path.joinpath(file.name))
        results[str(file.stem)] = res
    return results  # a dictionary of results for each file name
