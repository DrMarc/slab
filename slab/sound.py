import time
import pathlib
import tempfile
import numpy
import copy

try:
    import soundfile
    have_soundfile = True
except ImportError:
    have_soundfile = False
try:
    import soundcard
    have_soundcard = True
except ImportError:
    have_soundcard = False
try:
    import scipy.signal
    have_scipy = True
except ImportError:
    have_scipy = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    have_pyplot = True
except ImportError:
    have_pyplot = False

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
    '''
    Class for working with sounds, including loading/saving, manipulating and playing.

    Examples:

    >>> import slab
    >>> import numpy
    >>> print(slab.Sound(numpy.ones([10,2]),samplerate=10))
    <class 'slab.sound.Sound'> duration 1.0, samples 10, channels 2, samplerate 10
    >>> print(slab.Sound(numpy.ones([10,2]),samplerate=10).channel(0))
    <class 'slab.sound.Sound'> duration 1.0, samples 10, channels 1, samplerate 10

    **Properties**

    >>> sig = slab.Sound.tone()
    >>> sig.level = 80
    >>> sig.level
    80.0

    **Generating sounds**

    All sound generating methods can be used with durations arguments in samples (int) or seconds (float).
    One can also set the number of channels by setting the keyword argument nchannels to the desired value.

    **Plotting**

    >>> vowel = slab.Sound.vowel(vowel='a', duration=.5, samplerate=8000)
    >>> vowel.ramp()
    >>> vowel.spectrogram(dyn_range = 50)
    >>> vowel.spectrum(low=100, high=4000, log_power=True)
    >>> vowel.waveform(start=0, end=.1)

    '''
    # instance properties

    def _get_level(self):
        '''
        Returns level in dB SPL (RMS) assuming array is in Pascals.
        In the case of multi-channel sounds, returns an array of levels
        for each channel, otherwise returns a float.
        '''
        if self.nchannels == 1:
            rms_value = numpy.sqrt(numpy.mean(numpy.square(self.data-numpy.mean(self.data))))
            if rms_value == 0:
                rms_dB = 0
            else:
                rms_dB = 20.0*numpy.log10(rms_value/2e-5)
            return rms_dB + _calibration_intensity
        chans = self.channels()
        levels = [c.level for c in chans]
        return numpy.array(levels)

    def _set_level(self, level):
        '''
        Sets level in dB SPL (RMS) assuming array is in Pascals. `level`
        should be a value in dB, or a tuple of levels, one for each channel.
        '''
        rms_dB = self._get_level()
        if self.nchannels > 1:
            level = numpy.array(level)
            if level.size == 1:
                level = level.repeat(self.nchannels)
            level = numpy.reshape(level, (1, self.nchannels))
            rms_dB = numpy.reshape(rms_dB, (1, self.nchannels))
        gain = 10**((level-rms_dB)/20.)
        self.data *= gain

    level = property(fget=_get_level, fset=_set_level, doc='''
    Can be used to get or set the rms level of a sound, which should be in dB.
    For single channel sounds a value in dB is used, for multiple channel
    sounds a value in dB can be used for setting the level (all channels
    will be set to the same level), or a list/tuple/array of levels. Use
    :meth:`slab.Sound.calibrate` to make the computed level reflect output intensity.
    ''')

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
        '''
        Load the file given by filename (.wav) and returns a Sound object.
        '''
        if not have_soundfile:
            raise ImportError(
                'Reading wav files requires SoundFile (pip install git+https://github.com/bastibe/SoundFile.git')
        data, samplerate = soundfile.read(filename)
        return Sound(data, samplerate=samplerate)

    @staticmethod
    def tone(frequency=500, duration=1., phase=0, samplerate=None, nchannels=1):
        '''
        Returns a pure tone at frequency for duration, using the default
        samplerate or the given one.

        Arguments:
            frequency/phase: if single values, multiple channels can be specified with the `nchannels` argument.
                If sequences, one frequency or phase is used for each channel.
        '''
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        frequency = numpy.array(frequency)
        phase = numpy.array(phase)
        if frequency.size > nchannels and nchannels == 1:
            nchannels = frequency.size
        if phase.size > nchannels and nchannels == 1:
            nchannels = phase.size
        if frequency.size == nchannels:
            frequency.shape = (1, nchannels)
        if phase.size == nchannels:
            phase.shape = (1, nchannels)
        t = numpy.arange(0, duration, 1)/samplerate
        t.shape = (t.size, 1)  # ensures C-order
        x = numpy.sin(phase + 2*numpy.pi * frequency * numpy.tile(t, (1, nchannels)))
        return Sound(x, samplerate)

    @staticmethod
    def harmoniccomplex(f0=500, duration=1., amplitude=0, phase=0, samplerate=None, nchannels=1):
        '''
        Returns a harmonic complex composed of pure tones at integer multiples of the fundamental frequency `f0`.

        Arguments:
            amplitude/phase: can be a single value or a sequence. In the former case the value is set for all harmonics,
                and harmonics up to 1/5th of the sampling frequency are generated. In the latter case each harmonic
                parameter is set separately, and the number of harmonics generated corresponds to the length of the
                sequence. Amplitudes are relateve to full scale (i.e. 0 corresponds to maximum intensity, -30 would be
                30 dB softer).
            phase: can have a special non-numerical value, the string 'schroeder', in which case the harmonics are in
                Schoeder phase, producing a complex tone with minimal peak-to-peak amplitudes (Schroeder 1970).

        >>> sig = Sound.harmoniccomplex(f0=200, amplitude=[0,-10,-20,-30])
        >>> _ = sig.spectrum()
        '''
        samplerate = Sound.get_samplerate(samplerate)
        phases = numpy.array(phase).flatten()
        amplitudes = numpy.array(amplitude).flatten()
        if len(phases) > 1 or len(amplitudes) > 1:
            if (len(phases) > 1 and len(amplitudes) > 1) and (len(phases) != len(amplitudes)):
                raise ValueError('Please specify the same number of phases and amplitudes')
            nharmonics = max(len(phases), len(amplitudes))
        else:
            nharmonics = int(numpy.floor(samplerate/(5*f0)))
        if len(phases) == 1:
            phases = numpy.tile(phase, nharmonics)
        if len(amplitudes) == 1:
            amplitudes = numpy.tile(amplitude, nharmonics)
        freqs = numpy.linspace(f0, nharmonics * f0, nharmonics, endpoint=True)
        if isinstance(phase, str) and phase == 'schroeder':
            n = numpy.linspace(1, nharmonics, nharmonics, endpoint=True)
            phases = numpy.pi * n * (n + 1) / nharmonics
        out = Sound.tone(f0, duration, phase=phases[0], samplerate=samplerate, nchannels=nchannels)
        lvl = out.level
        out.level += amplitudes[0]
        for i in range(1, nharmonics):
            tmp = Sound.tone(frequency=freqs[i], duration=duration,
                             phase=phases[i], samplerate=samplerate, nchannels=nchannels)
            tmp.level = lvl + amplitudes[i]
            out += tmp
        return out

    @staticmethod
    def whitenoise(duration=1.0, samplerate=None, nchannels=1, normalise=True):
        '''
        Returns a white noise. If the samplerate is not specified, the global
        default value will be used. nchannels = 2 produces uncorrelated noise (dichotic).
        See also :func:`Binaural.whitenoise`.

        >>> noise = Sound.whitenoise(1.0,nchannels=2)
        '''
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        x = numpy.random.randn(duration, nchannels)
        if normalise:
            for i in range(nchannels):
                x[:, i] = ((x[:, i] - numpy.amin(x[:, i])) /
                           (numpy.amax(x[:, i]) - numpy.amin(x[:, i])) - 0.5) * 2
        return Sound(x, samplerate)

    @staticmethod
    def powerlawnoise(duration=1.0, alpha=1, samplerate=None, nchannels=1, normalise=True):
        '''
        Returns a power-law noise for the given duration.
        Spectral density per unit of bandwidth scales as 1/(f**alpha).

        Arguments:
            duration: duration of the output.
            alpha: power law exponent.
            samplerate: output samplerate

        >>> noise = Sound.powerlawnoise(0.2, 1, samplerate=8000)
        '''
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        n = duration
        n2 = int(n/2)
        f = numpy.array(numpy.fft.fftfreq(n, d=1.0/samplerate), dtype=complex)
        f.shape = (len(f), 1)
        f = numpy.tile(f, (1, nchannels))
        if n % 2 == 1:
            z = (numpy.random.randn(n2, nchannels) + 1j * numpy.random.randn(n2, nchannels))
            a2 = 1.0 / (f[1:(n2+1), :]**(alpha/2.0))
        else:
            z = (numpy.random.randn(n2-1, nchannels) + 1j * numpy.random.randn(n2-1, nchannels))
            a2 = 1.0 / (f[1:n2, :]**(alpha/2.0))
        a2 *= z
        if n % 2 == 1:
            d = numpy.vstack((numpy.ones((1, nchannels)), a2,
                              numpy.flipud(numpy.conj(a2))))
        else:
            d = numpy.vstack((numpy.ones((1, nchannels)), a2,
                              1.0 / (numpy.abs(f[n2])**(alpha/2.0)) *
                              numpy.random.randn(1, nchannels),
                              numpy.flipud(numpy.conj(a2))))
        x = numpy.real(numpy.fft.ifft(d.flatten()))
        x.shape = (n, nchannels)
        if normalise:
            for i in range(nchannels):
                x[:, i] = ((x[:, i] - numpy.amin(x[:, i])) /
                           (numpy.amax(x[:, i]) - numpy.amin(x[:, i])) - 0.5) * 2
        return Sound(x, samplerate)

    @staticmethod
    def pinknoise(duration=1.0, samplerate=None, nchannels=1, normalise=True):
        '''
        Returns pink noise, i.e :func:`powerlawnoise` with alpha=1.
        nchannels = 2 produces uncorrelated noise (dichotic).
        See also :func:`Binaural.pinknoise`.
        '''
        return Sound.powerlawnoise(duration, 1.0, samplerate=samplerate,
                                   nchannels=nchannels, normalise=normalise)

    @staticmethod
    def irn(frequency=100, gain=1, niter=4, duration=1.0, samplerate=None):
        '''
        Iterated ripple noise (IRN) is a broadband noise with temporal regularities,
        which can give rise to a perceptible pitch. Since the perceptual pitch to noise
        ratio of these stimuli can be altered without substantially altering their spectral
        content, they have been useful in exploring the role of temporal processing in pitch
        perception [Yost 1996, JASA]. The noise is obtained by adding attenuated and delayed
        versions of a white noise in the frequency domain.

        Arguments:
            frequency: the frequency of the resulting pitch in Hz
            gain: multiplicative factor of the repeated additions. Smaller values reduce the
                temporal regularities in the resulting IRN.
            niter: number of iterations of additions. Higher values increase pitch saliency.
        '''
        samplerate = Sound.get_samplerate(samplerate)
        delay = 1/frequency
        noise = Sound.whitenoise(duration, samplerate=samplerate)
        x = numpy.array(noise.data.T)[0]
        irn_add = numpy.fft.fft(x)
        n_samples, sample_dur = len(irn_add), float(1/samplerate)
        w = 2 * numpy.pi*numpy.fft.fftfreq(n_samples, sample_dur)
        d = float(delay)
        for k in range(1, niter+1):
            irn_add += (gain**k) * irn_add * numpy.exp(-1j * w * k * d)
        irn_add = numpy.fft.ifft(irn_add)
        x = numpy.real(irn_add)
        return Sound(x, samplerate)

    @staticmethod
    def click(duration=0.0001, samplerate=None, nchannels=1):
        'Returns a click of the given duration (*100 microsec*).'
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        return Sound(numpy.ones((duration, nchannels)), samplerate)

    @staticmethod
    def clicktrain(duration=1.0, frequency=500, clickduration=0.0001, samplerate=None):
        'Returns a series of n clicks (see :func:`click`) at a frequency of freq.'
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        clickduration = Sound.in_samples(clickduration, samplerate)
        interval = int(numpy.rint(1/frequency * samplerate))
        n = numpy.rint(duration/interval)
        oneclick = Sound.click(clickduration, samplerate=samplerate)
        oneclick.resize(interval)
        oneclick.repeat(n)
        return oneclick

    @staticmethod
    def chirp(duration=1.0, from_frequency=100, to_frequency=None, samplerate=None, kind='quadratic'):
        '''Returns a pure tone with increasing or decreasing frequency from and to given
        frequency endpoints using :func:`scipy.signal.chirp`.
        `kind` determines the type of ramp (see :func:`scipy.signal.chirp` for options).
        '''
        if not have_scipy:
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
    def silence(duration=1.0, samplerate=None, nchannels=1):
        'Returns a silent sound (all samples equal zero) for the given duration.'
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        return Sound(numpy.zeros((duration, nchannels)), samplerate)

    @staticmethod
    def vowel(vowel='a', gender=None, glottal_pulse_time=12, formant_multiplier=1, duration=1., samplerate=None, nchannels=1):
        '''
        Returns a vowel sound.

        Arguments:
            vowel: 'a', 'e', 'i', 'o', 'u', 'ae', 'oe', or 'ue' (pre-set format frequencies)
                or None for random formants in the range of the vowel formants.
            gender: 'male', 'female'; shortcut for setting glottal_pulse_time and formant_multiplier
            glottal_pulse_time: distance in milliseconds of glottal pulses (determines vocal trakt length)
            formant_multiplier: multiplier for the predefined formant frequencies (scales the voice pitch)
        '''
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        formant_freqs = {'a': (0.73, 1.09, 2.44), 'e': (0.36, 2.25, 3.0), 'i': (0.27, 2.29, 3.01),
                         'o': (0.35, 0.5, 2.6), 'u': (0.3, 0.87, 2.24), 'ae': (0.86, 2.05, 2.85), 'oe': (0.4, 1.66, 1.96),
                         'ue': (0.25, 1.67, 2.05)}
        if vowel is None:
            BW = 0.3
            formants = (0.22/(1-BW)+(0.86/(1+BW)-0.22/(1-BW))*numpy.random.rand(),
                        0.5/(1-BW)+(2.29/(1+BW)-0.5/(1-BW))*numpy.random.rand(),
                        1.96/(1-BW)+(3.01/(1+BW)-1.96/(1-BW))*numpy.random.rand())
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
        ST = 1000/samplerate
        times = ST * numpy.arange(duration)
        T05 = 2.5  # decay half-time for glottal pulses
        env = numpy.exp(-numpy.log(2)/T05 * numpy.mod(times, glottal_pulse_time))
        env = numpy.mod(times, glottal_pulse_time)**0.25 * env
        min_env = numpy.min(env[(times >= glottal_pulse_time/2) & (times <= glottal_pulse_time-ST)])
        env = numpy.maximum(env, min_env)
        out = numpy.zeros(len(times))
        for f in formants:
            A = numpy.min((0, -6*numpy.log2(f)))
            out = out + 10**(A/20) * env * numpy.sin(2 * numpy.pi *
                                                     f * numpy.mod(times, glottal_pulse_time))
        if nchannels > 1:
            out = numpy.tile(out, (nchannels, 1))
        vowel = Sound(data=out, samplerate=samplerate)
        vowel.filter(frequency=0.75*samplerate/2, kind='lp')
        return vowel

    @staticmethod
    def multitone_masker(duration=1.0, low_cutoff=125, high_cutoff=4000, bandwidth=1/3, samplerate=None):
        '''
        Returns a noise made of ERB-spaced random-phase sinetones in the band between `low_cutoff` and `high_cutoff`.
        This noise does not have random amplitude variations and is useful for testing CI patients.
        See Oxenham 2014, Trends Hear.

        >>> sig = Sound.multitone_masker()
        >>> sig.ramp()
        >>> _ = sig.spectrum()
        '''
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        # get centre_freqs
        freqs, _, _ = Filter._center_freqs(
            low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth)
        rand_phases = numpy.random.rand(len(freqs)) * 2 * numpy.pi
        sig = Sound.tone(frequency=freqs, duration=duration,
                         phase=rand_phases, samplerate=samplerate)
        # collapse across channels
        data = numpy.sum(sig.data, axis=1) / len(freqs)
        return Sound(data, samplerate=samplerate)

    @staticmethod
    def erb_noise(duration=1.0, low_cutoff=125, high_cutoff=4000, samplerate=None):
        '''
        Returns an equally-masking noise (ERB noise) in the band between `low_cutoff` and `high_cutoff`.

        >>> sig = Sound.erb_noise()
        >>> sig.ramp()
        >>> _ = sig.spectrum()
        '''
        samplerate = Sound.get_samplerate(samplerate)
        duration = Sound.in_samples(duration, samplerate)
        n = 2**(duration-1).bit_length()  # next power of 2
        st = 1 / samplerate
        df = 1 / (st * n)
        frq = df * numpy.arange(n/2)
        frq[0] = 1  # avoid DC = 0
        lev = -10*numpy.log10(24.7*(4.37*frq))
        filt = 10.**(lev/20)
        noise = numpy.random.randn(n)
        noise = numpy.real(numpy.fft.ifft(numpy.concatenate(
            (filt, filt[::-1])) * numpy.fft.fft(noise)))
        noise = noise/numpy.sqrt(numpy.mean(noise**2))
        band = numpy.zeros(len(lev))
        band[round(low_cutoff/df):round(high_cutoff/df)] = 1
        fnoise = numpy.real(numpy.fft.ifft(numpy.concatenate(
            (band, band[::-1])) * numpy.fft.fft(noise)))
        fnoise = fnoise[:duration]
        return Sound(data=fnoise, samplerate=samplerate)

    @staticmethod
    def sequence(*sounds):
        'Joins the sounds in the list `sounds` into a new sound object.'
        samplerate = sounds[0].samplerate
        for sound in sounds:
            if sound.samplerate != samplerate:
                raise ValueError('All sounds must have the same sample rate.')
        sounds = tuple(s.data for s in sounds)
        x = numpy.vstack(sounds)
        return Sound(x, samplerate)

    # instance methods
    def write(self, filename, normalise=True, fmt='WAV'):
        '''
        Save the sound as a WAV. If `normalise` is set to True, the maximal amplitude of the sound is normalised to 1.
        '''
        if not have_soundfile:
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
        """
        Adds an on and/or off ramp to the sound.

        Args:
            when (str): can take values 'onset', 'offset' or 'both'
            duration (int, float): time over which the ramping happens (in samples or seconds)
        Returns:
            slab.Sound: copy of the instance with the added ramp(s)
        """
        sound = copy.deepcopy(self)
        when = when.lower().strip()
        if envelope is None:
            envelope = lambda t: numpy.sin(numpy.pi * t / 2) ** 2  # squared sine window
        sz = Sound.in_samples(duration, sound.samplerate)
        multiplier = envelope(numpy.reshape(numpy.linspace(0.0, 1.0, sz), (sz, 1)))
        if when in ('onset', 'both'):
            sound.data[:sz, :] *= multiplier
        if when in ('offset', 'both'):
            sound.data[sound.nsamples-sz:, :] *= multiplier[::-1]
        return sound

    def repeat(self, n):
        """
        Repeat the sound n times.
        Args:
            n (int): number of repetitions
        Returns:
            slab.Sound: copy of the instance with n repetitions
        """
        sound = copy.deepcopy(self)
        sound.data = numpy.vstack((sound.data,)*int(n))
        return sound

    @staticmethod
    def crossfade(sound1, sound2, overlap=0.01):
        '''
        Return a new sound that is a crossfade of sound1 and sound2 with a given `overlap`.

        >>> noise = Sound.whitenoise(duration=1.0)
        >>> vowel = Sound.vowel()
        >>> noise2vowel = Sound.crossfade(noise,vowel,overlap=0.4)
        >>> noise2vowel.play()
        '''
        if sound1.nchannels != sound2.nchannels:
            raise ValueError('Cannot crossfade sounds with unequal numbers of channels.')
        if sound1.samplerate != sound2.samplerate:
            raise ValueError('Cannot crossfade sounds with unequal samplerates.')
        overlap = Sound.in_samples(overlap, samplerate=sound1.samplerate)
        n_total = sound1.nsamples + sound2.nsamples - overlap
        silence = Sound.silence(sound1.nsamples - overlap,
                                samplerate=sound1.samplerate, nchannels=sound1.nchannels)
        sound1.ramp(duration=overlap, when='offset')
        sound1.resize(n_total)  # extend sound1 to total length
        sound2.ramp(duration=overlap, when='onset')
        sound2 = Sound.sequence(silence, sound2)  # sound2 has to be prepended with silence
        return sound1 + sound2

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
        pulse_period = 1/pulse_frequency
        n_pulses = round(sound.duration / pulse_period)  # number of pulses in the stimulus
        pulse_period = sound.duration / n_pulses  # period in s, fits into stimulus duration
        pulse_samples = Sound.in_samples(pulse_period * duty, sound.samplerate)
        fall_samples = Sound.in_samples(rf_time, sound.samplerate)  # 5ms rise/fall time
        fall = numpy.cos(numpy.pi * numpy.arange(fall_samples) / (2 * (fall_samples)))**2
        pulse = numpy.concatenate((1-fall, numpy.ones(pulse_samples - 2 * fall_samples), fall))
        pulse = numpy.concatenate(
            (pulse, numpy.zeros(Sound.in_samples(pulse_period, sound.samplerate)-len(pulse))))
        envelope = numpy.tile(pulse, n_pulses)
        envelope = envelope[:, None]  # add an empty axis to get to the same shape as sound.data
        # if data is 2D (>1 channel) broadcase the envelope to fit
        sound.data *= numpy.broadcast_to(envelope, sound.data.shape)
        return sound

    def am(self, frequency=10, depth=1, phase=0):
        """
        Apply an amplitude modulation to the sound by multplication with a sine funnction
        Args:
            frequency (int): frequency of the modulating sine function in Hz
            depth (int, float): amplitude of the modulating sine function
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
        n = min(1000, self.nsamples)
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
        numerators = [(2 * numpy.pi * f4)**2 * (10**(A1000 / 20)), 0, 0, 0, 0]
        denominators = numpy.convolve(
            [1, 4 * numpy.pi * f4, (2 * numpy.pi * f4)**2], [1, 4 * numpy.pi * f1, (2 * numpy.pi * f1)**2])
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
                subprocess.call(['sox', '-d', '-r', str(samplerate), str(_tmpdir / 'tmp.wav'), 'trim', '0', str(duration)])
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
        fname = str(fname) # in case it is a pathlib.Path object, get the name string
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
            end = self.nsamples
        end = self.in_samples(end, self.samplerate)
        if axis is None:
            _, axis = plt.subplots()
        if self.nchannels == 1:
            axis.plot(self.times[start:end], self.channel(0)[start:end], **kwargs)
        elif self.nchannels == 2:
            axis.plot(self.times[start:end], self.channel(0)[start:end], label='left', **kwargs)
            axis.plot(self.times[start:end], self.channel(1)[start:end], label='right', **kwargs)
            axis.legend()
        else:
            for i in range(self.nchannels):
                axis.plot(self.times[start:end], self.channel(i)[start:end], label=f'channel {i}', **kwargs)
            plt.legend()
        axis.set(title='Waveform', xlabel='Time [sec]', ylabel='Amplitude')
        if show:
            plt.show()

    def spectrogram(self, window_dur=0.005, dyn_range=120, upper_frequency=None, other=None, show=True, axis=None, **kwargs):
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
        if self.nchannels > 1:
            raise ValueError('Can only compute spectrograms for mono sounds.')
        if other is not None:
            x = self.data.flatten() - other.data.flatten()
        else:
            x = self.data.flatten()
        # set default for step_dur optimal for Gaussian windows.
        step_dur = window_dur/numpy.sqrt(numpy.pi)/8
        # convert window & step durations from seconds to numbers of samples
        window_nsamp = Sound.in_samples(window_dur, self.samplerate) * 2
        step_nsamp = Sound.in_samples(step_dur, self.samplerate)
        # make the window. A Gaussian filter needs a minimum of 6σ - 1 samples, so working
        # backward from window_nsamp we can calculate σ.
        window_sigma = (window_nsamp+1)/6
        window = scipy.signal.windows.gaussian(window_nsamp, window_sigma)
        # convert step size into number of overlapping samples in adjacent analysis frames
        noverlap = window_nsamp - step_nsamp
        # compute the power spectral density
        freqs, times, power = scipy.signal.spectrogram(
            x, mode='psd', fs=self.samplerate, scaling='density', noverlap=noverlap, window=window, nperseg=window_nsamp)
        if show or (axis is not None):
            if not have_pyplot:
                raise ImportError('Ploting spectrograms requires matplotlib.')
            p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
            power = 10 * numpy.log10(power / (p_ref ** 2))  # logarithmic power for plotting
            # set lower bound of colormap (vmin) from dynamic range.
            dB_max = power.max()
            vmin = dB_max-dyn_range
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

    def cochleagram(self, bandwidth=1/5, show=True, axis=None, **kwargs):
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
        envs = envs.data ** (1/3)  # apply non-linearity (cube-root compression)
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
        freqs = numpy.fft.rfftfreq(self.nsamples, d=1/self.samplerate)
        sig_rfft = numpy.zeros((len(freqs), self.nchannels))
        for chan in range(self.nchannels):
            sig_rfft[:, chan] = numpy.abs(numpy.fft.rfft(self.data[:, chan], axis=0))
        # scale by the number of points so that the magnitude does not depend on the length of the signal
        pxx = sig_rfft/len(freqs)
        pxx = pxx**2  # square to get the power
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
                                      (numpy.arange(12, dtype=float)*-1))
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

        >>> sig = Sound.tone(frequency=500, nchannels=2)
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
                frame_duration = int(self.nsamples/2)  # long frames if not averaging
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
                out = numpy.sqrt((delta_p**2).sum(axis=0)) / power.shape[0]
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
                out = numpy.sqrt(numpy.mean(out**2))  # average feature time series
            elif mean == 'average':
                out = out.mean()
            out_all.append(out)  # concatenate channel data
        if mean is None:
            out_all = Signal(data=out_all, samplerate=self.samplerate)  # cast as Signal
        return out_all

    def vocode(self, bandwidth=1/3):
        '''
        Returns a noise vocoded version of the sound by computing the envelope in different frequency subbands,
        filling these envelopes with noise, and collapsing the subbands into one sound. This removes most spectral
        information but retains temporal information in a speech signal.

        Arguments:
            bandwidth: width of the subbands in octaves
        '''
        fbank = Filter.cos_filterbank(length=self.nsamples, bandwidth=bandwidth,
                                      low_cutoff=30, pass_bands=True, samplerate=self.samplerate)
        subbands = fbank.apply(self.channel(0))
        envs = subbands.envelope()
        envs.data[envs.data < 1e-9] = 0  # remove small values that cause waring with numpy.power
        noise = Sound.whitenoise(duration=self.nsamples,
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
        if numpy.any(jwd): # if not all elements are zero
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
        step_nsamp = numpy.floor(window_nsamp/numpy.sqrt(numpy.pi)/8).astype(int)
        # make the window, Gaussian filter needs a minimum of 6σ - 1 samples.
        window_sigma = numpy.ceil((window_nsamp+1)/6)
        window = numpy.tile(scipy.signal.windows.gaussian(
            window_nsamp, window_sigma), (self.nchannels, 1)).T
        idx = 0
        while idx + window_nsamp/2 < self.nsamples: # loop through windows, yield each one
            frame.data = self.data[idx:min(self.nsamples, idx + window_nsamp), :]
            frame.resize(window_nsamp)  # in case the last window is too short
            frame *= window
            yield frame
            idx += step_nsamp

    def frametimes(self, duration=1024):
        'Returns the time points at the frame centers constructed by the `frames` method.'
        window_nsamp = Sound.in_samples(duration, self.samplerate) * 2
        step_nsamp = numpy.floor(window_nsamp/numpy.sqrt(numpy.pi)/8).astype(int)
        samplepoints = []
        idx = 0
        while idx + window_nsamp/2 < self.nsamples:
            samplepoints.append(min(idx + window_nsamp/2, self.nsamples))
            idx += step_nsamp
        return numpy.array(samplepoints) / self.samplerate # convert to array of time points


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
