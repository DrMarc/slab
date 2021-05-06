import copy
import numpy
from slab.sound import Sound
from slab.signal import Signal
from slab.filter import Filter
from slab.hrtf import HRTF
from slab import data_path

class Binaural(Sound):
    """
    Class for working with binaural sounds, including ITD and ILD manipulation. Binaural inherits all sound
    generation functions  from the Sound class, but returns binaural signals. Recasting an object of class sound or
    sound with 1 or 3+ channels calls Sound.copychannel to return a binaural sound with two channels identical
    to the first channel of the original sound.

    Arguments:
        data (slab.Signal | numpy.ndarray | list | str): see documentation of slab.Sound for details. the `data` must
            have either one or two channels. If it has one, that channel is duplicated
        samplerate (int): samplerate in Hz, must only be specified when creating an instance from an array.
    Attributes:
        .left: the first data channel, containing the sound for the left ear.
        .right: the second data channel, containing the sound for the right ear
        .data: the data-array of the Sound object which has the shape `n_samples` x `n_channels`.
        .n_channels: the number of channels in `data`. Must be 2 for a binaural sound.
        .n_samples: the number of samples in `data`. Equals `duration` * `samplerate`.
        .duration: the duration of the sound in seconds. Equals `n_samples` / `samplerate`.
    """
    # instance properties
    def _set_left(self, other):
        if hasattr(other, 'samplerate'):  # probably an slab object
            self.data[:, 0] = other.data[:, 0]
        else:
            self.data[:, 0] = numpy.array(other)

    def _set_right(self, other):
        if hasattr(other, 'samplerate'):  # probably an slab object
            self.data[:, 1] = other.data[:, 0]
        else:
            self.data[:, 1] = numpy.array(other)

    left = property(fget=lambda self: self.channel(0), fset=_set_left,
                    doc='The left channel for a stereo sound.')
    right = property(fget=lambda self: self.channel(1), fset=_set_right,
                     doc='The right channel for a stereo sound.')

    def __init__(self, data, samplerate=None):
        if isinstance(data, (Sound, Signal)):
            if data.n_channels == 1:  # if there is only one channel, duplicate it.
                self.data = numpy.tile(data.data, 2)
            elif data.n_channels == 2:
                self.data = data.data
            else:
                raise ValueError("Data must have one or two channel!")
            self.samplerate = data.samplerate
        elif isinstance(data, (list, tuple)):
            if isinstance(data[0], (Sound, Signal)):
                if data[0].n_samples != data[1].n_samples:
                    raise ValueError('Sounds must have same number of samples!')
                if data[0].samplerate != data[1].samplerate:
                    raise ValueError('Sounds must have same samplerate!')
                super().__init__([data[0].data[:, 0], data[1].data[:, 0]], data[0].samplerate)
            else:
                super().__init__(data, samplerate)
        elif isinstance(data, str):
            super().__init__(data, samplerate)
            if self.n_channels == 1:
                self.data = numpy.tile(self.data, 2)  # duplicate channel if monaural file
        else:
            super().__init__(data, samplerate)
            if self.n_channels == 1:
                self.data = numpy.tile(self.data, 2)  # duplicate channel if monaural file
        if self.n_channels != 2:
            ValueError('Binaural sounds must have two channels!')

    def itd(self, duration=None, max_lag=0.001):
        """
        Either estimate the interaural time difference of the sound or generate a new sound with the specified
        interaural time difference. The resolution for computing the ITD is 1/samplerate seconds. A negative
        ITD value means that the right channel is delayed, meaning the sound source is to the left.

        Arguments:
            duration (None| int | float): Given None, the instance's ITD is computed. Given another value, a new sound
                with the desired interaural time difference in samples (given an integer) or seconds (given a float)
                is generated.
            max_lag (float): Maximum possible value for ITD estimation. Defaults to 1 millisecond which is barely
                outside the physiologically plausible range for humans. Is ignored if `duration` is specified.
        Returns:
             (int | slab.Binaural): The interaural time difference in samples or a copy of the instance with the
                specified interaural time difference.
        Examples::

            sound = slab.Binaural.whitenoise()
            lateral = sound.itd(duration=0.0005)  # generate a sound with 0.5 ms ITD
            lateral.itd()  # estimate the ITD of the sound
        """
        if duration is None:
            return self._get_itd(max_lag)
        else:
            return self._apply_itd(duration)

    def _get_itd(self, max_lag):
        max_lag = Sound.in_samples(max_lag, self.samplerate)
        xcorr = numpy.correlate(self.data[:, 0], self.data[:, 1], 'full')
        lags = numpy.arange(-max_lag, max_lag + 1)
        xcorr = xcorr[self.n_samples - 1 - max_lag:self.n_samples + max_lag]
        idx = numpy.argmax(xcorr)
        return lags[idx]

    def _apply_itd(self, duration):
        if duration == 0:
            return self  # nothing needs to be shifted
        if duration < 0:  # negative itds by convention shift to the left (i.e. delay right channel)
            channel = 1  # right
        else:
            channel = 0  # left
        return self.delay(duration=abs(duration), channel=channel)

    def ild(self, dB=None):
        """
        Either estimate the interaural level difference of the sound or generate a new sound with the specified
        interaural level difference. Negative ILD value means that the left channel is louder than the right
        channel, meaning that the sound source is to the left.

        Arguments:
            dB (None | int | float): If None, estimate the sound's ITD. Given a value, a new sound is generated with
                the desired interaural level difference in decibels.
        Returns:
            (float | slab.Binaural): The sound's aural level difference or a new instance with the specified ILD.
        Examples::

            sig = Binaural.whitenoise()
            lateral_right = sig.ild(3) # attenuate left channel by 3 dB
            lateral_left = sig.ild(-3) # attenuate right channel by 3 dB
        """
        if dB is None:
            return self.right.level - self.left.level
        else:
            return self._apply_ild(dB)

    def _apply_ild(self, dB):
        new = copy.deepcopy(self)  # so that we can return a new sound
        level = numpy.mean(self.level)
        new_levels = (level - dB/2, level + dB/2)
        new.level = new_levels
        return new

    def itd_ramp(self, from_itd=-6e-4, to_itd=6e-4):
        """
        Generate a sound with a linearly increasing or decreasing interaural time difference. This is achieved by sinc
        interpolation of one channel with a dynamic delay. The resulting virtual sound source moves left or right.

        Arguments:
            from_itd (float): interaural time difference in seconds at the start of the sound.
                Negative numbers correspond to sources to the left of the listener.
            to_itd (float): interaural time difference in seconds at the end of the sound.
        Returns:
            (slab.Binaural): a copy of the instance wit the desired ITD ramp.
        Examples::

            sig = Binaural.whitenoise()
            moving = sig.itd_ramp(from_itd=-0.001, to_itd=0.01)
            moving.play()
        """
        new = copy.deepcopy(self)
        # make the ITD ramps
        left_ramp = numpy.linspace(from_itd / 2, to_itd / 2, self.n_samples)
        right_ramp = numpy.linspace(-from_itd / 2, -to_itd / 2, self.n_samples)
        if self.n_samples >= 8192:
            filter_length = 1024
        elif self.n_samples >= 512:
            filter_length = self.n_samples // 16 * 2  # 1/8th of n_samples, always even
        else:
            raise ValueError('Signal too short! (min 512 samples)')
        new = new.delay(duration=left_ramp, channel=0, filter_length=filter_length)
        new = new.delay(duration=right_ramp, channel=1, filter_length=filter_length)
        return new

    def ild_ramp(self, from_ild=-50, to_ild=50):
        """
        Generate a sound with a linearly increasing or decreasing interaural level difference. The resulting
        virtual sound source moves to the left or right.

        Arguments:
            from_ild (int | float): interaural level difference in decibels at the start of the sound.
                Negative numbers correspond to sources to the left of the listener.
            to_ild (int | float): interaural level difference in decibels at the end of the sound.
        Returns:
            (slab.Binaural): a copy of the instance with the desired ILD ramp. Any previously existing level difference
                is removed.
        Examples::

            sig = Binaural.whitenoise()
            moving = sig.ild_ramp(from_ild=-10, to_ild=10)
            move.play()
        """
        new = self.ild(0)  # set ild to zero
        # make ramps
        left_ramp = numpy.linspace(-from_ild / 2, -to_ild / 2, self.n_samples)
        right_ramp = numpy.linspace(from_ild / 2, to_ild / 2, self.n_samples)
        left_ramp = 10**(left_ramp/20.)
        right_ramp = 10**(right_ramp/20.)
        # multiply channels with ramps
        new.data[:, 0] *= left_ramp
        new.data[:, 1] *= right_ramp
        return new

    @staticmethod
    def azimuth_to_itd(azimuth, frequency=2000, head_radius=8.75):
        """
        Compute the ITD for a sound source at a given azimuth. For frequencies >= 2 kHz the Woodworth (1962)
        formula is used. For frequencies <= 500 Hz the low-frequency approximation mentioned in Aronson and Hartmann
        (2014) is used. For frequencies in between, we interpolate linearly between the two formulas.

        Arguments:
            azimuth (int | float): The azimuth angle of the sound source, negative numbers refer to sources to the left.
            frequency (int | float): Frequency in Hz for which the ITD is estimated.
                Use the default for for sounds with a broadband spectrum.
            head_radius (int | float): Radius of the head in centimeters. The bigger the head, the larger the ITD.
        Returns:
            (float): The interaural time difference for a sound source at a given azimuth.
        Examples::

            # compute the ITD for a sound source 90 degrees to the left for a large head
            itd = slab.Binaural.azimuth_to_itd(-90, head_radius=10)
        """
        head_radius = head_radius / 100
        azimuth_radians = numpy.radians(azimuth)
        speed_of_sound = 344  # m/s
        itd_2000 = (head_radius / speed_of_sound) * \
            (azimuth_radians + numpy.sin(azimuth_radians))  # Woodworth
        itd_500 = (3 * head_radius / speed_of_sound) * numpy.sin(azimuth_radians)
        itd = numpy.interp(frequency, [500, 2000], [itd_500, itd_2000],
                           left=itd_500, right=itd_2000)
        return itd

    @staticmethod
    def azimuth_to_ild(azimuth, frequency=2000, hrtf=None):
        """
        Get the interaural level difference corresponding to a sound source at a given azimuth.

        Arguments:
            azimuth (int | float): The azimuth angle of the sound source, negative numbers refer to sources to the left.
            frequency (int | float): Frequency in Hz for which the ITD is estimated.
                Use the default for for sounds with a broadband spectrum.
            hrtf (None | slab.HRTF): head-related transfer function from which the ILD is taken.
                If None use the MIT KEMAR mannequin.
        Returns:
            (float): The interaural level difference for a sound source at a given azimuth in decibels.
        Examples::

            ild = slab.Binaural.azimuth_to_ild(-90) # ILD equivalent to 90 deg leftward source using KEMAR HRTF.
        """
        ils = Binaural._make_level_spectrum_filter(hrtf=hrtf)
        freqs = ils[1:, 0]  # get vector of frequencies in ils filter bank
        azis = ils[0, 1:]  # get vector of azimuths in ils filter bank
        ils = ils[1:, 1:]  # the rest is the filter
        levels = [numpy.interp(azimuth, azis, ils[i, :]) for i in range(ils.shape[0])]  # interpolate levels at azimuth
        return numpy.interp(frequency, freqs, levels)*-1   # interpolate level difference at frequency

    def at_azimuth(self, azimuth=0):
        """
        Convenience function for adding ITD and ILD corresponding to the given `azimuth` to the sound source.
        Values are obtained from azimuth_to_itd and azimuth_to_ild.
        Frequency parameters for these functions are generated from the centroid frequency of the sound.
        """
        centroid = numpy.array(self.spectral_feature(feature='centroid')).mean()
        itd = Binaural.azimuth_to_itd(azimuth, frequency=centroid)
        ild = Binaural.azimuth_to_ild(azimuth, frequency=centroid)
        out = self.itd(duration=itd)
        return out.ild(dB=ild)

    def externalize(self, hrtf=None):
        """
        Convolve the sound with a smoothed HRTF to evoke the impression of an external sound source without adding
        directional information, see Kulkarni & Colburn (1998) for why that works.

        Arguments:
            hrtf (None | slab.HRTF): The HRTF to use. If None use the one from the MIT KEMAR mannequin. The sound
                source at zero azimuth and elevation is used for convolution so it has to be present in the HRTF.
        Returns:
            (slab.Binaural): externalized copy of the instance.
        """
        if not hrtf:
            hrtf = HRTF(data_path()+'mit_kemar_normal_pinna.sofa')  # load the hrtf file
        # get HRTF for [0,0] direction:
        idx_frontal = numpy.where((hrtf.sources[:, 1] == 0) & (hrtf.sources[:, 0] == 0))[0][0]
        if not idx_frontal.size: # idx_frontal is empty
            raise ValueError('No frontal direction [0,0] found in HRTF.')
        _, h = hrtf.data[idx_frontal].tf(channels=0, nbins=12, show=False)  # get low-res version of HRTF spectrum
        h[0] = 1  # avoids low-freq attenuation in KEMAR HRTF (unproblematic for other HRTFs)
        resampled_signal = copy.deepcopy(self)
        # if sound and HRTF has different samplerates, resample the sound, apply the HRTF, and resample back:
        resampled_signal = resampled_signal.resample(hrtf.data[0].samplerate)  # resample to hrtf rate
        filt = Filter(10**(h/20), fir=False, samplerate=hrtf.data[0].samplerate)
        filtered_signal = filt.apply(resampled_signal)
        filtered_signal = filtered_signal.resample(self.samplerate)
        return filtered_signal

    @staticmethod
    def _make_level_spectrum_filter(hrtf=None):
        """
        Compute the frequency band specific interaural intensity differences for all sound source azimuth's in
        a head-related transfer function. For every azimuth in the hrtf, the respective transfer function is applied
        to a sound. This sound is then divided into frequency sub-bands. The interaural level spectrum is the level
        difference between right and left for each of these sub-bands for each azimuth.

        Arguments:
            hrtf (None | slab.HRTF): The head-related transfer function used to compute the level spectrum. If None,
                use the recordings from the KEMAR mannequin. For the KEMAR, the level spectrum is saved to a file
                and loaded, the next time this function is executed to save computation time.
        Returns:
            (numpy.ndarray): A two dimensional array where the size of the first dimension is given by the number of
                sub-bands for which the level difference was computed plus one and the size of the second dimension is
                given by the number of sound source azimuth's in the hrft plus one. The first element of the first row
                is the sampling frequency and the other elements of the first row contain the azimuth of the respective
                column. In the remaining rows, the first element is the frequency of the sub-band and the other elements
                are the interaural level differences for each azimuth.
        Examples::

            ils = slab.Binaural.make_level_spectrum_filter()  # get the ils from the KEMAR recordings
            ils[0, 0] # the sampling rate
            ils[0, 1:] # the sound source azimuth's for which the level difference was calculated
            ils[1:, 0]  # the sub-band frequencies
            ils[5, :]  # the level difference for each azimuth in the 5th sub-band
        """
        if not hrtf:
            try:
                ils = numpy.load(data_path() + 'KEMAR_interaural_level_spectrum.npy')
                return ils
            except FileNotFoundError:
                hrtf = HRTF(data_path() +'mit_kemar_normal_pinna.sofa')  # load the hrtf file
                save_standard = True
        elif isinstance(hrtf, HRTF):
            save_standard = False
        else:
            raise ValueError("hrft must be either None or an instance of slab.HRTF!")
        # get the filters for the frontal horizontal arc
        idx = numpy.where((hrtf.sources[:, 1] == 0) & (
            (hrtf.sources[:, 0] <= 90) | (hrtf.sources[:, 0] >= 270)))[0]
        # at this point, we could just get the transfer function of each filter with hrtf.data[idx[i]].tf(),
        # but it may be better to get the spectral left/right differences with ERB-spaced frequency resolution:
        azi = hrtf.sources[idx, 0]
        # 270<azi<360 -> azi-360 to get negative angles on the left
        azi[azi >= 270] = azi[azi >= 270]-360
        sort = numpy.argsort(azi)
        fbank = Filter.cos_filterbank(samplerate=hrtf.samplerate, pass_bands=True)
        freqs = fbank.filter_bank_center_freqs()
        noise = Sound.pinknoise(samplerate=hrtf.samplerate)
        ils = numpy.zeros((len(freqs) + 1, len(idx) + 1))
        ils[0, 0] = hrtf.samplerate  # save samplerate in ils filter
        ils[1:, 0] = freqs  # first row are the frequencies
        for n, i in enumerate(idx[sort]):  # put the level differences in order of increasing angle
            noise_filt = Binaural(hrtf.data[i].apply(noise))
            noise_bank_left = fbank.apply(noise_filt.left)
            noise_bank_right = fbank.apply(noise_filt.right)
            ils[1:, n+1] = noise_bank_right.level - noise_bank_left.level
            ils[0, n+1] = azi[sort[n]]  # first entry is the angle
        if save_standard is True:
            numpy.save(data_path() + 'KEMAR_interaural_level_spectrum.npy', ils)
        return ils

    def interaural_level_spectrum(self, azimuth, level_spectrum_filter=None):
        """
        Apply a interaural level spectrum, corresponding to a sound sources azimuth, to a
        binaural sound. The interaural level spectrum consists of frequency specific interaural level differences
        which are computed from a head related transfer function (see the _make_level_spectrum_filter method).
        The binaural sound is divided into frequency sub-bands and the levels of each sub-band are set according to
        the respective level in the interaural level spectrum. Then, the sub-bands are summed up again into one
        binaural sound.

        Arguments:
            azimuth (int | float): azimuth for which the interaural level spectrum is calculated.
            level_spectrum_filter (None | numpy.ndarray): If None, the method _make_level_spectrum_filter is called
                which returns the interaural level spectrum of the KEMAR mannequin's head related transfer function.
                Any array given for this argument should be generated with the _make_level_spectrum_filter as well.
        Returns:
            (slab.Binaural): A binaural sound with the interaural level spectrum corresponding to the given azimuth.
        Examples::

            noise = Binaural.pinknoise(kind='diotic')
            noise.interaural_level_spectrum(azimuth=-45).play()
        """
        if not level_spectrum_filter:
            ils = Binaural._make_level_spectrum_filter()
        elif not isinstance(level_spectrum_filter, numpy.ndarray):
            raise ValueError("level_spectrum_filter must be either None or an array which can be generated using the "
                             "method make_spectrum_level_filter")
        else:
            ils = level_spectrum_filter
        ils_samplerate = int(ils[0, 0])
        original_samplerate = self.samplerate
        azis = ils[0, 1:]  # get vector of azimuths in ils filter bank
        ils = ils[1:, 1:]  # get the level differences
        # interpolate levels at azimuth
        levels = numpy.array([numpy.interp(azimuth, azis, ils[i, :]) for i in range(ils.shape[0])])
        # resample the signal to the rate of the HRTF from which the filter was computed:
        resampled = self.resample(samplerate=ils_samplerate)
        fbank = Filter.cos_filterbank(length=resampled.n_samples, samplerate=ils_samplerate, pass_bands=True)
        subbands_left = fbank.apply(resampled.left)
        subbands_right = fbank.apply(resampled.right)
        # change subband levels:
        subbands_left.level = subbands_left.level + levels / 2
        subbands_right.level = subbands_right.level - levels / 2
        out_left = Filter.collapse_subbands(subbands_left, filter_bank=fbank)
        out_right = Filter.collapse_subbands(subbands_right, filter_bank=fbank)
        out = Binaural([out_left, out_right])
        return out.resample(samplerate=original_samplerate)

    @staticmethod
    def whitenoise(duration=1.0, kind='diotic', samplerate=None):
        """
        Generate binaural white noise. `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.whitenoise`.
        """
        out = Binaural(Sound.whitenoise(duration=duration, n_channels=2, samplerate=samplerate))
        if kind == 'diotic':
            out.left = out.right
        return out

    @staticmethod
    def pinknoise(duration=1.0, kind='diotic', samplerate=None):
        """
        Generate binaural pink noise. `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.pinknoise`.
        """
        return Binaural.powerlawnoise(
                duration=duration, alpha=1, samplerate=samplerate)

    @staticmethod
    def powerlawnoise(duration=1.0, alpha=1, kind='diotic', samplerate=None):
        """
        Generate binaural power law noise. `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.powerlawnoise`.
        """
        if kind == 'dichotic':
            out = Binaural(Sound.powerlawnoise(
                duration=duration, alpha=alpha, samplerate=samplerate, n_channels=2))
            out.left = Sound.powerlawnoise(
                duration=duration, alpha=alpha, samplerate=samplerate, n_channels=1)
        elif kind == 'diotic':
            out = Binaural(Sound.powerlawnoise(
                duration=duration, alpha=alpha, samplerate=samplerate, n_channels=2))
            out.left = out.right
        else:
            raise ValueError("kind must be 'dichotic' or 'diotic'.")
        return out

    @staticmethod
    def irn(frequency=100, gain=1, n_iter=4, duration=1.0, kind='diotic', samplerate=None):
        """
        Generate iterated ripple noise (IRN). `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.irn`.
        """
        out = Binaural(Sound.irn(frequency=frequency, gain=gain, n_iter=n_iter, duration=duration,
            samplerate=samplerate, n_channels=2))
        if kind == 'diotic':
            out.left = out.right
        else:
            raise ValueError("kind must be 'dichotic' or 'diotic'.")
        return out

    @staticmethod
    def tone(frequency=500, duration=1., phase=0, samplerate=None):
        """ Identical to slab.Sound.tone, but with two channels. """
        return Binaural(Sound.tone(frequency=frequency, duration=duration, phase=phase, samplerate=samplerate,
                                   n_channels=2))

    @staticmethod
    def harmoniccomplex(f0=500, duration=1., amplitude=0, phase=0, samplerate=None):
        """ Identical to slab.Sound.harmoniccomplex, but with two channels. """
        return Binaural(Sound.harmoniccomplex(f0=f0, duration=duration, amplitude=amplitude, phase=phase,
                                              samplerate=samplerate, n_channels=2))

    @staticmethod
    def click(duration=0.0001, samplerate=None):
        """ Identical to slab.Sound.click, but with two channels. """
        return Binaural(Sound.click(duration=duration, samplerate=samplerate, n_channels=2))

    @staticmethod
    def clicktrain(**kwargs):
        """ Identical to slab.Sound.clicktrain, but with two channels. """
        return Binaural(Sound.clicktrain(**kwargs))

    @staticmethod
    def chirp(**kwargs):
        """ Identical to slab.Sound.chirp, but with two channels. """
        return Binaural(Sound.chirp(**kwargs))

    @staticmethod
    def silence(duration=1.0, samplerate=None):
        """ Identical to slab.Sound.silence, but with two channels. """
        return Binaural(Sound.silence(duration=duration, samplerate=samplerate, n_channels=2))

    @staticmethod
    def vowel(vowel='a', gender=None, glottal_pulse_time=12, formant_multiplier=1, duration=1., samplerate=None):
        """ Identical to slab.Sound.vowel, but with two channels. """
        return Binaural(Sound.vowel(vowel=vowel, gender=gender, glottal_pulse_time=glottal_pulse_time,
                                    formant_multiplier=formant_multiplier, duration=duration, samplerate=samplerate,
                                    n_channels=2))

    @staticmethod
    def multitone_masker(**kwargs):
        """ Identical to slab.Sound.multitone_masker, but with two channels. """
        return Binaural(Sound.multitone_masker(**kwargs))

    @staticmethod
    def equally_masking_noise(**kwargs):
        """ Identical to slab.Sound.erb_noise, but with two channels. """
        return Binaural(Sound.equally_masking_noise(**kwargs))

    def aweight(self):
        """ Identical to slab.Sound.aweight, but with two channels. """
        return Binaural(Sound.aweight(self))
