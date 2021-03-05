import copy
import numpy
from slab.sound import Sound
from slab.signal import Signal
from slab.filter import Filter
from slab.hrtf import HRTF


class Binaural(Sound):
    """
    Class for working with binaural sounds, including ITD and ILD manipulation. Binaural inherits all signal
    generation functions  from the Sound class, but returns binaural signals. Recasting an object of class sound or
    signal with 1 or 3+ channels calls Sound.copychannel to return a binaural sound with two channels identical
    to the first channel of the original signal.
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
        .duration: the duration of the sound in seconds. Equals `n_samples` / `samplerate`. """
    # instance properties
    def _set_left(self, other):
        if isinstance(other, Sound):
            self.data[:, 0] = other.data[:, 0]
        else:
            self.data[:, 0] = numpy.array(other)

    def _set_right(self, other):
        if isinstance(other, Sound):
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
                super().__init__((data[0].data[:, 0], data[1].data[:, 0]), data[0].samplerate)
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
        """ Either estimate the interaural time difference of the sound or generate a new sound with the specified
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
        Examples:
            sound = slab.Binaural.whitenoise()
            lateral = sound.itd(duration=0.0005)  # generate a sound with 0.5 ms ITD
            lateral.itd()  # estimate the ITD of the sound """
        if duration is None:
            return self._get_itd(max_lag)
        else:
            return self._apply_itd(duration)

    def _get_itd(self, max_lag):
        max_lag = Sound.in_samples(max_lag, self.samplerate)
        xcorr = numpy.correlate(self.data[:,0], self.data[:,1], 'full')
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
        """ Either estimate the interaural level difference of the sound or generate a new sound with the specified
            interaural level difference. Negative ILD value means that the left channel is louder than the right channel,
            meaning that the sound source is to the left.
        Arguments:
            dB (None | int | float): If None, estimate the sound's ITD. Given a value, a new sound is generated with
                the desired interaural level difference in decibels.
        Returns:
            (float | slab.Binaural): The sound's aural level difference or a new instance with the specified ILD.
        Examples:
            sig = Binaural.whitenoise()
            lateral_right = sig.ild(3) # attenuate left channel by 3 dB
            lateral_left = sig.ild(-3) # attenuate right channel by 3 dB """
        if dB is None:
            return self.right.level - self.left.level
        else:
            return self._apply_ild(dB)

    def _apply_ild(self, dB):
        new = copy.deepcopy(self)  # so that we can return a new signal
        level = numpy.mean(self.level)
        new_levels = (level + dB/2, level - dB/2)
        new.level = new_levels
        return new

    def itd_ramp(self, from_itd=-6e-4, to_itd=6e-4):
        """ Generate a sound with a linearly increasing or decreasing interaural time difference.
        This is achieved by sinc interpolation of one channel with a dynamic delay. The resulting virtual sound source
        moves to the left or right.
        Arguments:
            from_itd (float): interaural time difference in seconds at the start of the sound.
                Negative numbers correspond to sources to the left of the listener.
            to_itd (float): interaural time difference in seconds at the end of the sound.
        Returns:
            (slab.Binaural): a copy of the instance wit the desired ITD ramp.
        Examples:
            sig = Binaural.whitenoise()
            moving = sig.itd_ramp(from_itd=-0.001, to_itd=0.01)
            moving.play() """
        new = copy.deepcopy(self)
        # make the ITD ramps
        left_ramp = numpy.linspace(from_itd / 2, to_itd / 2, self.n_samples)
        right_ramp = numpy.linspace(-from_itd / 2, -to_itd / 2, self.n_samples)
        if self.n_samples >= 8192:
            filter_length = 1024
        elif self.n_samples >= 512:
            filter_length = self.n_samples // 16 * 2  # 1/8th of n_samples, always even
        else:
            ValueError('Signal too short! (min 512 samples)')
        new = new.delay(duration=left_ramp, channel=0, filter_length=filter_length)
        new = new.delay(duration=right_ramp, channel=1, filter_length=filter_length)
        return new

    def ild_ramp(self, from_ild=-50, to_ild=50):
        """ Generate a sound with a linearly increasing or decreasing interaural level difference. The resulting
        virtual sound source moves to the left or right.
        Arguments:
            from_ild (int | float): interaural level difference in decibels at the start of the sound.
                Negative numbers correspond to sources to the left of the listener.
            to_ild (int | float): interaural level difference in decibels at the end of the sound.
        Returns:
            (slab.Binaural): a copy of the instance with the desired ILD ramp. Any previously existing level difference
                is removed.
        Examples:
            sig = Binaural.whitenoise()
            moving = sig.ild_ramp(from_ild=-10, to_ild=10)
            move.play() """
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
        """ Compute the ITD for a sound source at a given azimuth. For frequencies >= 2 kHz the Woodworth (1962)
        formula is used. For frequencies <= 500 Hz the low-frequency approximation mentioned in Aronson and Hartmann
        (2014) is used. For frequencies in between, we interpolate linearly between the two formulas.
        Arguments:
            azimuth (int | float): The azimuth angle of the sound source, negative numbers refer to sources to the left.
            frequency (int | float): Frequency in Hz for which the ITD is estimated.
                Use the default for for sounds with a broadband spectrum.
            head_radius (int | float): Radius of the head in centimeters. The bigger the head, the larger the ITD.
        Returns:
            (float): The interaural time difference for a sound source at a given azimuth.
        Examples:
            # compute the ITD for a sound source 90 degrees to the left for a large head
            itd = slab.Binaural.azimuth_to_itd(-90, head_radius=10) """
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
        """ Get the interaural level difference corresponding to a sound source at a given azimuth.
        Arguments:
            azimuth (int | float): The azimuth angle of the sound source, negative numbers refer to sources to the left.
            frequency (int | float): Frequency in Hz for which the ITD is estimated.
                Use the default for for sounds with a broadband spectrum.
            hrtf (None | slab.HRTF): head-related transfer function from which the ILD is taken.
                If None use the MIT KEMAR mannequin.
        Returns:
            (float): The interaural level difference for a sound source at a given azimuth in decibels.
        Examples:
            ild = slab.Binaural.azimuth_to_ild(-90) # ILD equivalent to 90 deg leftward source using KEMAR HRTF.d """
        ils = Binaural._make_level_spectrum_filter(hrtf=hrtf)
        freqs = ils[1:, 0]  # get vector of frequencies in ils filter bank
        azis = ils[0, 1:]  # get vector of azimuths in ils filter bank
        ils = ils[1:, 1:]  # the rest is the filter
        levels = [numpy.interp(azimuth, azis, ils[i, :]) for i in range(ils.shape[0])]  # interpolate levels at azimuth
        return numpy.interp(frequency, freqs, levels)   # interpolate level difference at frequency

    def at_azimuth(self, azimuth=0):
        """ Convenience function for adding ITD and ILD corresponding to the given `azimuth` to the sound source.
            Values are obtained from azimuth_to_itd and azimuth_to_ild.
            Frequency parameters for these functions are generated from the centroid frequency of the sound. """
        centroid = numpy.array(self.spectral_feature(feature='centroid')).mean()
        itd = Binaural.azimuth_to_itd(azimuth, frequency=centroid)
        ild = Binaural.azimuth_to_ild(azimuth, frequency=centroid)
        out = self.apply_itd(duration=itd)
        return out.apply_ild(dB=ild)

    def externalize(self, hrtf=None):
        """ Convolve the sound with a smoothed HRTF to evoke the impression of an external sound source without adding
         directional information, see Kulkarni & Colburn (1998) for why that works.
        Arguments:
            hrtf (None | slab.HRTF): The HRTF to use. If None use the one from the MIT KEMAR mannequin. The sound
                source at zero azimuth and elevation is used for convolution so it has to be present in the HRTF.
        Returns:
            (slab.Binaural): externalized copy of the instance. """  # TODO: is this working properly?
        if not hrtf:
            from slab import DATAPATH
            hrtf = HRTF(DATAPATH+'mit_kemar_normal_pinna.sofa')  # load the hrtf file
        # get HRTF for [0,0] direction:
        idx_frontal = numpy.where((hrtf.sources[:, 1] == 0) & (hrtf.sources[:, 0] == 0))[0][0]
        if not idx_frontal.size: # idx_frontal is empty
            raise ValueError('No frontal direction [0,0] found in HRTF.')
        _, h = hrtf.data[idx_frontal].tf(channels=0, nbins=12, show=False)  # get low-res version of HRTF spectrum
        resampled_signal = copy.deepcopy(self)
        # if signal and HRTF has different samplerates, resample the signal, apply the HRTF, and resample back:
        resampled_signal = resampled_signal.resample(hrtf.data[0].samplerate)  # resample to hrtf rate
        filt = Filter(10**(h/20), fir=False, samplerate=hrtf.data[0].samplerate)
        filtered_signal = filt.apply(resampled_signal)
        filtered_signal = filtered_signal.resample(self.samplerate)
        return filtered_signal

    @staticmethod
    def _make_level_spectrum_filter(hrtf=None):
        """ Generate a level spectrum from the horizontal recordings in a HRTF. The defaut
        :meth:`slab.Filter.cos_filterbank` is used and the same filter bank has to be used when applying
        the level spectrum to a sound. """
        from slab import DATAPATH
        save_standard = False
        if not hrtf:
            try:
                ils = numpy.load(DATAPATH + 'KEMAR_interaural_level_spectrum.npy')
                return ils
            except FileNotFoundError:
                hrtf = HRTF(DATAPATH+'mit_kemar_normal_pinna.sofa')  # load the hrtf file
                save_standard = True
        # get the filters for the frontal horizontal arc
        idx = numpy.where((hrtf.sources[:, 1] == 0) & (
            (hrtf.sources[:, 0] <= 90) | (hrtf.sources[:, 0] >= 270)))[0]
        # at this point, we could just get the transfer function of each filter with hrtf.data[idx[i]].tf(),
        # but it may be better to get the spectral left/right differences with ERB-spaced frequency resolution:
        azi = hrtf.sources[idx, 0]
        # 270<azi<360 -> azi-360 to get negative angles the left
        azi[azi >= 270] = azi[azi >= 270]-360
        sort = numpy.argsort(azi)
        fbank = Filter.cos_filterbank(samplerate=hrtf.samplerate)
        freqs = fbank.filter_bank_center_freqs()
        noise = Sound.pinknoise(samplerate=hrtf.samplerate)
        ils = numpy.zeros((len(freqs) + 1, len(idx) + 1))
        ils[:, 0] = numpy.concatenate(([0], freqs))  # first row are the frequencies
        for n, i in enumerate(idx[sort]):  # put the level differences in order of increasing angle
            noise_filt = Binaural(hrtf.data[i].apply(noise))
            noise_bank_left = fbank.apply(noise_filt.left)
            noise_bank_right = fbank.apply(noise_filt.right)
            ils[1:, n+1] = noise_bank_right.level - noise_bank_left.level
            ils[0, n+1] = azi[sort[n]]  # first entry is the angle
        if save_standard is True:
            numpy.save(DATAPATH + 'KEMAR_interaural_level_spectrum.npy', ils)
        return ils

    def interaural_level_spectrum(self, azimuth, level_spectrum_filter=None):
        '''
        Apply a frequency-dependend interaural level difference corresponding to a given `azimuth` to a binaural sound.
        The level difference cues are taken from a filter generated with the :meth:`._make_level_spectrum_filter`
        function from a :class:`slab.HRTF` object. The default will generate the filter from the MIT KEMAR recordings.
        The left and right channel of the sound should have the same level.

        >>> noise = Binaural.pinknoise(kind='diotic')
        >>> noise.interaural_level_spectrum(azimuth=-45).play()
        '''
        if not level_spectrum_filter:
            ils = Binaural._make_level_spectrum_filter()
        ils = ils[:, 1:]  # remove the frequency values (not necessary here)
        azis = ils[0, :]  # get vector of azimuths in ils filter bank
        ils = ils[1:, :]  # the rest is the filter
        # interpolate levels at azimuth
        levels = [numpy.interp(azimuth, azis, ils[:, i]) for i in range(ils.shape[1])]
        fbank = Filter.cos_filterbank(length=self.n_samples, samplerate=self.samplerate)
        subbands_left = fbank.apply(self.left)
        subbands_right = fbank.apply(self.right)
        # change subband levels:
        subbands_left.level = subbands_left.level + levels / 2
        subbands_right.level = subbands_right.level - levels / 2
        out_left = Filter.collapse_subbands(subbands_left, filter_bank=fbank)
        out_right = Filter.collapse_subbands(subbands_right, filter_bank=fbank)
        return Binaural([out_left, out_right])

    @staticmethod
    def whitenoise(duration=1.0, kind='diotic', samplerate=None, normalise=True):
        '''
        Returns a white noise. `kind` = 'diotic' produces the same noise samples in both channels, `kind` = 'dichotic'
        produces uncorrelated noise.

        >>> noise = Binaural.whitenoise(kind='diotic')
        '''
        out = Binaural(Sound.whitenoise(duration=duration, n_channels=2,
                                        samplerate=samplerate, normalise=normalise))
        if kind == 'diotic':
            out.left = out.right
        return out

    @staticmethod
    def pinknoise(duration=1.0, kind='diotic', samplerate=None, normalise=True):
        '''
        Returns a pink noise. `kind` = 'diotic' produces the same noise samples in both channels, `kind` = 'dichotic'
        produces uncorrelated noise.

        >>> noise = Binaural.pinknoise(kind='diotic')
        '''
        return Binaural.powerlawnoise(
                duration=duration, alpha=1.0, samplerate=samplerate, normalise=normalise)

    @staticmethod
    def powerlawnoise(duration=1.0, alpha=1, kind='diotic', samplerate=None, normalise=True):
        if kind == 'dichotic':
            out = Binaural(Sound.powerlawnoise(
                duration=duration, alpha=alpha, samplerate=samplerate, nchannels=2, normalise=normalise))
            out.left = Sound.powerlawnoise(
                duration=duration, alpha=alpha, samplerate=samplerate, nchannels=1, normalise=normalise)
        elif kind == 'diotic':
            out = Binaural(Sound.powerlawnoise(
                duration=duration, alpha=alpha, samplerate=samplerate, nchannels=2, normalise=normalise))
            out.left = out.right
        return out

    @staticmethod
    def tone(frequency=500, duration=1., phase=0, samplerate=None):
        return Binaural(Sound.tone(frequency=frequency, duration=duration, phase=phase, samplerate=samplerate, nchannels=2))

    @staticmethod
    def harmoniccomplex(f0=500, duration=1., amplitude=0, phase=0, samplerate=None):
        return Binaural(Sound.harmoniccomplex(f0=f0, duration=duration, amplitude=amplitude, phase=phase, samplerate=samplerate, nchannels=2))

    @staticmethod
    def irn(**kwargs):
        return Binaural(Sound.irn(**kwargs))

    @staticmethod
    def click(duration=0.0001, samplerate=None):
        return Binaural(Sound.click(duration=duration, samplerate=samplerate, nchannels=2))

    @staticmethod
    def clicktrain(**kwargs):
        return Binaural(Sound.clicktrain(**kwargs))

    @staticmethod
    def chirp(**kwargs):
        return Binaural(Sound.chirp(**kwargs))

    @staticmethod
    def silence(duration=1.0, samplerate=None):
        return Binaural(Sound.silence(duration=duration, samplerate=samplerate, nchannels=2))

    @staticmethod
    def vowel(vowel='a', gender=None, glottal_pulse_time=12, formant_multiplier=1, duration=1., samplerate=None):
        return Binaural(Sound.vowel(vowel=vowel, gender=gender, glottal_pulse_time=glottal_pulse_time, formant_multiplier=formant_multiplier, duration=duration, samplerate=samplerate, nchannels=2))

    @staticmethod
    def multitone_masker(**kwargs):
        return Binaural(Sound.multitone_masker(**kwargs))

    @staticmethod
    def erb_noise(**kwargs):
        return Binaural(Sound.erb_noise(**kwargs))

    def aweight(self):
        return Binaural(Sound.aweight(self))


if __name__ == '__main__':
    sig = Binaural.pinknoise(duration=0.5, samplerate=44100)
    sig.filter(kind='bp', f=[100, 6000])
    sig.ramp(when='both', duration=0.15)
    sig_itd = sig.itd_ramp(500e-6, -500e-6)
    sig_itd.play()
