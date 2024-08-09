import copy
import numpy
import math
import sys
from slab.sound import Sound
from slab.signal import Signal
from slab.filter import Filter
from slab.hrtf import HRTF

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
        name (str): A string label for the Sound object. The inbuilt sound generating functions will automatically
            set .name to the name of the method used. Useful for logging during experiments.

    Attributes:
        .left: the first data channel, containing the sound for the left ear.
        .right: the second data channel, containing the sound for the right ear
        .data: the data-array of the Sound object which has the shape `n_samples` x `n_channels`.
        .n_channels: the number of channels in `data`. Must be 2 for a binaural sound.
        .n_samples: the number of samples in `data`. Equals `duration` * `samplerate`.
        .duration: the duration of the sound in seconds. Equals `n_samples` / `samplerate`.
        .name: string label of the sound.
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

    left = property(fget=lambda self: Sound(self.channel(0)), fset=_set_left,
                    doc='The left channel for a stereo sound.')
    right = property(fget=lambda self: Sound(self.channel(1)), fset=_set_right,
                     doc='The right channel for a stereo sound.')

    def __init__(self, data, samplerate=None, name='unnamed'):
        if isinstance(data, (Sound, Signal)):
            if hasattr(data, 'name'):
                self.name = data.name
            else:
                self.name = name
            if data.n_channels == 1:  # if there is only one channel, duplicate it.
                self.data = numpy.tile(data.data, 2)
            elif data.n_channels == 2:
                self.data = data.data
            else:
                raise ValueError("Data must have one or two channel!")
            self.samplerate = data.samplerate
        elif isinstance(data, (list, tuple)): # list of Sounds
            if isinstance(data[0], (Sound, Signal)):
                if data[0].n_samples != data[1].n_samples:
                    raise ValueError('Sounds must have same number of samples!')
                if data[0].samplerate != data[1].samplerate:
                    raise ValueError('Sounds must have same samplerate!')
                super().__init__([data[0].data[:, 0], data[1].data[:, 0]], data[0].samplerate, name=data[0].name)
            else: # list of samples
                super().__init__(data, samplerate, name=name)
        elif isinstance(data, str): # file name
            super().__init__(data, samplerate)
            if self.n_channels == 1:
                self.data = numpy.tile(self.data, 2)  # duplicate channel if monaural file
        else: # anything but Sound, list, or file name
            super().__init__(data, samplerate, name=name)
            if self.n_channels == 1:
                self.data = numpy.tile(self.data, 2)  # duplicate channel if monaural file
        if self.n_channels != 2: # bail if unable to enforce 2 channels
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
        out = copy.deepcopy(self)
        out.name = f'{str(duration)}-itd_{self.name}'
        return out._apply_itd(duration)

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
        channel, meaning that the sound source is to the left. The level difference is achieved by adding half the ILD
        to one channel and subtracting half from the other channel, so that the mean intensity remains constant.

        Arguments:
            dB (None | int | float): If None, estimate the sound's ITD. Given a value, a new sound is generated with
                the desired interaural level difference in decibels.
        Returns:
            (float | slab.Binaural): The sound's interaural level difference, or a new sound with the specified ILD.
        Examples::

            sig = slab.Binaural.whitenoise()
            lateral_right = sig.ild(3) # attenuate left channel by 1.5 dB and amplify right channel by the same amount
            lateral_left = sig.ild(-3) # vice-versa
        """
        if dB is None:
            return self.right.level - self.left.level
        out = copy.deepcopy(self)  # so that we can return a new sound
        level = numpy.mean(self.level)
        out_levels = (level - dB/2, level + dB/2)
        out.level = out_levels
        out.name = f'{str(dB)}-ild_{self.name}'
        return out

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

            sig = slab.Binaural.whitenoise()
            moving = sig.itd_ramp(from_itd=-0.001, to_itd=0.01)
            moving.play()
        """
        out = copy.deepcopy(self)
        # make the ITD ramps
        left_ramp = numpy.linspace(from_itd / 2, to_itd / 2, self.n_samples)
        right_ramp = numpy.linspace(-from_itd / 2, -to_itd / 2, self.n_samples)
        if self.n_samples >= 8192:
            filter_length = 1024
        elif self.n_samples >= 512:
            filter_length = self.n_samples // 16 * 2  # 1/8th of n_samples, always even
        else:
            raise ValueError('Signal too short! (min 512 samples)')
        out = out.delay(duration=left_ramp, channel=0, filter_length=filter_length)
        out = out.delay(duration=right_ramp, channel=1, filter_length=filter_length)
        out.name = f'ild-ramp_{self.name}'
        return out

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

            sig = slab.Binaural.whitenoise()
            moving = sig.ild_ramp(from_ild=-10, to_ild=10)
            move.play()
        """
        out = self.ild(0)  # set ild to zero
        # make ramps
        left_ramp = numpy.linspace(-from_ild / 2, -to_ild / 2, self.n_samples)
        right_ramp = numpy.linspace(from_ild / 2, to_ild / 2, self.n_samples)
        left_ramp = 10**(left_ramp/20.)
        right_ramp = 10**(right_ramp/20.)
        # multiply channels with ramps
        out.data[:, 0] *= left_ramp
        out.data[:, 1] *= right_ramp
        out.name = f'ild-ramp_{self.name}'
        return out

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
    def azimuth_to_ild(azimuth, frequency=2000, ils=None):
        """
        Get the interaural level difference corresponding to a sound source at a given azimuth and frequency.

        Arguments:
            azimuth (int | float): The azimuth angle of the sound source, negative numbers refer to sources to the left.
            frequency (int | float): Frequency in Hz for which the ITD is estimated.
                Use the default for for sounds with a broadband spectrum.
            ils (dict | None): interaural level spectrum from which the ILD is taken. If None,
            `make_interaural_level_spectrum()` is called. For repeated use, it is better to generate and keep the ils in
            a variable to avoid re-computing it.
        Returns:
            (float): The interaural level difference for a sound source at a given azimuth in decibels.
        Examples::

            ils = slab.Binaural.make_interaural_level_spectrum() # using default KEMAR HRTF
            ild = slab.Binaural.azimuth_to_ild(-90, ils=ils) # ILD equivalent to 90 deg leftward source for KEMAR
        """
        if ils is None:
            ils = Binaural.make_interaural_level_spectrum()
        # interpolate levels at azimuth:
        level_diffs = ils['level_diffs']
        levels = [numpy.interp(azimuth, ils['azimuths'], level_diffs[i, :]) for i in range(level_diffs.shape[0])]
        return numpy.interp(frequency, ils['frequencies'], levels)*-1   # interpolate level difference at frequency

    def at_azimuth(self, azimuth=0, ils=None):
        """
        Convenience function for adding ITD and ILD corresponding to the given `azimuth` to the sound source.
        Values are obtained from azimuth_to_itd and azimuth_to_ild.
        Frequency parameters for these functions are generated from the centroid frequency of the sound.

        Arguments:
            azimuth (int | float): The azimuth angle of the sound source, negative numbers refer to sources to the left.
            ils (dict | None): interaural level spectrum from which the ILD is taken. If None,
            `make_interaural_level_spectrum()` is called. For repeated use, it is better to generate and keep the ils in
            a variable to avoid re-computing it.
        Returns:
            (slab.Binaural): a sound with the appropriate ITD and ILD applied

        """
        centroid = numpy.array(self.spectral_feature(feature='centroid')).mean()
        itd = Binaural.azimuth_to_itd(azimuth, frequency=centroid)
        ild = Binaural.azimuth_to_ild(azimuth, frequency=centroid, ils=ils)
        out = self.itd(duration=itd)
        out.name = f'{azimuth}-azi_{self.name}'
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
        if hrtf is None:
            hrtf = HRTF.kemar()  # load KEMAR as default
        # get HRTF for [0,0] direction:
        idx_frontal = numpy.where((hrtf.sources.vertical_polar[:, 1] == 0) &
                                  (hrtf.sources.vertical_polar[:, 0] == 0))[0][0]
        if not idx_frontal.size: # idx_frontal is empty
            raise ValueError('No frontal direction [0,0] found in HRTF.')
        _, h = hrtf.data[idx_frontal].tf(channels=0, n_bins=12, show=False)  # get low-res version of HRTF spectrum
        h[0] = 1  # avoids low-freq attenuation in KEMAR HRTF (unproblematic for other HRTFs)
        resampled_signal = copy.deepcopy(self)
        # if sound and HRTF has different samplerates, resample the sound, apply the HRTF, and resample back:
        resampled_signal = resampled_signal.resample(hrtf.data[0].samplerate)  # resample to hrtf rate
        filt = Filter(10**(h/20), fir='TF', samplerate=hrtf.data[0].samplerate)
        out = filt.apply(resampled_signal)
        out = out.resample(self.samplerate)
        out.name = f'externalized_{self.name}'
        return out

    @staticmethod
    def make_interaural_level_spectrum(hrtf=None):
        """
        Compute the frequency band specific interaural intensity differences for all sound source azimuth's in
        a head-related transfer function. For every azimuth in the hrtf, the respective transfer function is applied
        to a sound. This sound is then divided into frequency sub-bands. The interaural level spectrum is the level
        difference between right and left for each of these sub-bands for each azimuth.
        When individual HRTFs are not avilable, the level spectrum of the KEMAR mannequin may be used (default).
        Note that the computation may take a few minutes. Save the level spectrum to avoid re-computation, for instance
        with pickle or numpy.save (see documentation on readthedocs for examples).

        Arguments:
            hrtf (None | slab.HRTF): The head-related transfer function used to compute the level spectrum. If None,
                use the recordings from the KEMAR mannequin.
        Returns:
            (dict): A dictionary with keys `samplerate`, `frequencies` [n], `azimuths` [m], and `level_diffs` [n x m],
                where `frequencies` lists the centres of sub-bands for which the level difference was computed, and
                `azimuths` lists the sound source azimuth's in the hrft. `level_diffs` is a matrix of the interaural
                level difference for each sub-band and azimuth.
        Examples::

            ils = slab.Binaural.make_interaural_level_spectrum()  # get the ils from the KEMAR recordings
            ils['samplerate'] # the sampling rate
            ils['frequencies'] # the sub-band frequencies
            ils['azimuths']  # the sound source azimuth's for which the level difference was calculated
            ils['level_diffs'][5, :]  # the level difference for each azimuth in the 5th sub-band
        """
        if not hrtf:
            hrtf = HRTF.kemar()  # load KEMAR by default
        # get the filters for the frontal horizontal arc
        idx = numpy.where((hrtf.sources.vertical_polar[:, 1] == 0) & (
            (hrtf.sources.vertical_polar[:, 0] <= 90) | (hrtf.sources.vertical_polar[:, 0] >= 270)))[0]
        # at this point, we could just get the transfer function of each filter with hrtf.data[idx[i]].tf(),
        # but it may be better to get the spectral left/right differences with ERB-spaced frequency resolution:
        azi = hrtf.sources.vertical_polar[idx, 0]
        # 270<azi<360 -> azi-360 to get negative angles on the left
        azi[azi >= 270] = azi[azi >= 270]-360
        sort = numpy.argsort(azi)
        fbank = Filter.cos_filterbank(samplerate=hrtf.samplerate, pass_bands=True)
        freqs = fbank.filter_bank_center_freqs()
        noise = Sound.pinknoise(duration=5., samplerate=hrtf.samplerate)
        ils = dict()
        ils['samplerate'] = hrtf.samplerate
        ils['frequencies'] = freqs
        ils['azimuths'] = azi[sort]
        ils['level_diffs'] = numpy.zeros((len(freqs), len(idx)))
        for n, i in enumerate(idx[sort]):  # put the level differences in order of increasing angle
            noise_filt = Binaural(hrtf.data[i].apply(noise))
            noise_bank_left = fbank.apply(noise_filt.left)
            noise_bank_right = fbank.apply(noise_filt.right)
            ils['level_diffs'][:, n] = noise_bank_right.level - noise_bank_left.level
        return ils

    def interaural_level_spectrum(self, azimuth, ils=None):
        """
        Apply an interaural level spectrum, corresponding to a sound sources azimuth, to a
        binaural sound. The interaural level spectrum consists of frequency specific interaural level differences
        which are computed from a head related transfer function (see the `make_interaural_level_spectrum()` method).
        The binaural sound is divided into frequency sub-bands and the levels of each sub-band are set according to
        the respective level in the interaural level spectrum. Then, the sub-bands are summed up again into one
        binaural sound.

        Arguments:
            azimuth (int | float): azimuth for which the interaural level spectrum is calculated.
            ils (dict): interaural level spectrum to apply.  If None, `make_interaural_level_spectrum()` is called.
            For repeated use, it is better to generate and keep the ils in a variable to avoid re-computing it.
        Returns:
            (slab.Binaural): A binaural sound with the interaural level spectrum corresponding to the given azimuth.
        Examples::

            noise = slab.Binaural.pinknoise(kind='diotic')
            ils = slab.Binaural.make_interaural_level_spectrum() # using default KEMAR HRTF
            noise.interaural_level_spectrum(azimuth=-45, ils=ils).play()
        """
        if ils is None:
            ils = Binaural.make_interaural_level_spectrum()
        ils_samplerate = ils['samplerate']
        original_samplerate = self.samplerate
        azis = ils['azimuths']
        level_diffs = ils['level_diffs']
        levels = numpy.array([numpy.interp(azimuth, azis, level_diffs[i, :]) for i in range(level_diffs.shape[0])])
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
        out.name = f'ils_{self.name}'
        return out.resample(samplerate=original_samplerate)

    def drr(self, winlength=0.0025):
        """
        Calculate the direct-to-reverberant-ratio, DRR for the impulse input. This is calculated by
        DRR = 10 * log10( X(T0-C:T0+C)^2 / X(T0+C+1:end)^2 ), where X is the approximated integral of the impulse,
        T0 is the time of the direct impulse, and C=2.5ms (Zahorik, P., 2002: 'Direct-to-reverberant energy ratio
        sensitivity', The Journal of the Acoustical Society of America, 112, 2110-2117)

        Arguments:
            winlength (int | float): specifies the length of the direct sound window. This window is used to calculate
            the energy of impulse sound, starting from the position of the peak amplitude of the impulse.

        Returns:
            Direct-to-reverberation value of the input impulse measured in dB

        Return type:
            (float)
        """
        # convert winlength parameter to samples
        winlength = Sound.in_samples(winlength, self.samplerate)
        # winlength should take minimum 2 sample values
        winlength = max(2, winlength)
        if winlength > len(self.data):
            raise ValueError("'winlength' should be shorter than input sound.")
        elif winlength > 0.01 * self.samplerate:
            raise ValueError("'winlength' is suggested to be under 10ms.")
        if self.samplerate < 5000:
            raise ValueError("Sampling frequency is too low, it should be at least 5000 Hz.")
        correction = round(0.5 * self.samplerate / 1000 / 1000) # safety margin of 0.5ms
        # calculate drr for left channel
        impulse = self.data[:, 0]
        impulse_sq = numpy.square(impulse)
        # find the location (index) of the maximal peak in the impulse signal
        peak_index = impulse_sq.argmax()
        # calculate direct and reverberant energy values by integrating under their RMS curve
        win_start_index = max(0, peak_index - correction)
        win_end_index = peak_index + winlength
        direct = impulse[win_start_index: win_end_index]
        direct_e = numpy.trapz(numpy.square(direct))
        reverb = impulse[win_end_index + 1:]
        reverb_e = numpy.trapz(numpy.square(reverb))
        # Calculate DRR as a value in dB
        if direct_e == 0:
            raise ValueError("Direct energy is 0. Please check that your input parameters are reasonable.")
        elif reverb_e == 0:
            raise ValueError("Reverb energy is 0. Are you sure this is an impulse?")
        elif reverb_e < sys.float_info.min:
            raise ValueError("Reverb energy is too low. Please check that your input parameters are reasonable.")
        ratio = direct_e / reverb_e
        drr = 10 * math.log10(ratio)
        # Return output
        return drr

    @staticmethod
    def whitenoise(kind='diotic', **kwargs):
        """
        Generate binaural white noise. `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.whitenoise`.
        """
        if kind == 'dichotic':
            out = Binaural(Sound.whitenoise(n_channels=2, **kwargs))
            out.left = Sound.whitenoise(n_channels=1, **kwargs)
        elif kind == 'diotic':
            out = Binaural(Sound.whitenoise(n_channels=2, **kwargs))
            out.left = out.right
        else:
            raise ValueError("kind must be 'dichotic' or 'diotic'.")
        out.name = f'{kind}-{out.name}'
        return out

    @staticmethod
    def pinknoise(kind='diotic', **kwargs):
        """
        Generate binaural pink noise. `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.pinknoise`.
        """
        out = Binaural.powerlawnoise(alpha=1, kind=kind, **kwargs)
        out.name = f'{kind}-pinknoise'
        return out

    @staticmethod
    def powerlawnoise(kind='diotic', **kwargs):
        """
        Generate binaural power law noise. `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.powerlawnoise`.
        """
        if kind == 'dichotic':
            out = Binaural(Sound.powerlawnoise(n_channels=2, **kwargs))
            out.left = Sound.powerlawnoise(n_channels=1, **kwargs)
        elif kind == 'diotic':
            out = Binaural(Sound.powerlawnoise(n_channels=2, **kwargs))
            out.left = out.right
        else:
            raise ValueError("kind must be 'dichotic' or 'diotic'.")
        out.name = f'{kind}-{out.name}'
        return out

    @staticmethod
    def irn(kind='diotic', **kwargs):
        """
        Generate iterated ripple noise (IRN). `kind`='diotic' produces the same noise samples in both channels,
        `kind`='dichotic' produces uncorrelated noise. The rest is identical to `slab.Sound.irn`.
        """
        out = Binaural(Sound.irn(n_channels=2, **kwargs))
        if kind == 'diotic':
            out.left = out.right
        else:
            raise ValueError("kind must be 'dichotic' or 'diotic'.")
        out.name = f'{kind}-{out.name}'
        return out

    @staticmethod
    def tone(**kwargs):
        """ Identical to slab.Sound.tone, but with two channels. """
        return Binaural(Sound.tone(n_channels=2, **kwargs))

    @staticmethod
    def dynamic_tone(**kwargs):
        """ Identical to slab.Sound.dynamic_tone, but with two channels. """
        return Binaural(Sound.dynamic_tone(n_channels=2, **kwargs))

    @staticmethod
    def harmoniccomplex(**kwargs):
        """ Identical to slab.Sound.harmoniccomplex, but with two channels. """
        return Binaural(Sound.harmoniccomplex(n_channels=2, **kwargs))

    @staticmethod
    def click(**kwargs):
        """ Identical to slab.Sound.click, but with two channels. """
        return Binaural(Sound.click(n_channels=2, **kwargs))

    @staticmethod
    def clicktrain(**kwargs):
        """ Identical to slab.Sound.clicktrain, but with two channels. """
        return Binaural(Sound.clicktrain(**kwargs))

    @staticmethod
    def chirp(**kwargs):
        """ Identical to slab.Sound.chirp, but with two channels. """
        return Binaural(Sound.chirp(**kwargs))

    @staticmethod
    def silence(**kwargs):
        """ Identical to slab.Sound.silence, but with two channels. """
        return Binaural(Sound.silence(**kwargs))

    @staticmethod
    def vowel(**kwargs):
        """ Identical to slab.Sound.vowel, but with two channels. """
        return Binaural(Sound.vowel(**kwargs))

    @staticmethod
    def multitone_masker(**kwargs):
        """ Identical to slab.Sound.multitone_masker, but with two channels. """
        return Binaural(Sound.multitone_masker(**kwargs))

    @staticmethod
    def equally_masking_noise(**kwargs):
        """ Identical to slab.Sound.erb_noise, but with two channels. """
        return Binaural(Sound.equally_masking_noise(**kwargs))
