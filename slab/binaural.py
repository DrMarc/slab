'''
Class for binaural sounds (i.e. with 2 channels).
'''

import copy
import numpy

from slab.sound import Sound
from slab.signal import Signal
from slab.filter import Filter
from slab.hrtf import HRTF


class Binaural(Sound):
    '''
    Class for working with binaural sounds, including ITD and ILD manipulation. Binaural inherits all signal generation functions from the Sound class, but returns binaural signals. Recasting an object of class sound or signal with 1 or 3+ channels calls Sound.copychannel to return a binaural sound with two channels identical to the first channel of the original signal.

    Properties:
        Binaural.left: left (0th)channel
        Binaural.right: right (1st) channel

    >>> sig = Binaural.whitenoise()
    >>> sig.nchannels
    2
    >>> all(sig.left - sig.right)
    False
    '''
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
            if data.nchannels != 2:
                data.copychannel(2)
            self.data = data.data
            self.samplerate = data.samplerate
        elif isinstance(data, (list, tuple)):
            if isinstance(data[0], (Sound, Signal)):
                if data[0].nsamples != data[1].nsamples:
                    raise ValueError('Sounds must have same number of samples!')
                if data[0].samplerate != data[1].samplerate:
                    raise ValueError('Sounds must have same samplerate!')
                super().__init__((data[0].data[:, 0], data[1].data[:, 0]), data[0].samplerate)
            else:
                super().__init__(data, samplerate)
        elif isinstance(data, str):
            super().__init__(data, samplerate)
            if self.nchannels != 2:
                self.copychannel(2) # duplicate channel if monaural file
        else:
            super().__init__(data, samplerate)
            if self.nchannels != 2: # last check that it is a 2-channel sound
                ValueError('Binaural sounds must have two channels!')

    def itd(self, duration):
        '''
        Returns a binaural sound object with one channel delayed with respect to the other channel by `duration`, which
        can be the number of samples or a length of time in seconds. Negative durations delay the right channel (virtual
        sound source moves to the left).

        >>> sig = Binaural.whitenoise()
        >>> _ = sig.itd(1) # delay left channel by 1 sample
        >>> _ = sig.itd(-0.001) # delay right channel by 1ms
        '''
        duration = Sound.in_samples(duration, self.samplerate)
        new = copy.deepcopy(self)  # so that we can return a new signal
        if duration == 0:
            return new  # nothing needs to be shifted
        if duration < 0:  # negative itds by convention shift to the left (i.e. delay right channel)
            channel = 1  # right
        else:
            channel = 0  # left
        new.delay(duration=abs(duration), channel=channel)
        return new

    def ild(self, dB):
        '''
        Returns a sound object with one channel attenuated with respect to the other channel by dB. Negative dB values
        attenuate the right channel (virtual sound source moves to the left). The mean intensity of the signal
        is kept constant.

        >>> sig = Binaural.whitenoise()
        >>> _ = sig.ild(3) # attenuate left channel by 3 dB
        >>> _ = sig.ild(-3) # attenuate right channel by 3 dB
        '''
        new = copy.deepcopy(self)  # so that we can return a new signal
        level = numpy.mean(self.level)
        new_levels = (level + dB/2, level - dB/2)
        new.level = new_levels
        return new

    def itd_ramp(self, from_itd=-6e-4, to_itd=6e-4):
        '''
        Returns a sound object with a linearly increasing or decreasing interaural time difference. This is achieved by
        sinc interpolation of one channel with a dynamic delay. The resulting virtual sound source moves to the left or
        right. `from_itd` and `to_itd` are the itd values at the beginning and end of the sound.The defaults (-0.6ms to
        0.6ms) produce a sound moving from left to right.

        >>> sig = Binaural.whitenoise()
        >>> _ = sig.itd_ramp(from_itd=-0.001, to_itd=0.01)
        '''
        new = copy.deepcopy(self)
        # make the ITD ramps
        left_ramp = numpy.linspace(-from_itd/2, -to_itd/2, self.nsamples)
        right_ramp = numpy.linspace(from_itd/2, to_itd/2, self.nsamples)
        if self.nsamples >= 8192:
            filter_length = 1024
        elif self.nsamples >= 512:
            filter_length = self.nsamples//16 * 2  # 1/8th of nsamples, always even
        else:
            ValueError('Signal too short! (min 512 samples)')
        new.delay(duration=left_ramp, channel=0, filter_length=filter_length)
        new.delay(duration=right_ramp, channel=1, filter_length=filter_length)
        return new

    def ild_ramp(self, from_ild=-50, to_ild=50):
        '''
        Returns a sound object with a linearly increasing or decreasing interaural level difference. The resulting
        virtual sound source moves to the left or right. `from_ild` and `to_ild` are the itd values at the beginning and
        end of the sound. Any existing ILD is removed! The defaults (-50 to 50dB) produce a sound moving from left to right.

        >>> sig = Binaural.whitenoise()
        >>> move = sig.ild_ramp(from_ild=-10, to_ild=10)
        >>> move.play()
        '''
        new = self.ild(0)  # set ild to zero
        # make ramps
        left_ramp = numpy.linspace(-from_ild/2, -to_ild/2, self.nsamples)
        right_ramp = numpy.linspace(from_ild/2, to_ild/2, self.nsamples)
        left_ramp = 10**(left_ramp/20.)
        right_ramp = 10**(right_ramp/20.)
        # multiply channels with ramps
        new.data[:, 0] *= left_ramp
        new.data[:, 1] *= right_ramp
        return new

    @staticmethod
    def azimuth_to_itd(azimuth, frequency=2000, head_radius=8.75):
        '''
        Returns the ITD corresponding to a given `azimuth` and `head_radius`. ITD depends slightly on sound `frequency`.
        For frequencies >= 2 kHz the Woodworth (1962) formula is used. For frequencies <= 500 Hz the low-frequency
        approximation mentioned in Aronson and Hartmann (2014) is used. For frequencies in between, we interpolate
        linearely between the two formulas. Use the default frequency for broadband sounds.

        >>> itd = slab.Binaural.azimuth_to_itd(-90, head_radius=10) # ITD equivalent to 90 deg (left) for a large head
        '''
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
        '''
        Returns the ILD corresponding to a given azimuth (or a sequence of azimuths). ILD depends on sound frequency.
        The ILD is taken from the MIT KEMAR recordings by default, but a different :class:`slab.HRTF` object can be
        supplied. Use the default frequency for broadband sounds.

        >>> ild = slab.Binaural.azimuth_to_ild(-90) # ILD equivalent to 90 deg leftward source using KEMAR HRTF
        '''
        ils = Binaural._make_level_spectrum_filter(hrtf=hrtf)
        freqs = ils[1:, 0]  # get vector of frequencies in ils filter bank
        azis = ils[0, 1:]  # get vector of azimuths in ils filter bank
        ils = ils[1:, 1:]  # the rest is the filter
        # interpolate levels at azimuth
        levels = [numpy.interp(azimuth, azis, ils[i, :]) for i in range(ils.shape[0])]
        # interpolate level difference at frequency
        return numpy.interp(frequency, freqs, levels)

    def at_azimuth(self, azimuth=0):
        '''
        Convenience function for adding ITD and ILD corresponding to the given `azimuth` to the sound. Values are
        obtained from azimuth_to_itd and azimuth_to_ild. Frequency parameters for these functions are generated from the
        centroid frequency of the sound.
        '''
        centroid = numpy.array(self.spectral_feature(feature='centroid')).mean()
        itd = Binaural.azimuth_to_itd(azimuth, frequency=centroid)
        ild = Binaural.azimuth_to_ild(azimuth, frequency=centroid)
        out = self.itd(duration=itd)
        return out.ild(dB=ild)

    def externalize(self, hrtf=None):
        '''
        Convolve the sound object in place with a smoothed HRTF (KEMAR if no :class:`slab.HRTF` object is supplied) to
        evoke the impression of an external sound source without adding directional information.
        See Kulkarni & Colburn (1998) for why that works.
        '''
        from slab import DATAPATH
        if not hrtf:
            hrtf = HRTF(DATAPATH+'mit_kemar_normal_pinna.sofa')  # load the hrtf file
        idx_frontal = numpy.where((hrtf.sources[:, 1] == 0) and (hrtf.sources[:, 0] == 0))[
            0][0]  # get HRTF for [0,0] direction
        _, h = hrtf.data[idx_frontal].tf(channels=0, nbins=12, plot=False)  # get low-res spectrum
        # samplerate shoulf be hrtf.data[0].samplerate, hack to avoid having to resample, ok for externalization if rates are similar
        filt = Filter(10**(h/20), fir=False, samplerate=self.samplerate)
        out = filt.apply(copy.deepcopy(self))
        return out

    @staticmethod
    def _make_level_spectrum_filter(hrtf=None):
        '''
        Generate a level spectrum from the horizontal recordings in a :class:`slab.HRTF` file. The defaut
        :meth:`slab.Filter.cos_filterbank` is used and the same filter bank has to be used when applying
        the level spectrum to a sound.
        '''
        from slab import DATAPATH
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
        if save_standard:
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
            ils = Binaural._make_level_spectrum_filter()  # TODO: should cache this as a file in /data or global
        ils = ils[:, 1:]  # remove the frequency values (not necessary here)
        azis = ils[0, :]  # get vector of azimuths in ils filter bank
        ils = ils[1:, :]  # the rest is the filter
        # interpolate levels at azimuth
        levels = [numpy.interp(azimuth, azis, ils[:, i]) for i in range(ils.shape[1])]
        fbank = Filter.cos_filterbank(length=self.nsamples, samplerate=self.samplerate)
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
        out = Binaural(Sound.whitenoise(duration=duration, nchannels=2,
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
    def irn(frequency=100, gain=1, niter=4, duration=1.0, samplerate=None):
        return Binaural(Sound.irn(frequency=frequency, gain=gain, niter=niter, duration=duration, samplerate=samplerate))

    @staticmethod
    def click(duration=0.0001, samplerate=None):
        return Binaural(Sound.click(duration=duration, samplerate=samplerate, nchannels=2))

    @staticmethod
    def clicktrain(duration=1.0, frequency=500, clickduration=1, samplerate=None):
        return Binaural(Sound.clicktrain(duration=duration, frequency=frequency, clickduration=clickduration, samplerate=samplerate))

    @staticmethod
    def chirp(duration=1.0, from_frequency=100, to_frequency=None, samplerate=None, kind='quadratic'):
        return Binaural(Sound.chirp(duration=duration, from_frequency=from_frequency, to_frequency=to_frequency, samplerate=samplerate, kind=kind))

    @staticmethod
    def silence(duration=1.0, samplerate=None):
        return Binaural(Sound.silence(duration=duration, samplerate=samplerate, nchannels=2))

    @staticmethod
    def vowel(vowel='a', gender=None, glottal_pulse_time=12, formant_multiplier=1, duration=1., samplerate=None):
        return Binaural(Sound.vowel(vowel=vowel, gender=gender, glottal_pulse_time=glottal_pulse_time, formant_multiplier=formant_multiplier, duration=duration, samplerate=samplerate, nchannels=2))

    @staticmethod
    def multitone_masker(duration=1.0, low_cutoff=125, high_cutoff=4000, bandwidth=1/3, samplerate=None):
        return Binaural(Sound.multitone_masker(duration=duration, low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth, samplerate=samplerate))

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
