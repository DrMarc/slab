'''
Class for binaural sounds (i.e. with 2 channels).
This module uses doctests. Use like so:
python -m doctest binaural.py
'''

import copy
import numpy

from slab.sound import Sound
from slab.signals import Signal
from slab.filter import Filter
from slab.hrtf import HRTF


class Binaural(Sound):
    # TODO: Add reverb (image model) room simulation.
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
            if data.nchannels != 2:  # TODO: doesn't result in 2 channels!!!
                data.copychannel(2)
            self.data = data.data
            self.samplerate = data.samplerate
        elif isinstance(data, (list, tuple)):
            if isinstance(data[0], (Sound, Signal)):  # TODO: check is still ok without slab.
                if data[0].nsamples != data[1].nsamples:
                    ValueError('Sounds must have same number of samples!')
                if data[0].samplerate != data[1].samplerate:  # TODO: This doesn't catch for some reason!
                    ValueError('Sounds must have same samplerate!')
                super().__init__((data[0].data[:, 0], data[1].data[:, 0]), data[0].samplerate)
            else:
                super().__init__(data, samplerate)
        else:
            super().__init__(data, samplerate)
            if self.nchannels != 2:
                ValueError('Binaural sounds must have two channels!')

    def itd(self, duration=600e-6):
        '''
        Returns a binaural sound object with one channel delayed with respect to the other channel by duration (*600 microseconds*), which can be the number of samples or a length of time in seconds.
        Negative dB values delay the right channel (virtual sound source moves to the left). itd requires a sound with two channels.
        >>> sig = Binaural.whitenoise()
        >>> _ = sig.itd(1)
        >>> _ = sig.itd(-0.001)

        '''
        duration = Sound.in_samples(duration, self.samplerate)
        new = copy.deepcopy(self)  # so that we can return a new signal
        if duration == 0:
            return new  # nothing needs to be shifted
        if duration < 0:  # negative itds by convention shift to the left (i.e. delay right channel)
            channel = 1  # right
        else:
            channel = 0  # left
        new.delay(duration=abs(duration), chan=channel)
        return new

    def ild(self, dB):
        '''
        Returns a sound object with one channel attenuated with respect to
        the other channel by dB. Negative dB values attenuate the right channel
        (virtual sound source moves to the left). The mean intensity of the signal
        is kept constant.
        ild requires a sound with two channels.
        >>> sig = Binaural.whitenoise()
        >>> _ = sig.ild(3)
        >>> _ = sig.ild(-3)

        '''
        new = copy.deepcopy(self)  # so that we can return a new signal
        level = numpy.mean(self.level)
        new_levels = (level + dB/2, level - dB/2)
        new.level = new_levels
        return new

    def itd_ramp(self, from_itd, to_itd):
        '''
        Returns a sound object with a linearly increasing or decreasing
        interaural time difference. This is achieved by sinc interpolation
        of one channel with a dynamic delay. The resulting virtual sound
        source moves to the left or right. from_itd and to_itd are the itd
        values at the beginning and end of the sound. Delays in between are
        linearely interpolated.
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
            filter_length = self.nsamples//16*2  # 1/8th of nsamples, always even
        else:
            ValueError('Signal too short! (min 512 samples)')
        new.delay(duration=left_ramp, chan=0, filter_length=filter_length)
        new.delay(duration=right_ramp, chan=1, filter_length=filter_length)
        return new

    def ild_ramp(self, from_ild, to_ild):
        '''
        Returns a sound object with a linearly increasing or decreasing interaural level difference. The resulting virtual sound source moves to the left
        or right. from_ild and to_ild are the itd values at the beginning and end of the sound. ILDs in between are linearely interpolated. moving_ild requires a sound with two channels.
        >>> sig = Binaural.whitenoise()
        >>> move = sig.ild_ramp(from_ild=-50, to_ild=50)
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
        Returns the ITD corresponding to a given azimuth and head radius
        (default 8.75 cm). ITD depends slightly on sound frequency.
        For frequencies >= 2 kHz the Woodworth (1962) formula is used.
        For frequencies <= 500 Hz the low-frequency approximation mentioned
        in Aronson and Hartmann (2014) is used. For frequencies in between,
        we interpolate linearely between the two formulas. Use the default
        freqency (*2000*) for broadband sounds.
        Example:
        >>> slab.Binaural.azimuth_to_itd(-90)

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
        Returns the ILD corresponding to a given azimuth. ILD depends on
        sound frequency. The ILD is taken from the MIT KEMAR recordings
        by default, but a different slab.HRTF object can be supplied.
        Use the default freqency (*2000*) for broadband sounds.
        Example:
        >>> slab.Binaural.azimuth_to_ild(-90)

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
        Convenience function for adding ITD and ILD corresponding to the
        given to the sound. Values are obtained from azimuth_to_itd
        and azimuth_to_ild. Frequency parameters for these functions are
        generated from the centroid frequency of the sound.
        Returns a new Binaural object.
        '''
        centroid = numpy.array(self.spectral_feature(feature='centroid')).mean()
        itd = Binaural.azimuth_to_itd(azimuth, frequency=centroid)
        ild = Binaural.azimuth_to_ild(azimuth, frequency=centroid)
        out = self.itd(duration=itd)
        return out.ild(dB=ild)

    def externalize(self, hrtf=None):
        '''
        Convolve the sound object in place with a smoothed HRTF (KEMAR
        if no slab.HRTF object is supplied) to evoke the impression of
        an external sound source without adding directional information.
        See Kulkarni & Colburn (1998) for why that works.
        '''
        from slab import DATAPATH
        if not hrtf:
            hrtf = HRTF(DATAPATH+'mit_kemar_normal_pinna.sofa')  # load the hrtf file
        idx_frontal = numpy.where((hrtf.sources[:, 1] == 0) & (hrtf.sources[:, 0] == 0))[
            0][0]  # get HRTF for [0,0] direction
        w, h = hrtf.data[idx_frontal].tf(channels=0, nbins=12, plot=False)  # get low-res spectrum
        # samplerate shoulf be hrtf.data[0].samplerate, hack to avoid having to resample, ok for externalization if rate are similar
        filt = Filter(10**(h/20), fir=False, samplerate=self.samplerate)
        out = filt.apply(copy.deepcopy(self))
        return out

    def measure_itd(self):
        '''
        Compute the ITD of a binaural sound by cross-correlation in a
        physiological range of lags up to 800 µseconds.
        '''
        pass # TODO: implement!

    @staticmethod
    def _make_level_spectrum_filter(hrtf=None):
        '''
        Generate a level spectrum from the horizontal recordings in an HRTF file. The defaut Filter.cos_filterbank is used and the same filter bank has to be used when applying the level spectrum to a sound.
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
        Apply a frequency-dependend interaural level difference
        corresponding to a given azimuth to a binaural sound.
        The level difference cues are taken from a filter generated
        with the _make_level_spectrum_filter function from an hrtf
        recording. The default will generate the filter from the MIT
        KEMAR recordings. The left and right channel of the sound
        should have the same level.
        Example:
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
        Returns a white noise. If the samplerate is not specified, the global default value will be used. kind = 'diotic' produces the same noise samples in both channels, kind = 'dichotic' produces uncorrelated noise.
        >>> noise = Binaural.whitenoise(kind='diotic')
        '''
        if kind == 'dichotic':
            out = Binaural(Sound.whitenoise(duration=duration, nchannels=2,
                                            samplerate=samplerate, normalise=normalise))
        elif kind == 'diotic':
            out = Binaural(Sound.whitenoise(duration=duration, nchannels=2,
                                            samplerate=samplerate, normalise=normalise))
            out.left = out.right
        return out

    @staticmethod
    def pinknoise(duration=1.0, kind='diotic', samplerate=None, normalise=True):
        '''
        Returns a pink noise. If the samplerate is not specified, the global default value will be used. kind = 'diotic' produces the same noise samples in both channels, kind = 'dichotic' produces uncorrelated noise.
        >>> noise = Binaural.pinknoise(kind='diotic')
        '''
        if kind == 'dichotic':
            out = Binaural(Sound.powerlawnoise(
                duration, 1.0, samplerate=samplerate, nchannels=2, normalise=normalise))
            out.left = Sound.powerlawnoise(
                duration, 1.0, samplerate=samplerate, normalise=normalise)
        elif kind == 'diotic':
            out = Binaural(Sound.powerlawnoise(
                duration, 1.0, samplerate=samplerate, nchannels=2, normalise=normalise))
            out.left = out.right
        return out


if __name__ == '__main__':
    sig = Binaural.pinknoise(duration=0.5, samplerate=44100)
    sig.filter(kind='bp', f=[100, 6000])
    sig.ramp(when='both', duration=0.15)
    sig_itd = sig.itd_ramp(500e-6, -500e-6)
    sig_itd.play()
