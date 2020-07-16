'''
Class for HRTFs, filterbanks, and microphone/speaker transfer functions
'''

import copy
import numpy

try:
    import matplotlib.pyplot as plt
    have_pyplot = True
except ImportError:
    have_pyplot = False
try:
    import scipy.signal
    have_scipy = True
except ImportError:
    have_scipy = False

from slab.signals import Signal  # getting the base class


class Filter(Signal):
    '''
    Class for generating and manipulating filterbanks and transfer functions.

    '''
    # instance properties
    nfilters = property(fget=lambda self: self.nchannels, doc='The number of filters in the bank.')
    ntaps = property(fget=lambda self: self.nsamples, doc='The number of filter taps.')
    nfrequencies = property(fget=lambda self: self.nsamples, doc='The number of frequency bins.')
    frequencies = property(fget=lambda self: numpy.fft.rfftfreq(self.ntaps*2-1, d=1/self.samplerate)
                           if not self.fir else None, doc='The frequency axis of the filter.')  # TODO: freqs for FIR?

    def __init__(self, data, samplerate=None, fir=True):
        if fir and not have_scipy:
            raise ImportError('FIR filters require scipy.')
        super().__init__(data, samplerate)
        self.fir = fir

    def __repr__(self):
        return f'{type(self)} (\n{repr(self.data)}\n{repr(self.samplerate)}\n{repr(self.fir)})'

    def __str__(self):
        if self.fir:
            return f'{type(self)}, filters {self.nfilters}, FIR: taps {self.ntaps}, samplerate {self.samplerate}'
        else:
            return f'{type(self)}, filters {self.nfilters}, FFT: freqs {self.nfrequencies}, samplerate {self.samplerate}'

    @staticmethod
    def rectangular_filter(frequency=100, kind='hp', samplerate=None, length=1000, fir=False):
        '''
        Generates a rectangular filter and returns it as new Filter object.
        frequency: edge frequency in Hz (*1*) or tuple of frequencies for bp and bs.
        type: 'lp' (lowpass), *'hp'* (highpass), bp (bandpass), 'bs' (bandstop, notch)
        TODO: For costum filter shapes f and type are tuples with frequencies
        in Hz and corresponding attenuations in dB. If f is a numpy array it is
        taken as the target magnitude of the spectrum (imposing one sound's
        spectrum on the current sound).
        Examples:
        >>> sig = Sound.whitenoise()
        >>> filt = Filter(f=3000, kind='lp', fir=False)
        >>> sig = filt.apply(sig)
        >>> _ = sig.spectrum()

        '''
        samplerate = Filter.get_samplerate(samplerate)
        if fir:  # design a FIR filter
            if kind in ['lp', 'bs']:
                pass_zero = True
            elif kind in ['hp', 'bp']:
                pass_zero = False
            filt = scipy.signal.firwin(length, frequency, pass_zero=pass_zero, fs=samplerate)
        else:  # FFR filter
            st = 1 / (samplerate/2)
            df = 1 / (st * length)
            filt = numpy.zeros(length)
            if kind == 'lp':
                filt[:round(frequency/df)] = 1
            if kind == 'hp':
                filt[round(frequency/df):] = 1
            if kind == 'bp':
                filt[round(frequency[0]/df):round(frequency[1]/df)] = 1
            if kind == 'notch':
                filt[:round(frequency[0]/df)] = 1
                filt[round(frequency[1]/df):] = 1
        return Filter(data=filt, samplerate=samplerate, fir=fir)

    def apply(self, sig, compensate_shift=False):
        '''
        Apply the filter to signal sig. If signal and filter have the same number of channels,
        each filter channel will be applied to the corresponding channel in the signal.
        If the filter has multiple channels and the signal only 1, each filter is applied to othe same signal.
        In that case the filtered signal wil contain the same number of channels as the filter with every
        channel being a copy of the original signal with one filter channel applied. If the filter has only
        one channel and the signal has multiple channels, the same filter is applied to each signal channel.
        When applying a FIR filter one can set compensate_shift = True. This will padd the output signal
        With zeros equal to half of the filter length and then later remove the same number of samples
        at the end of the signal.
        '''
        if (self.samplerate != sig.samplerate) and (self.samplerate != 1):
            raise ValueError('Filter and signal have different sampling rates.')
        out = copy.deepcopy(sig)
        if compensate_shift:
            if not self.fir:
                raise ValueError('Delay compensation only implemented for FIR filters!')
            else:
                n_shift = int(self.nsamples/2-1)
                pad = numpy.zeros([n_shift, out.nchannels])
                out.data = numpy.concatenate([out, pad])
        if self.fir:
            if self.nfilters == sig.nchannels:  # filter each channel with corresponding filter
                for i in range(self.nfilters):
                    out.data[:, i] = scipy.signal.lfilter(
                        self.data[:, i], [1], out.data[:, i], axis=0)
            elif (self.nfilters == 1) and (sig.nchannels > 1):  # filter each channel
                for i in range(self.nfilters):
                    out.data[:, i] = scipy.signal.lfilter(
                        self.data.flatten(), [1], out.data[:, i], axis=0)
            elif (self.nfilters > 1) and (sig.nchannels == 1):  # apply all filters in bank to signal
                out.data = numpy.empty((sig.nsamples, self.nfilters))
                for filt in range(self.nfilters):
                    out.data[:, filt] = scipy.signal.lfilter(
                        self[:, filt], [1], sig.data, axis=0).flatten()
            else:
                raise ValueError(
                    'Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
            if compensate_shift:
                out.data = out.data[n_shift:, :]
        else:  # FFT filter
            sig_rfft = numpy.fft.rfft(sig.data, axis=0)
            sig_freq_bins = numpy.fft.rfftfreq(sig.nsamples, d=1/sig.samplerate)
            filt_freq_bins = self.frequencies
            # interpolate the FFT filter bins to match the length of the fft of the signal
            if self.nfilters == sig.nchannels:  # filter each channel with corresponding filter
                for chan in range(sig.nchannels):
                    _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, chan])
                    out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.nsamples)
            elif (self.nfilters == 1) and (sig.nchannels > 1):  # filter each channel
                _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self.data.flatten())
                for chan in range(sig.nchannels):
                    out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.nsamples)
            elif (self.nfilters > 1) and (sig.nchannels == 1):  # apply all filters in bank to signal
                out.data = numpy.empty((sig.nsamples, self.nfilters))
                for filt in range(self.nfilters):
                    _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, filt])
                    out.data[:, filt] = numpy.fft.irfft(sig_rfft.flatten() * _filt, sig.nsamples)
            else:
                raise ValueError(
                    'Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
        return out

    def tf(self, channels='all', nbins=None, plot=True, axes=None, show=True, **kwargs):
        '''
        Computes the transfer function of a filter (magnitude over frequency).
        Return transfer functions of filter at index 'channels' (int or list) or,
        if channels='all' (default) return all transfer functions.
        If plot=True (default) then plot the response and return the figure handle,
        else return magnitude and frequency vectors.
        '''
        # check chan is in range of nfilters
        if isinstance(channels, int):
            channels = [channels]
        elif channels == 'all':
            channels = list(range(self.nfilters))  # now we have a list of filter indices to process
        if not nbins:
            nbins = self.data.shape[0]
        if self.fir:
            h = numpy.empty((nbins, len(channels)))
            for i, idx in enumerate(channels):
                w, _h = scipy.signal.freqz(self.channel(idx), worN=nbins, fs=self.samplerate)
                h[:, i] = 20 * numpy.log10(numpy.abs(_h.flatten()))
        else:
            w = self.frequencies
            h = 20 * numpy.log10(self.data[:, channels])
            # interpolate if necessary
            if not nbins == len(w):
                w_interp = numpy.linspace(0, w[-1], nbins)
                h_interp = numpy.zeros((nbins, len(channels)))
                for idx, _ in enumerate(channels):
                    h_interp[:, idx] = numpy.interp(w_interp, w, h[:, idx])
                h = h_interp
                w = w_interp
        if plot:
            if axes is None:
                axes = plt.subplot(111)
            axes.plot(w, h, **kwargs)
            axes.set_xlabel('Frequency [Hz]')
            axes.set_ylabel('Amplitude [dB]')
            axes.set_title('Frequency Response')
            axes.grid(True)
            if show:
                plt.show()
        else:
            return w, h

    @staticmethod
    # TODO: oversampling factor needed for cochleagram!
    def cos_filterbank(length=5000, bandwidth=1/3, low_cutoff=0, high_cutoff=None, pass_bands=False, samplerate=None):
        """Create ERB cosine filterbank of n_filters.
        length ... Length of signal to be filtered with the generated
                filterbank. The signal length determines the length of the filters.
        samplerate ... Sampling rate associated with the signal waveform.
        bandwidth ... of the filters (subbands) in octaves (default 1/3)
        low_cutoff ... Lower limit of frequency range (def  saults to 0).
        high_cutoff ... Upper limit of frequency range (defaults to samplerate/2).
        pass_bands ... boolean [*False*], whether to include half a cosine filter as lowpass and highpass.
                       If True, allows reconstruction of original bandwidth when collapsing subbands.

        Example:
        >>> sig = Sound.pinknoise(samplerate=44100)
        >>> fbank = Filter.cos_filterbank(length=sig.nsamples, bandwidth=1/10, low_cutoff=100, samplerate=sig.samplerate)
        >>> fbank.tf(plot=True)
        >>> sig_filt = fbank.apply(sig)
        """
        samplerate = Signal.get_samplerate(samplerate)
        if not high_cutoff:
            high_cutoff = samplerate / 2
        freq_bins = numpy.fft.rfftfreq(length, d=1/samplerate)
        nfreqs = len(freq_bins)
        center_freqs, bandwidth, erb_spacing = Filter._center_freqs(
            low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth, pass_bands=pass_bands)
        nfilters = len(center_freqs)
        filts = numpy.zeros((nfreqs, nfilters))
        freqs_erb = Filter._freq2erb(freq_bins)
        for i in range(nfilters):
            l = center_freqs[i] - erb_spacing
            h = center_freqs[i] + erb_spacing
            avg = center_freqs[i]  # center of filter
            rnge = erb_spacing * 2  # width of filter
            filts[(freqs_erb > l) & (freqs_erb < h), i] = numpy.cos(
                (freqs_erb[(freqs_erb > l) & (freqs_erb < h)] - avg) / rnge * numpy.pi)
        return Filter(data=filts, samplerate=samplerate, fir=False)

    @staticmethod
    def _center_freqs(low_cutoff, high_cutoff, bandwidth=1/3, pass_bands=False):
        ref_freq = 1000  # Hz, reference for conversion between oct and erb bandwidth
        ref_erb = Filter._freq2erb(ref_freq)
        erb_spacing = Filter._freq2erb(ref_freq*2**bandwidth) - ref_erb
        h = Filter._freq2erb(high_cutoff)
        l = Filter._freq2erb(low_cutoff)
        nfilters = int(numpy.round((h - l) / erb_spacing))
        center_freqs, erb_spacing = numpy.linspace(l, h, nfilters, retstep=True)
        if not pass_bands:
            center_freqs = center_freqs[1:-1] # exclude low and highpass filters
        bandwidth = numpy.log2(Filter._erb2freq(ref_erb + erb_spacing) / ref_freq) # convert erb_spacing to octaves
        return center_freqs, bandwidth, erb_spacing

    @staticmethod
    def collapse_subbands(subbands, filter_bank=None):
        if not filter_bank:
            filter_bank = Filter.cos_filterbank(
                length=subbands.nsamples, samplerate=subbands.samplerate)
        if subbands.samplerate != filter_bank.samplerate:
            raise ValueError('Signal and filter bank need to have the same samplerate!')
        subbands_rfft = numpy.fft.rfft(subbands.data, axis=0)
        subbands = numpy.fft.irfft(subbands_rfft * filter_bank.data, axis=0)
        return Signal(data=subbands.sum(axis=1), samplerate=filter_bank.samplerate)

    def filter_bank_center_freqs(self):
        if self.fir:
            raise NotImplementedError('Not implemented for FIR filter banks.')
        freqs = self.frequencies
        center_freqs = numpy.zeros(self.nfilters)
        for i in range(self.nfilters):  # for each filter
            idx = numpy.argmax(self.channel(i).data)  # get index of max Gain
            center_freqs[i] = freqs[idx]  # look-up freq of index -> centre_freq for that filter
        return center_freqs

    @staticmethod
    def equalizing_filterbank(target, signal, length=1000, low_cutoff=200, high_cutoff=16000, bandwidth=1/8, alpha=1.0):
        '''
        Generate an equalizing filter from the difference between a signal and a target.
        The main intent of the function is to help with equalizing the differences between transfer functions of
        different loudspeaker. Signal and target are both divided into ERB-sapced frequency bands and the level
        diference is calculated for each band. The differences are normalized to the range 0 to 2 and used as gain
        for the filter in each frequency band. 0 means, that the respective band is maximally supressed, 2 means it is
        maximally amplified. The overall effect of the filter can be regulated by setting alpha (default is 1).
        Alpha < 1 will reduce the total effect of the filter while alpha > 1 will amplify it (WARNING: large filter
        gains may result in temporal distortions of the signal).
        Target and signal must both be instances of slab.Sound. The target must have only a single channel, the signal
        can have multiple ones.
        '''
        if target.nchannels > 1:
            raise ValueError("The target sound must have only one channel!")
        if bool(target.nsamples % 2):  # number of samples must be even:
            target.resize(target.nsamples+1)
        if bool(signal.nsamples % 2):
            signal.resize(signal.nsamples+1)
        if target.samplerate > signal.samplerate:  # resample higher to lower rate if necessary
            target = target.resample(signal.samplerate)
        else:
            signal = signal.resample(target.samplerate)
        fbank = Filter.cos_filterbank(length=length, bandwidth=bandwidth, low_cutoff=low_cutoff, high_cutoff=high_cutoff, samplerate=target.samplerate)
        center_freqs, _, _ = Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
        center_freqs = Filter._erb2freq(center_freqs)
        # level of the target in each of the subbands
        levels_target = fbank.apply(target).level
        # make it the same shape as levels_signal
        levels_target = numpy.tile(levels_target, (signal.nchannels, 1)).T
        # level of each channel in the signal in each of the subbands
        levels_signal = numpy.ones((len(center_freqs), signal.nchannels))
        for idx in range(signal.nchannels):
            levels_signal[:, idx] = \
                fbank.apply(signal.channel(idx)).level
        amp_diffs = levels_target - levels_signal
        max_diffs = numpy.max(numpy.abs(amp_diffs), axis=0)
        max_diffs[max_diffs == 0] = 1
        # normalize by divding by maximum for each speaker
        amp_diffs = amp_diffs/max_diffs
        amp_diffs *= alpha  # apply factor for filter regulation
        amp_diffs += 1  # add 1 because gain = 1 means "do nothing"
        # filter freqs must include 0 and nyquist frequency:
        freqs = numpy.concatenate(([0], center_freqs, [target.samplerate/2]))
        filt = numpy.zeros((length, signal.nchannels))  # filter data
        # create the filter for each channel of the signal:
        for idx in range(signal.nchannels):
            # gain must be 0 at 0 Hz and nyquist frequency
            gain = numpy.concatenate(([0], amp_diffs[:, idx], [0]))
            filt[:, idx] = scipy.signal.firwin2(
                length, freq=freqs, gain=gain, fs=target.samplerate)
        return Filter(data=filt, samplerate=target.samplerate, fir=True)

    def save(self, filename):
        '''
        Save the filter in numpy's .npy format to a file.
        '''
        fs = numpy.tile(self.samplerate, reps=self.nfilters)
        fir = numpy.tile(self.fir, reps=self.nfilters)
        fs = fs[numpy.newaxis, :]
        fir = fir[numpy.newaxis, :]
        to_save = numpy.concatenate((fs, fir, self.data))  # prepend the samplerate as new 'filter'
        numpy.save(filename, to_save)

    @staticmethod
    def load(filename):
        '''
        Load a filter from a .npy file.
        '''
        data = numpy.load(filename)
        samplerate = data[0][0]  # samplerate is in the first filter
        fir = bool(data[1][0])  # fir is in the first filter
        data = data[2:, :]  # drop the samplerate and fir entries
        return Filter(data, samplerate=samplerate, fir=fir)

    @staticmethod
    def _freq2erb(freq_hz):
        'Converts Hz to human ERBs, using the formula of Glasberg and Moore.'
        return 9.265 * numpy.log(1 + freq_hz / (24.7 * 9.265))

    @staticmethod
    def _erb2freq(n_erb):
        'Converts human ERBs to Hz, using the formula of Glasberg and Moore.'
        return 24.7 * 9.265 * (numpy.exp(n_erb / 9.265) - 1)


if __name__ == '__main__':
    filt = Filter.rectangular_filter(frequency=15000, kind='hp', samplerate=44100)
    sig_filt = filt.apply(filt)
    f, Pxx = scipy.signal.welch(sig_filt.data, sig_filt.samplerate, axis=0)
    plt.semilogy(f, Pxx)
    plt.show()
