'''
Class for HRTFs, filterbanks, and microphone/speaker transfer functions
'''

import copy
import numpy as np

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
    nfilters = property(fget=lambda self: self.nchannels,
                        doc='The number of filters in the bank.')
    ntaps = property(fget=lambda self: self.nsamples,
                     doc='The number of filter taps.')
    nfrequencies = property(fget=lambda self: self.nsamples,
                            doc='The number of frequency bins.')
    frequencies = property(fget=lambda self:
                           np.fft.rfftfreq(self.ntaps*2-1,
                                           d=1/self.samplerate)
                           if not self.fir else None,  # TODO: freqs for FIR?
                           doc='The frequency axis of the filter.')

    def __init__(self, data, samplerate=None, fir=True):

        if isinstance(data, str):  # load data from a file
            if samplerate is not None:
                raise ValueError('Cannot specify samplerate when initialising'
                                 'Sound from a file.')
            else:
                _ = self.load(data)
                data = _.data
                samplerate = _.samplerate
        if fir and not have_scipy:
            raise ImportError('FIR filters require scipy.')
        super().__init__(data, samplerate)
        self.fir = fir

    def __repr__(self):
        return f'{type(self)}(\n{repr(self.data)}\n{repr(self.samplerate)}'
        '\n{repr(self.fir)})'

    def __str__(self):
        if self.fir:
            return f'{type(self)}, filters {self.nfilters},'
            'FIR: taps {self.ntaps}, samplerate {self.samplerate}'
        else:
            return f'{type(self)}, filters {self.nfilters},'
            'FFT: freqs {self.nfrequencies}, samplerate {self.samplerate}'

    @staticmethod
    def rectangular_filter(frequency=100, kind='hp', samplerate=None,
                           length=1000, fir=False):
        '''
        Generates a rectangular filter and returns it as new Filter object.
        frequency: edge frequency in Hz (*1*) or tuple of frequencies for bp
        and bs.
        type: 'lp' (lowpass), *'hp'* (highpass), bp (bandpass),
        'bs' (bandstop, notch)
        TODO: For costum filter shapes f and type are tuples with frequencies
        in Hz and corresponding attenuations in dB. If f is a np array it is
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
            filt = scipy.signal.firwin(length, frequency, pass_zero=pass_zero,
                                       fs=samplerate)
        else:  # FFR filter
            st = 1 / (samplerate/2)
            df = 1 / (st * length)
            filt = np.zeros(length)
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
        Apply the filter to signal sig.
        '''
        if (self.samplerate != sig.samplerate) and (self.samplerate != 1):
            raise ValueError('Filter and signal have'
                             'different sampling rates.')
        out = copy.deepcopy(sig)
        if compensate_shift:
            if not self.fir:
                raise ValueError('Delay compensation only implemented'
                                 'for FIR filters!')
            else:
                n_shift = int(self.nsamples/2-1)
                pad = np.zeros([n_shift, out.nchannels])
                out.data = np.concatenate([out, pad])
        if self.fir:
            # filter each channel with corresponding filter
            if self.nfilters == sig.nchannels:
                for i in range(self.nfilters):
                    out.data[:, i] = scipy.signal.lfilter(
                        self.data[:, i], [1], out.data[:, i], axis=0)
            # filter each channel
            elif (self.nfilters == 1) and (sig.nchannels > 1):
                for i in range(self.nfilters):
                    out.data[:, i] = scipy.signal.lfilter(
                        self.data.flatten(), [1], out.data[:, i], axis=0)
            # apply all filters in bank to signal
            elif (self.nfilters > 1) and (sig.nchannels == 1):
                out.data = np.repeat(out.data, self.nfilters, axis=1)
                for filt in range(self.nfilters):
                    out.data[:, filt] = scipy.signal.lfilter(
                        self[:, filt], [1], out.data[:, filt], axis=0)
            else:
                raise ValueError('Number of filters must equal number of'
                                 'signal channels, or either one of them must'
                                 'be equal to 1.')
            if compensate_shift:
                out.data = out.data[n_shift:, :]
        else:  # FFT filter
            sig_rfft = np.fft.rfft(sig.data, axis=0)
            sig_freq_bins = np.fft.rfftfreq(sig.nsamples,
                                            d=1/sig.samplerate)
            # interpolate the FFT filter bins to match length of signals fft:
            filt_freq_bins = self.frequencies
            # filter each channel with corresponding filter
            if self.nfilters == sig.nchannels:
                for chan in range(sig.nchannels):
                    _filt = np.interp(sig_freq_bins,
                                      filt_freq_bins, self[:, chan])
                    out.data[:, chan] = np.fft.irfft(sig_rfft[:, chan] *
                                                     _filt, sig.nsamples)
            # filter each channel
            elif (self.nfilters == 1) and (sig.nchannels > 1):
                _filt = np.interp(sig_freq_bins, filt_freq_bins,
                                  self.data.flatten())
                for chan in range(sig.nchannels):
                    out.data[:, chan] = np.fft.irfft(sig_rfft[:, chan] *
                                                     _filt, sig.nsamples)
            # apply all filters in bank to signal
            elif (self.nfilters > 1) and (sig.nchannels == 1):
                out.data = np.empty((sig.nsamples, self.nfilters))
                for filt in range(self.nfilters):
                    _filt = np.interp(sig_freq_bins,
                                      filt_freq_bins, self[:, filt])
                    out.data[:, filt] = np.fft.irfft(sig_rfft.flatten() *
                                                     _filt, sig.nsamples)
            else:
                raise ValueError('Number of filters must equal number of'
                                 'signal channels, or either one of them must'
                                 'be equal to 1.')
        return out

    def tf(self, channels='all', nbins=None, plot=True, axes=None, **kwargs):
        '''
        Computes the transfer function of a filter (magnitude over frequency).
        Return transfer functions of filter at index 'channels' (int or list)
        or, if channels='all' (default) return all transfer functions.
        If plot=True (default) then plot the response and return the figure
        handle, else return magnitude and frequency vectors.
        '''
        # check chan is in range of nfilters
        if isinstance(channels, int):
            channels = [channels]
        elif channels == 'all':
            # list of filter indices to process
            channels = list(range(self.nfilters))
        if not nbins:
            nbins = self.data.shape[0]
        if self.fir:
            h = np.empty((nbins, len(channels)))
            for i, idx in enumerate(channels):
                w, _h = scipy.signal.freqz(self.channel(idx),
                                           worN=nbins, fs=self.samplerate)
                h[:, i] = 20 * np.log10(np.abs(_h.flatten()))
        else:
            w = self.frequencies
            h = 20 * np.log10(self.data[:, channels])
            # interpolate if necessary
            if not nbins == len(w):
                w_interp = np.linspace(0, w[-1], nbins)
                h_interp = np.zeros((nbins, len(channels)))
                for idx, _ in enumerate(channels):
                    h_interp[:, idx] = np.interp(w_interp, w, h[:, idx])
                h = h_interp
                w = w_interp
        if plot:
            show = False
            if axes is None:
                show = True
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
    # TODO: oversampling fator needed for cochleagram! HP, LP filters needed
    def cos_filterbank(length=5000, bandwidth=1/3, low_lim=0, hi_lim=None,
                       samplerate=None):
        """Create ERB cosine filterbank of n_filters.
        length: Length of signal to be filtered with the generated
        filterbank. The signal length determines the length of the filters.
        samplerate: Sampling rate associated with the signal waveform.
        bandwidth: of the filters (subbands) in octaves (default 1/3)
        low_lim: Lower limit of frequency range (def  saults to 0).
        hi_lim: Upper limit of frequency range (defaults to samplerate/2).
        Example:
        >>> sig = Sound.pinknoise(samplerate=44100)
        >>> fbank = Filter.cos_filterbank(length=sig.nsamples, bandwidth=1/10,
        >>> low_lim=100, hi_lim=None, samplerate=sig.samplerate)
        >>> fbank.tf(plot=True)
        >>> sig_filt = fbank.apply(sig)
        """
        samplerate = Signal.get_samplerate(samplerate)
        if not hi_lim:
            hi_lim = samplerate / 2
        freq_bins = np.fft.rfftfreq(length, d=1/samplerate)
        nfreqs = len(freq_bins)
        center_freqs, bandwidth, erb_spacing = Filter._center_freqs(
            low_lim=low_lim, hi_lim=hi_lim, bandwidth=bandwidth)
        nfilters = len(center_freqs)
        filts = np.zeros((nfreqs, nfilters))
        freqs_erb = Filter._freq2erb(freq_bins)
        for i in range(nfilters):
            low = center_freqs[i] - erb_spacing
            hi = center_freqs[i] + erb_spacing
            avg = center_freqs[i]  # center of filter
            rnge = erb_spacing * 2  # width of filter
            filts[(freqs_erb > low) & (freqs_erb < hi), i] = np.cos(
                (freqs_erb[(freqs_erb > low) & (freqs_erb < hi)] - avg)
                / rnge * np.pi)
        return Filter(data=filts, samplerate=samplerate, fir=False)

    @staticmethod
    def _center_freqs(low_lim, hi_lim, bandwidth=1/3):
        # reference in Hz for conversion between oct and erb bandwidth
        ref_freq = 1000
        ref_erb = Filter._freq2erb(ref_freq)
        erb_spacing = Filter._freq2erb(ref_freq*2**bandwidth) - ref_erb
        hi = Filter._freq2erb(hi_lim)
        low = Filter._freq2erb(low_lim)
        nfilters = int(np.round((hi - low) / erb_spacing))
        center_freqs, erb_spacing = np.linspace(low, hi, nfilters,
                                                retstep=True)
        center_freqs = center_freqs[1:-1]     # we need to exclude the endpoints
        # convert actual erb_spacing to octaves
        bandwidth = np.log2(Filter._erb2freq(ref_erb + erb_spacing)
                            / ref_freq)
        return center_freqs, bandwidth, erb_spacing

    @staticmethod
    def collapse_subbands(subbands, filter_bank=None):
        if not filter_bank:
            filter_bank = Filter.cos_filterbank(
                length=subbands.nsamples, samplerate=subbands.samplerate)
        if subbands.samplerate != filter_bank.samplerate:
            raise ValueError('Signal and filter bank need to have'
                             'the same samplerate!')
        subbands_rfft = np.fft.rfft(subbands.data, axis=0)
        subbands = np.fft.irfft(subbands_rfft * filter_bank.data, axis=0)
        return Signal(data=subbands.sum(axis=1),
                      samplerate=filter_bank.samplerate)

    def filter_bank_center_freqs(self):  # TODO: test this!
        if self.fir:
            raise NotImplementedError('Not implemented for FIR filter banks.')
        freqs = self.frequencies
        center_freqs = np.zeros(self.nfilters)
        for i in range(self.nfilters):  # for each filter
            idx = np.argmax(self.channel(i).data)  # get index of max Gain
            # freq of index = center_freq for that filter
            center_freqs[i] = freqs[idx]
        return center_freqs

    @staticmethod
    def impulse_response(played_signal, recorded_signals):
        """
        compute the impulse response of the system as the deconvolution of the
        played and recorded signals. the IRs are returned in a sound object
        """
        if bool(played_signal.nsamples % 2):
            played_signal.resize(played_signal.nsamples+1)
        if bool(recorded_signals.nsamples % 2):
            recorded_signals.resize(recorded_signals.nsamples+1)
        # resample higher to lower rate if necessary
        if played_signal.samplerate > recorded_signals.samplerate:
            played_signal = played_signal.resample(recorded_signals.samplerate)
        else:
            recorded_signals = \
                recorded_signals.resample(played_signal.samplerate)

        impulse_responses = np.zeros([recorded_signals.nsamples,
                                      recorded_signals.nchannels])
        for i in range(recorded_signals.nchannels):
            response = \
                1/len(recorded_signals[:, i]) * scipy.signal.fftconvolve(
                    recorded_signals[:, i], played_signal.data.flatten()[::-1],
                    mode='full')
            impulse_responses[:, i] = response[recorded_signals.nsamples //
                                               2:-recorded_signals.nsamples //
                                               2+1]

        return Filter(data=impulse_responses,
                      samplerate=played_signal.samplerate)

    @staticmethod
    def equalizing_filterbank(target, signal, length=1000, low_lim=200,
                              hi_lim=16000, bandwidth=1/8, factor=None):
        '''
        Generate an equalizing filter from the difference between a signal and
        a target through a linear time-invariant system (speaker, headphones,
        microphone). The main intent of the function is to help with
        equalizing the differences between transfer functions in a loudspeaker
        array. played_signal: Sound or Signal object of the played waveform
        recorded_signals. Arguments:
        target:
            instance of slab.Sound, must have exactly one channel
        signal:
            instance of slab.Sound, can have multiple channels
        length:
            int, filter length in samples
        low_lim:
            int, lowest frequency in Hz for the equalization, defaults to
            200, because making it lower seems to cause artifacts.
        hi_lim:
            Sint, highest frequency in Hz for the equalization
        bandwidth:
            float, spacing of the filter subbands in octaves.
        factor:
            float or tuple of floats. Determines the over all gain of the
            filter. Values below 1 reduce values above increase the gain. If a
            tuple is given, a vector with linearily spaced elements from
            filter[0] to filter[1] is generated. This compensates the decrease
            in filter effect with increasing frequency. Defaults to 1.0
        '''
        if target.nchannels > 1:
            raise ValueError("The target sound must have only one channel!")
        # number of samples must be even:
        if bool(target.nsamples % 2):
            target.resize(target.nsamples+1)
        if bool(signal.nsamples % 2):
            signal.resize(signal.nsamples+1)
        # resample higher to lower rate if necessary
        if target.samplerate > signal.samplerate:
            target = target.resample(signal.samplerate)
        else:
            signal = signal.resample(target.samplerate)
        if hi_lim is None:
            hi_lim = target.samplerate/2
        fbank = Filter.cos_filterbank(length=length, bandwidth=bandwidth,
                                      low_lim=low_lim, hi_lim=hi_lim,
                                      samplerate=target.samplerate)
        center_freqs, _, _ = Filter._center_freqs(low_lim, hi_lim, bandwidth)
        center_freqs = Filter._erb2freq(center_freqs)
        # level of the target in each of the subbands
        levels_target = fbank.apply(target).level
        # make it the same shape as levels_signal
        levels_target = np.tile(levels_target, (signal.nchannels, 1)).T
        # level of each channel in the signal in each of the subbands
        levels_signal = np.ones((len(center_freqs), signal.nchannels))
        for idx in range(signal.nchannels):
            levels_signal[:, idx] = \
                fbank.apply(signal.channel(idx)).level
        amp_diffs = levels_target - levels_signal
        max_diffs = np.max(np.abs(amp_diffs), axis=0)
        max_diffs[max_diffs == 0] = 1
        # normalize by divding by maximum for each speaker
        amp_diffs = amp_diffs/max_diffs
        if factor is not None:  # apply factor
            if isinstance(factor, tuple):  # make linspaced factor vector
                factor = np.expand_dims(np.linspace(factor[0], factor[1],
                                                    len(center_freqs)), axis=1)
            amp_diffs *= factor
        amp_diffs += 1  # add 1 because gain = 1 means "do nothing"
        # filter freqs must include 0 and nyquist frequency:
        freqs = np.concatenate(([0], center_freqs, [target.samplerate/2]))
        filt = np.zeros((length, signal.nchannels))  # filter data
        # create the filter for each channel of the signal:
        for idx in range(signal.nchannels):
            # gain must be 0 at 0 Hz and nyquist frequency
            gain = np.concatenate(([0], amp_diffs[:, idx], [0]))
            filt[:, idx] = scipy.signal.firwin2(
                length, freq=freqs, gain=gain, fs=target.samplerate)

        return Filter(data=filt, samplerate=target.samplerate, fir=True)

    def save(self, filename):
        '''
        Save the filter in np's .npy format to a file.
        '''
        fs = np.tile(self.samplerate, reps=self.nfilters)
        fir = np.tile(self.fir, reps=self.nfilters)
        fs = fs[np.newaxis, :]
        fir = fir[np.newaxis, :]
        # prepend the samplerate as new 'filter'
        to_save = np.concatenate((fs, fir, self.data))
        np.save(filename, to_save)

    @staticmethod
    def load(filename):
        '''
        Load a filter from a .npy file.
        '''
        data = np.load(filename)
        samplerate = data[0][0]  # samplerate is in the first filter
        fir = bool(data[1][0])  # fir is in the first filter
        data = data[2:, :]  # drop the samplerate entry
        return Filter(data, samplerate=samplerate, fir=fir)

    @staticmethod
    def _freq2erb(freq_hz):
        'Converts Hz to human ERBs, using the formula of Glasberg and Moore.'
        return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))

    @staticmethod
    def _erb2freq(n_erb):
        'Converts human ERBs to Hz, using the formula of Glasberg and Moore.'
        return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)
