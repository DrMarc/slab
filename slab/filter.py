import copy
import numpy

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False
try:
    import scipy.signal
except ImportError:
    scipy = False

from slab.signal import Signal  # getting the base class


class Filter(Signal):
    """ Class for generating and manipulating filter banks and transfer functions. Filters can be either finite impulse
    response (FIR) or Fourier filters.
    Arguments:
        data (numpy.ndarray | slab.Signal | list): samples of the filter. If it is an array, the first dimension
            should represent the number of samples and the second one the number of channels. If it's an object,
            it must have a .data attribute containing an array. If it's a list, the elements can be arrays or objects.
            The output will be a multi-channel signal with each channel corresponding to an element of the list.
        samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
        fir (bool): weather this is a finite impulse filter (True) or a Fourier filter (False),
    Attributes:
    Examples. """
    # instance properties
    # TODO: this might be confusing because the filter still has the attributes n_channels and n_samples
    n_filters = property(fget=lambda self: self.n_channels, doc='The number of filters in the bank.')
    n_taps = property(fget=lambda self: self.n_samples, doc='The number of filter taps.')
    n_frequencies = property(fget=lambda self: self.n_samples, doc='The number of frequency bins.')
    frequencies = property(fget=lambda self: numpy.fft.rfftfreq(self.n_taps * 2 - 1, d=1 / self.samplerate)
                           if not self.fir else None, doc='The frequency axis of the filter.')

    def __init__(self, data, samplerate=None, fir=True):
        if fir and scipy is False:
            raise ImportError('FIR filters require scipy.')
        super().__init__(data, samplerate)
        self.fir = fir

    def __repr__(self):
        return f'{type(self)} (\n{repr(self.data)}\n{repr(self.samplerate)}\n{repr(self.fir)})'

    def __str__(self):
        if self.fir:
            return f'{type(self)}, filters {self.n_filters}, FIR: taps {self.n_taps}, samplerate {self.samplerate}'
        return f'{type(self)}, filters {self.n_filters}, FFT: freqs {self.n_frequencies}, samplerate {self.samplerate}'

    @staticmethod
    def band(kind='hp', frequency=100, gain=None, samplerate=None, length=1000, fir=True):
        """ Generate simple passband or stopband filters, or filters with a transfer function defined by  pairs
        of `frequency` and `gain` values.
        Arguments:
            kind: The type of filter to generate. Can be 'lp' (lowpass), 'hp' (highpass), 'bp' (bandpass)
                or 'bs' (bandstop/notch). If `gain` is specified the `kind` argument is ignored.
            frequency (int | float | tuple | list): For a low- or highpass filter, a single integer or float value must
                be given which is the filters edge frequency in Hz. a bandpass or -stop filter takes a tuple of two
                values which are the filters lower and upper edge frequencies. Given a list of values and a list of
                equal length as `gain` the resulting filter will have the specified gain at each frequency.
            gain (None | list): Must be None when generating an lowpass, highpass, bandpass or bandstop filter. For
                generating a custom filter, define a list of the same length as `frequency` with values between
                1.0 (no suppression at that frequency) and 0.0 (maximal suppression at that frequency).
            samplerate (int | None): the samplerate of the signal. If None, use the default samplerate.
            length (int): The number of samples in the filter
            fir: If true generate a finite impulse response filter, else generate a Fourier filter.
        Returns:
            (slab.Filter): a filter with the specified properties
        Examples:
            filt = slab.Filter.band(frequency=3000, kind='lp')  # lowpass filter
            filt = slab.Filter.band(frequency=(100, 2000), kind='bs')  # bandstop filter
            filt = slab.Filter.band(frequency=[100, 1000, 3000, 6000], gain=[0., 1., 0., 1.])  # custom filter """
        samplerate = Filter.get_samplerate(samplerate)
        if fir:  # design a FIR filter
            if scipy is False:
                raise ImportError('Generating FIR filters requires Scipy.')
            if gain is None:  # design band filter
                if kind in ['lp', 'bs']:
                    pass_zero = True
                elif kind in ['hp', 'bp']:
                    pass_zero = False
                if kind in ['hp', 'bs'] and not length % 2:  # high at nyquist requires odd n_taps
                    filt = scipy.signal.firwin(length-1, frequency, pass_zero=pass_zero, fs=samplerate)
                    filt = numpy.append(filt, 0)  # extend to original length
                else:
                    filt = scipy.signal.firwin(length, frequency, pass_zero=pass_zero, fs=samplerate)
            else:
                if not (isinstance(gain, list) and isinstance(frequency, list)):
                    raise ValueError("Gain and frequency must be lists for generating custom filters!")
                dc = dc_gain = []
                if frequency[0] != 0:
                    dc = dc_gain = [0]
                nyq = nyq_gain = []
                if frequency[-1] != samplerate/2:
                    nyq = [samplerate/2]
                    nyq_gain = [0]
                filt = scipy.signal.firwin2(numtaps=length, freq=dc+frequency+nyq, gain=dc_gain+gain+nyq_gain,
                                            fs=samplerate)
        else:  # FFR filter
            df = (samplerate/2) / (length-1)
            if gain is None:
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
            else:
                freqs = numpy.arange(0, (samplerate/2)+df, df)  # frequency bins
                filt = numpy.interp(freqs, numpy.array([0] + frequency + [samplerate/2]), numpy.array([0] + gain + [0]))
        return Filter(data=filt, samplerate=samplerate, fir=fir)

    def apply(self, sig):
        """ Apply the filter to a signal. If signal and filter have the same number of channels,
        each filter channel will be applied to the corresponding channel in the signal.
        If the filter has multiple channels and the signal only 1, each filter is applied to the same signal.
        In that case the filtered signal wil contain the same number of channels as the filter with every
        channel being a copy of the original signal with one filter channel applied. If the filter has only
        one channel and the signal has multiple channels, the same filter is applied to each signal channel.
        Arguments:
            sig (slab.Signal): The signal to be filtered, must be an instance of `slab.Signal` or of another class
                that inherits from it, like `slab.Sound`.
        Returns:
            (slab.Signal): a filtered copy of `sig`.
        Examples:
            filt = slab.Filter.band(frequency=(100, 1500), kind='bp')  # bandpass filter
            sound = slab.Sound.whitenoise()  # generate sound
            filtered_sound = filt.apply(sound)  # apply the filter to the sound """
        if (self.samplerate != sig.samplerate) and (self.samplerate != 1):
            raise ValueError('Filter and signal have different sampling rates.')
        out = copy.deepcopy(sig)
        if self.fir:
            if scipy is False:
                raise ImportError('Applying FIR filters requires Scipy.')
            if self.n_filters == sig.n_channels:  # filter each channel with corresponding filter
                for i in range(self.n_filters):
                    out.data[:, i] = scipy.signal.filtfilt(
                        self.data[:, i], [1], out.data[:, i], axis=0)
            elif (self.n_filters == 1) and (sig.n_channels > 1):  # filter each channel
                for i in range(self.n_filters):
                    out.data[:, i] = scipy.signal.filtfilt(
                        self.data.flatten(), [1], out.data[:, i], axis=0)
            elif (self.n_filters > 1) and (sig.n_channels == 1):  # apply all filters in bank to signal
                out.data = numpy.empty((sig.n_samples, self.n_filters))
                for filt in range(self.n_filters):
                    out.data[:, filt] = scipy.signal.filtfilt(
                        self[:, filt], [1], sig.data, axis=0).flatten()
            else:
                raise ValueError(
                    'Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
        else:  # FFT filter
            sig_rfft = numpy.fft.rfft(sig.data, axis=0)
            sig_freq_bins = numpy.fft.rfftfreq(sig.n_samples, d=1 / sig.samplerate)
            filt_freq_bins = self.frequencies
            # interpolate the FFT filter bins to match the length of the fft of the signal
            if self.n_filters == sig.n_channels:  # filter each channel with corresponding filter
                for chan in range(sig.n_channels):
                    _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, chan])
                    out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.n_samples)
            elif (self.n_filters == 1) and (sig.n_channels > 1):  # filter each channel
                _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self.data.flatten())
                for chan in range(sig.n_channels):
                    out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.n_samples)
            elif (self.n_filters > 1) and (sig.n_channels == 1):  # apply all filters in bank to signal
                out.data = numpy.empty((sig.n_samples, self.n_filters))
                for filt in range(self.n_filters):
                    _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, filt])
                    out.data[:, filt] = numpy.fft.irfft(sig_rfft.flatten() * _filt, sig.n_samples)
            else:
                raise ValueError(
                    'Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
        return out

    def tf(self, channels='all', n_bins=None, show=True, axis=None, **kwargs):
        """ Compute a filter's transfer function (magnitude over frequency) and optionally plot it.
        Arguments:
            channels (str | list | int): the filter channels to compute the transfer function for. Defaults to the
                string "all" which includes all channels. To compute the transfer function for multiple channels, pass
                a list of channel integers. For the transfer function for a single channel pass it's index as integer.
            n_bins (None): number of bins in the transfer function (determines frequency resolution).
                If None, use the maximum number of bins.
            show (bool): whether to show the plot right after drawing.
            axis (matplotlib.axes.Axes | None): axis to plot to. If None create a new plot.
            ** kwargs: keyword arguments for the plot, see documentation of matplotlib.pyplot.plot for details.
        Returns:
            w (numpy.ndarray): the frequency bins in the range from 0 Hz to the Nyquist frequency.
            h (numpy.ndarray: the magnitude of each frequency in `w`.
            None: If `show` is True OR and `axis` was specified, a plot is drawn and nothing is returned.
        Examples:
            filt = slab.Filter.band(frequency=(100, 1500), kind='bp')  # bandpass filter
            filt.tf(show=True)  # compute and plot the transfer functions
            w, h = filt.tf(show=False)  # compute and return the transfer functions """
        if isinstance(channels, int):
            if channels > self.n_filters:  # check chan is in range of n_filters
                raise IndexError("Channel index out of range!")
            else:
                channels = [channels]
        elif isinstance(channels, list):  # check that all the elements are unique and in range
            if len(channels) != len(set(channels)):
                raise ValueError("There should be no duplicates in the list of channels!")
            if min(channels) < 0 or max(channels) < self.n_filters:
                raise IndexError("Channel index out of range!")
            if not all([isinstance(i, int) for i in channels]):
                raise ValueError("Channels must be integers!")
        elif channels == 'all':
            channels = list(range(self.n_filters))  # now we have a list of filter indices to process
        if not n_bins:
            n_bins = self.data.shape[0]
        if self.fir:
            if scipy is False:
                raise ImportError('Computing transfer functions of FIR filters requires Scipy.')
            h = numpy.empty((n_bins, len(channels)))
            for i, idx in enumerate(channels):
                w, _h = scipy.signal.freqz(self.channel(idx), worN=n_bins, fs=self.samplerate)
                h[:, i] = 20 * numpy.log10(numpy.abs(_h.flatten()))
        else:
            w = self.frequencies
            data = self.data[:, channels]
            data[data == 0] += numpy.finfo(float).eps
            h = 20 * numpy.log10(data)
            if not n_bins == len(w): # interpolate if necessary
                w_interp = numpy.linspace(0, w[-1], n_bins)
                h_interp = numpy.zeros((n_bins, len(channels)))
                for idx, _ in enumerate(channels):
                    h_interp[:, idx] = numpy.interp(w_interp, w, h[:, idx])
                h = h_interp
                w = w_interp
        if show or (axis is not None):
            if plt is False:
                raise ImportError('Plotting transfer functions requires matplotlib.')
            if axis is None:
                _, axis = plt.subplots()
            axis.plot(w, h, **kwargs)
            axis.set(title='Frequency [Hz]', xlabel='Amplitude [dB]', ylabel='Frequency Response')
            axis.grid(True)
            if show:
                plt.show()
        else:
            return w, h

    @staticmethod
    # TODO: oversampling factor needed for cochleagram!
    def cos_filterbank(length=5000, bandwidth=1/3, low_cutoff=0, high_cutoff=None, pass_bands=False, samplerate=None):
        # TODO: what is erb, what is the purpose of these filters?
        """ Create a set of ERB-spaced Fourier filters.
        Attributes:
            length (int): The number of bins in the filter, determines the frequency resolution.
            samplerate: sampling rate associated with the signal waveform
            bandwidth: of the filters (subbands) in octaves
            low_cutoff: lower limit of frequency range
            high_cutoff: upper limit of frequency range (defaults to samplerate/2).
            pass_bands: True or False, whether to include half cosine filters as lowpass and highpass
                If True, allows reconstruction of original bandwidth when collapsing subbands.

        >>> sig = Sound.pinknoise(samplerate=44100)
        >>> fbank = Filter.cos_filterbank(length=sig.n_samples, bandwidth=1/10, low_cutoff=100, samplerate=sig.samplerate)
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
            center_freqs = center_freqs[1:-1]  # exclude low and highpass filters
        bandwidth = numpy.log2(Filter._erb2freq(ref_erb + erb_spacing) /
                               ref_freq)  # convert erb_spacing to octaves
        return center_freqs, bandwidth, erb_spacing

    @staticmethod
    def collapse_subbands(subbands, filter_bank=None):
        """ """
        if not filter_bank:
            filter_bank = Filter.cos_filterbank(
                length=subbands.n_samples, samplerate=subbands.samplerate)
        if subbands.samplerate != filter_bank.samplerate:
            raise ValueError('Signal and filter bank need to have the same samplerate!')
        subbands_rfft = numpy.fft.rfft(subbands.data, axis=0)
        subbands = numpy.fft.irfft(subbands_rfft * filter_bank.data, axis=0)
        return Signal(data=subbands.sum(axis=1), samplerate=filter_bank.samplerate)

    def filter_bank_center_freqs(self):
        """ """
        if self.fir:
            raise NotImplementedError('Not implemented for FIR filter banks.')
        freqs = self.frequencies
        center_freqs = numpy.zeros(self.n_filters)
        for i in range(self.n_filters):  # for each filter
            idx = numpy.argmax(self.channel(i).data)  # get index of max Gain
            center_freqs[i] = freqs[idx]  # look-up freq of index -> centre_freq for that filter
        return center_freqs

    @staticmethod
    def equalizing_filterbank(target, signal, length=1000, low_cutoff=200, high_cutoff=None, bandwidth=1/8, alpha=1.0):
        '''
        Generate an equalizing filter from the difference between a signal and a target.
        The main intent of the function is to help with equalizing the differences between transfer functions of
        different loudspeaker. Signal and target are both divided into ERB-sapced frequency bands and the level
        difference is calculated for each band. The differences are normalized to the range 0 to 2 and used as gain
        for the filter in each frequency band. 0 means, that the respective band is maximally supressed, 2 means it is
        maximally amplified. The overall effect of the filter can be regulated by setting alpha (default is 1).
        Alpha < 1 will reduce the total effect of the filter while alpha > 1 will amplify it (WARNING: large filter
        gains may result in temporal distortions of the signal).
        Target and signal must both be instances of slab.Sound. The target must have only a single channel, the signal
        can have multiple ones.
        '''
        if not have_scipy:
            raise ImportError('Generating equalizing filter banks requires Scipy.')
        if target.n_channels > 1:
            raise ValueError("The target sound must have only one channel!")
        if bool(target.n_samples % 2): # number of samples must be even:
            target.resize(target.n_samples + 1)
        if bool(signal.n_samples % 2):
            signal.resize(signal.n_samples + 1)
        if target.samplerate > signal.samplerate: # resample higher to lower rate if necessary
            target = target.resample(signal.samplerate)
        elif target.samplerate < signal.samplerate:
            signal = signal.resample(target.samplerate)
        if high_cutoff is None:
            high_cutoff = target.samplerate/2
        fbank = Filter.cos_filterbank(length=length, bandwidth=bandwidth,
                                      low_cutoff=low_cutoff, high_cutoff=high_cutoff, samplerate=target.samplerate)
        center_freqs, _, _ = Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
        center_freqs = list(Filter._erb2freq(center_freqs))
        center_freqs[-1] = min(center_freqs[-1], target.samplerate/2)
        # level of the target in each of the subbands
        levels_target = fbank.apply(target).level
        # make it the same shape as levels_signal
        levels_target = numpy.tile(levels_target, (signal.nchannels, 1)).T
        # level of each channel in the signal in each of the subbands
        levels_signal = numpy.ones((len(center_freqs), signal.nchannels))
        for idx in range(signal.nchannels):
            levels_signal[:, idx] = fbank.apply(signal.channel(idx)).level
        amp_diffs = levels_target - levels_signal
        max_diffs = numpy.max(numpy.abs(amp_diffs), axis=0)
        max_diffs[max_diffs == 0] = 1
        amp_diffs = amp_diffs/max_diffs
        amp_diffs *= alpha  # apply factor for filter regulation
        amp_diffs += 1  # add 1 because gain = 1 means "do nothing"
        center_freqs = [0] + center_freqs
        filts = numpy.zeros((length, signal.nchannels))
        for chan in range(signal.nchannels):
            gain = [1] + list(amp_diffs[:,chan])
            filt = Filter.band(frequency=list(center_freqs), gain=gain, samplerate=target.samplerate, fir=True)
            filts[:,chan] = filt.data.flatten()
        return Filter(data=filts, samplerate=target.samplerate, fir=True)

    def save(self, filename):
        'Save the filter in Numpys .npy format to a file.'
        fs = numpy.tile(self.samplerate, reps=self.n_filters)
        fir = numpy.tile(self.fir, reps=self.n_filters)
        fs = fs[numpy.newaxis, :]
        fir = fir[numpy.newaxis, :]
        to_save = numpy.concatenate((fs, fir, self.data))  # prepend the samplerate as new 'filter'
        numpy.save(filename, to_save)

    @staticmethod
    def load(filename):
        'Load a filter from a .npy file.'
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
