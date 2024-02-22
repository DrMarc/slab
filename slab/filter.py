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

import slab.signal
from slab.signal import Signal  # getting the base class


class Filter(Signal):
    """
    Class for generating and manipulating filter banks and transfer functions. Filters can be finite impulse
    response ('FIR'), impulse response ('IR'), or transfer functions ('TF'). FIR filters are applied using
    two-way filtering (see scipy.signal.filtfilt), which avoids adding group delays and is intended for high-
    and lowpass filters and the like. IR filters are applied using convolution (scipy.signal.fftconvolve) and
    are intended for long room impulse responses, binaural impulse responses and the like, where group delays
    are intended.

    Arguments:
        data (numpy.ndarray | slab.Signal | list): samples of the filter. If it is an array, the first dimension
            should represent the number of samples and the second one the number of channels. If it's an object,
            it must have a .data attribute containing an array. If it's a list, the elements can be arrays or objects.
            The output will be a multi-channel sound with each channel corresponding to an element of the list.
        samplerate (int | None): the samplerate of the sound. If None, use the default samplerate.
        fir (str): the kind of filter; options are 'FIR', 'IR', or 'TF'.
    Attributes:
        .n_filters: number of filters in the object (overloads `n_channels` attribute in the parent `Signal` class)
        .n_taps: the number of taps in a finite impulse response filter. Analogous to `n_samples` in a `Signal`.
        .n_frequencies: the number of frequency bins in a Fourier filter. Analogous to `n_samples` in a `Signal`.
        .frequencies: the frequency axis of a Fourier filter.
    """
    # instance properties
    n_filters = property(fget=lambda self: self.n_channels, doc='The number of filters in the bank.')
    n_taps = property(fget=lambda self: self.n_samples, doc='The number of filter taps.')
    n_frequencies = property(fget=lambda self: self.n_samples, doc='The number of frequency bins.')
    frequencies = property(fget=lambda self: numpy.fft.rfftfreq(self.n_frequencies * 2 - 1, d=1 / self.samplerate)
                           if self.fir=='TF' else None, doc='The frequency axis of the filter.')

    def __init__(self, data, samplerate=None, fir='FIR'):
        if isinstance(fir, bool):
            raise AttributeError("Assigning a boolean to fir is deprecated. Use 'TF' instead of False, and 'FIR' or 'IR' otherwise.")
        if 'IR' in fir and scipy is False:
            raise ImportError('(F)IR filters require scipy.')
        super().__init__(data, samplerate)
        if not (fir=='FIR' or fir=='IR' or fir=='TF'):
            raise ValueError("fir must be 'FIR', 'IR', or 'TF'!")
        self.fir = fir

    def __repr__(self):
        return f'{type(self)} (\n{repr(self.data)}\n{repr(self.samplerate)}\n{repr(self.fir)})'

    def __str__(self):
        if 'IR' in self.fir:
            return f'{type(self)}, filters {self.n_filters}, FIR: taps {self.n_taps}, samplerate {self.samplerate}'
        return f'{type(self)}, filters {self.n_filters}, TF: freqs {self.n_frequencies}, samplerate {self.samplerate}'

    @staticmethod
    def band(kind='hp', frequency=100, gain=None, samplerate=None, length=1000, fir='FIR'):
        """
        Generate simple passband or stopband filters, or filters with a transfer function defined by  pairs
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
            samplerate (int | None): the samplerate of the sound. If None, use the default samplerate.
            length (int): The number of samples in the filter
            fir (str): If 'FIR' or 'IR' generate a finite impulse response filter, else generate a transfer function.
        Returns:
            (slab.Filter): a filter with the specified properties
        Examples::

            filt = slab.Filter.band(frequency=3000, kind='lp')  # lowpass filter
            filt = slab.Filter.band(frequency=(100, 2000), kind='bs')  # bandstop filter
            filt = slab.Filter.band(frequency=[100, 1000, 3000, 6000], gain=[0., 1., 0., 1.])  # custom filter
        """
        if samplerate is None:
            samplerate = slab.signal._default_samplerate
        if 'IR' in fir:  # design a FIR filter
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
        else:  # TF filter
            df = (samplerate/2) / (length-1)
            if gain is None:
                filt = numpy.zeros(length)
                if kind == 'lp':
                    filt[:round(frequency/df)] = 1
                elif kind == 'hp':
                    filt[round(frequency/df):] = 1
                elif kind == 'bp':
                    filt[round(frequency[0]/df):round(frequency[1]/df)] = 1
                elif kind == 'bs':
                    filt[:round(frequency[0]/df)] = 1
                    filt[round(frequency[1]/df):] = 1
                else:
                    raise ValueError('Unknown filter kind. Use lp, hp, bp, or bs.')
            else:
                freqs = numpy.arange(0, (samplerate/2)+df, df)  # frequency bins
                filt = numpy.interp(freqs, numpy.array([0] + frequency + [samplerate/2]), numpy.array([0] + gain + [0]))
        return Filter(data=filt, samplerate=samplerate, fir=fir)

    def apply(self, sig):
        """
        Apply the filter to a sound. If sound and filter have the same number of channels,
        each filter channel will be applied to the corresponding channel in the sound.
        If the filter has multiple channels and the sound only 1, each filter is applied to the same sound.
        In that case the filtered sound wil contain the same number of channels as the filter with every
        channel being a copy of the original sound with one filter channel applied. If the filter has only
        one channel and the sound has multiple channels, the same filter is applied to each sound channel.
        FIR filters are applied using two-way filtering (see scipy.signal.filtfilt), which avoids adding
        group delays and is intended for high- and lowpass filters and the like. IR filters are applied
        using convolution (scipy.signal.fftconvolve) and are intended for long room impulse responses,
        binaural impulse responses and the like, where group delays are intended.

        Arguments:
            sig (slab.Signal | slab.Sound): The sound to be filtered.
        Returns:
            (slab.Signal | slab.Sound): a filtered copy of `sig`.
        Examples::

            filt = slab.Filter.band(frequency=(100, 1500), kind='bp')  # bandpass filter
            sound = slab.Sound.whitenoise()  # generate sound
            filtered_sound = filt.apply(sound)  # apply the filter to the sound
        """
        if (self.samplerate != sig.samplerate) and (self.samplerate != 1):
            raise ValueError('Filter and sound have different sampling rates.')
        out = copy.deepcopy(sig)
        if self.fir == 'FIR':
            if scipy is False:
                raise ImportError('Applying FIR filters requires Scipy.')
            if out.n_samples < self.n_samples * 3:
                padlen = out.n_samples - 1
            else:
                padlen = None
            if self.n_filters == sig.n_channels:  # filter each channel with corresponding filter
                for i in range(self.n_filters):
                    out.data[:, i] = scipy.signal.filtfilt(
                        self.data[:, i], [1], out.data[:, i], axis=0, padlen=padlen)
            elif (self.n_filters == 1) and (sig.n_channels > 1):  # filter each channel
                for i in range(sig.n_channels):
                    out.data[:, i] = scipy.signal.filtfilt(
                        self.data.flatten(), [1], out.data[:, i], axis=0,  padlen=padlen)
            elif (self.n_filters > 1) and (sig.n_channels == 1):  # apply all filters in bank to sound
                out.data = numpy.empty((sig.n_samples, self.n_filters))
                for filt in range(self.n_filters):
                    out.data[:, filt] = scipy.signal.filtfilt(
                        self[:, filt], [1], sig.data, axis=0,  padlen=padlen).flatten()
            else:
                raise ValueError(
                    'Number of filters must equal number of sound channels, or either one of them must be equal to 1.')
        elif self.fir == 'IR':
            length = sig.n_samples + self.n_taps - 1
            if self.n_filters == sig.n_channels:  # filter each channel with corresponding filter
                out.data = numpy.empty((length, sig.n_channels))
                for i in range(self.n_filters):
                    out.data[:, i] = scipy.signal.fftconvolve(sig[:, i], self[:, i], mode='full')
            elif (self.n_filters == 1) and (sig.n_channels > 1):  # filter each channel
                out.data = numpy.empty((length, sig.n_channels))
                for i in range(sig.n_channels):
                    out.data[:, i] = scipy.signal.fftconvolve(sig[:, i], self[:, 0], mode='full')
            elif (self.n_filters > 1) and (sig.n_channels == 1):  # apply all filters in bank to sound
                out.data = numpy.empty((length, self.n_filters))
                for filt in range(self.n_filters):
                    out.data[:, filt] = scipy.signal.fftconvolve(sig[:, 0], self[:, filt], mode='full').flatten()
            else:
                raise ValueError(
                    'Number of filters must equal number of sound channels, or either one of them must be equal to 1.')
        else:  # FFT filter
            sig_rfft = numpy.fft.rfft(sig.data, axis=0)
            sig_freq_bins = numpy.fft.rfftfreq(sig.n_samples, d=1 / sig.samplerate)
            filt_freq_bins = self.frequencies
            # interpolate the FFT filter bins to match the length of the fft of the sound
            if self.n_filters == sig.n_channels:  # filter each channel with corresponding filter
                for chan in range(sig.n_channels):
                    _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, chan])
                    out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.n_samples)
            elif (self.n_filters == 1) and (sig.n_channels > 1):  # filter each channel
                _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self.data.flatten())
                for chan in range(sig.n_channels):
                    out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.n_samples)
            elif (self.n_filters > 1) and (sig.n_channels == 1):  # apply all filters in bank to sound
                out.data = numpy.empty((sig.n_samples, self.n_filters))
                for filt in range(self.n_filters):
                    _filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, filt])
                    out.data[:, filt] = numpy.fft.irfft(sig_rfft.flatten() * _filt, sig.n_samples)
            else:
                raise ValueError(
                    'Number of filters must equal number of sound channels, or either one of them must be equal to 1.')
        return out

    def tf(self, channels='all', n_bins=None, show=True, axis=None):
        """
        Compute a filter's transfer function (magnitude over frequency) and optionally plot it.

        Arguments:
            channels (str | list | int): the filter channels to compute the transfer function for. Defaults to the
                string "all" which includes all channels. To compute the transfer function for multiple channels, pass
                a list of channel integers. For the transfer function for a single channel pass it's index as integer.
            n_bins (None): number of bins in the transfer function (determines frequency resolution).
                If None, use the maximum number of bins.
            show (bool): whether to show the plot right after drawing.
            axis (matplotlib.axes.Axes | None): axis to plot to. If None create a new plot.
        Returns:
            (numpy.ndarray): the frequency bins in the range from 0 Hz to the Nyquist frequency.
            (numpy.ndarray: the magnitude of each frequency in `w`.
            None: If `show` is True OR and `axis` was specified, a plot is drawn and nothing is returned.
        Examples::

            filt = slab.Filter.band(frequency=(100, 1500), kind='bp')  # bandpass filter
            filt.tf(show=True)  # compute and plot the transfer functions
            w, h = filt.tf(show=False)  # compute and return the transfer functions
        """
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
        if n_bins is None:
            n_bins = self.data.shape[0]
        if 'IR' in self.fir:
            if scipy is False:
                raise ImportError('Computing transfer functions of FIR filters requires Scipy.')
            h = numpy.empty((n_bins, len(channels)))
            w = numpy.empty(n_bins)
            for i, idx in enumerate(channels):
                w, _h = scipy.signal.freqz(self.channel(idx), worN=n_bins, fs=self.samplerate)
                h[:, i] = 20 * numpy.log10(numpy.abs(_h.flatten()))
        else:
            w = self.frequencies
            data = self.data[:, channels]
            data[data == 0] += numpy.finfo(float).eps
            h = 20 * numpy.log10(data)
            if not n_bins == len(w):  # interpolate if necessary
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
            axis.plot(w, h)
            axis.set(title='Frequency [Hz]', ylabel='Amplitude [dB]', xlabel='Frequency Response')
            axis.grid(True)
            if show:
                plt.show()
        else:
            return w, h

    @staticmethod
    # TODO: oversampling factor needed for cochleagram!
    def cos_filterbank(length=5000, bandwidth=1/3, low_cutoff=0, high_cutoff=None, pass_bands=False, n_filters=None, samplerate=None):
        """
        Generate a set of Fourier filters. Each filter's transfer function is given by the positive phase of a
        cosine wave. The amplitude of the cosine is that filters central frequency. Following the organization of the
        cochlea, the width of the filter increases in proportion to it's center frequency. This increase is defined
        by Moore & Glasberg's formula for the equivalent rectangular bandwidth (ERB) of auditory filters.
        The number of filters is either determined by the `n_filters` argument or calculated based on the desired
        `bandwidth` or. This function is used for example to divide a sound into sub-bands for equalization.

        Attributes:
            length (int): The number of bins in each filter, determines the frequency resolution.
            bandwidth (float): Width of the sub-filters in octaves. The smaller the bandwidth, the more filters will be generated.
            low_cutoff (int | float): The lower limit of frequency range in Hz.
            high_cutoff (int | float): The upper limit of frequency range in Hz. If None, use the Nyquist frequency.
            pass_bands (bool): Whether to include a half cosine at the filter bank's lower and upper edge frequency.
                If True, allows reconstruction of original bandwidth when collapsing subbands.
            n_filters (int | None): Number of filters. When this is not None, the `bandwidth` argument is ignored.
            samplerate (int | None): the samplerate of the sound that the filter shall be applied to.
                If None, use the default samplerate.s
        Examples::

            sig = slab.Sound.pinknoise(samplerate=44100)
            fbank = slab.Filter.cos_filterbank(length=sig.n_samples, bandwidth=1/10, low_cutoff=100,
                                          samplerate=sig.samplerate)
            fbank.tf()
            # apply the filter bank to the data. The filtered sound will contain as many channels as there are
            # filters in the bank. Every channel is a copy of the original sound with one filter applied.
            # In this context, the channels are the signals sub-bands:
            sig_filt = fbank.apply(sig)
        """
        if samplerate is None:
            samplerate = slab.signal._default_samplerate
        if not high_cutoff:
            high_cutoff = samplerate / 2
        freq_bins = numpy.fft.rfftfreq(length, d=1/samplerate)
        n_freqs = len(freq_bins)
        center_freqs, bandwidth, erb_spacing = Filter._center_freqs(
            low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth, pass_bands=pass_bands, n_filters=n_filters)
        n_filters = len(center_freqs)
        filts = numpy.zeros((n_freqs, n_filters))
        freqs_erb = Filter._freq2erb(freq_bins)
        for i in range(n_filters):
            l = center_freqs[i] - erb_spacing
            h = center_freqs[i] + erb_spacing
            avg = center_freqs[i]  # center of filter
            width = erb_spacing * 2  # width of filter
            filts[(freqs_erb > l) & (freqs_erb < h), i] = numpy.cos(
                (freqs_erb[(freqs_erb > l) & (freqs_erb < h)] - avg) / width * numpy.pi)
        return Filter(data=filts, samplerate=samplerate, fir='TF')

    @staticmethod
    def _center_freqs(low_cutoff, high_cutoff, bandwidth=1/3, pass_bands=False, n_filters=None):
        ref_freq = 1000  # Hz, reference for conversion between oct and erb bandwidth
        h = Filter._freq2erb(high_cutoff)
        l = Filter._freq2erb(low_cutoff)
        ref_erb = Filter._freq2erb(ref_freq)
        if n_filters is None:
            erb_spacing = Filter._freq2erb(ref_freq*2**bandwidth) - ref_erb
            n_filters = int(numpy.round((h - l) / erb_spacing))
        elif n_filters is not None and pass_bands is False:
            # add 2 so that after omitting pass_bands we get the desired n_filt
            n_filters += 2
        center_freqs, erb_spacing = numpy.linspace(l, h, n_filters, retstep=True)
        if not pass_bands:
            center_freqs = center_freqs[1:-1]  # exclude low and highpass filters
        bandwidth = numpy.log2(Filter._erb2freq(ref_erb + erb_spacing) /
                               ref_freq)  # convert erb_spacing to octaves
        return center_freqs, bandwidth, erb_spacing

    @staticmethod
    def collapse_subbands(subbands, filter_bank=None):
        """
        Sum a sound that has been filtered with a filterbank and which channels represent the sub-bands of
        the original sound. For each sound channel, the fourier transform is calculated and the result is
        multiplied with the corresponding filter in the filter bank. For the resulting spectrum, and inverse
        fourier transform is performed. The resulting sound is summed over the all channels.

        Arguments:
            subbands (slab.Signal): The sound which is divided into subbands by filtering. The number of channels
                in the sound must be equal to the number of filters in the filter bank.
            filter_bank (None | slab.Filter): The filter bank applied to the sound's subbands. The number of
                filters must be equal to the number of channels in the sound. If None a filter bank with the default
                parameters is generated. Note that the filters must have a number of frequency bins equal to the
                number of samples in the sound.
        Returns:
            (slab.Signal): A sound generated from summing the spectra of the subbands.
        Examples::

            sig = slab.Sound.whitenoise()  # generate a sound
            fbank = slab.Filter.cos_filterbank(length=sig.n_samples)  # generate a filter bank
            subbands = fbank.apply(sig)  # divide the sound into subbands by applying the filter
            # by collapsing the subbands, a new sound is generated that is (almost) equal to the original sound:
            collapsed = fbank.collapse_subbands(subbands, fbank)
        """
        new = copy.deepcopy(subbands)
        if not filter_bank:
            filter_bank = Filter.cos_filterbank(
                length=subbands.n_samples, samplerate=subbands.samplerate)
        elif 'IR' in filter_bank.fir:
            raise ValueError("Not implemented for FIR filters!")
        if subbands.samplerate != filter_bank.samplerate:
            raise ValueError('Signal and filter bank need to have the same samplerate!')
        subbands_rfft = numpy.fft.rfft(subbands.data, axis=0)
        subbands = numpy.fft.irfft(subbands_rfft * filter_bank.data, axis=0)
        collapsed = Signal(data=subbands.sum(axis=1), samplerate=filter_bank.samplerate)
        new.data, new.samplerate = collapsed.data, collapsed.samplerate
        return new

    def filter_bank_center_freqs(self):
        """
        Get the maximum of each filter in a filter bank. For filter banks generated with the `cos_filterbank`
        method this corresponds to the filters center frequency.

        Returns:
            (numpy.ndarray): array with length equal to the number of filters in the bank, containing each filter's
                center frequency.
        """
        if 'IR' in self.fir:
            raise NotImplementedError('Not implemented for FIR filter banks.')
        freqs = self.frequencies
        center_freqs = numpy.zeros(self.n_filters)
        for i in range(self.n_filters):  # for each filter
            idx = numpy.argmax(self.channel(i).data)  # get index of max Gain
            center_freqs[i] = freqs[idx]  # look-up freqe of index -> centre_freq for that filter
        return center_freqs

    @staticmethod
    def equalizing_filterbank(reference, sound, length=1000, bandwidth=1/8, low_cutoff=200, high_cutoff=None,
                              alpha=1.0, filt_meth='filtfilt'):
        """
        Generate an equalizing filter from the spectral difference between a `sound` and a `reference`. Both are
        divided into sub-bands using the `cos_filterbank` and the level difference per sub-band is calculated. The
        sub-band frequencies and level differences are then used to generate an equalizing filter that makes the
        spectrum of the `sound` more equal to the one of the `reference`. The main use case is equalizing the
        differences between transfer functions of individual loudspeakers.

        Arguments:
            reference (slab.Sound): The reference for equalization, i.e. what the sound should look like after
                applying the equalization. Must have exactly one channel.
            sound (slab.Sound): The sound to equalize. Can have multiple channels.
            length (int): Number of frequency bins in the filter.
            bandwidth (float): Width of the filters, used to divide the signal into subbands, in octaves. A small
                bandwidth results in a fine tuned transfer function which is useful for equalizing small notches.
            low_cutoff (int | float): The lower limit of frequency range in Hz.
            high_cutoff (int | float): The upper limit of frequency range in Hz. If None, use the Nyquist frequency.
            alpha (float): Filter regularization parameter. Values below 1.0 reduce the filter's effect, values above
                amplify it. WARNING: large filter gains may result in temporal distortions of the sound
            filt_meth (str): filtering method, must be in ('filtfilt', 'fft', 'slab', 'causal'). 'filtfilt' and 'slab'
                applies the filter twice (forward and backward) so the filtered signal is not time-shifted, but in this
                case we need a filter with half of the desired amplitude so the end result has the correct modification.
                For 'fft' and 'causal', the filter with desired amplitude is returned
        Returns:
            (slab.Filter): An equalizing filter bank with a number of filters equal to the number of channels in the
                equalized `sound`.
        Example::

            # generate a sound and apply some arbitrary filter to it
            sound = slab.Sound.pinknoise(samplerate=44100)
            filt = slab.Filter.band(frequency=[100., 800., 2000., 4300., 8000., 14500., 18000.],
                                    gain=[0., 1., 0., 1., 0., 1., 0.], samplerate=sound.samplerate)
            filtered = filt.apply(sound)
            # make an equalizing filter and apply it to the filtered signal. The result looks more like the original
            fbank = slab.Filter.equalizing_filterbank(sound, filtered, low_cutoff=200, high_cutoff=16000)
            equalized = fbank.apply(filtered)
        """
        if scipy is False:
            raise ImportError('Generating equalizing filter banks requires Scipy.')
        if filt_meth not in ('filtfilt', 'fft', 'slab', 'causal'):
            raise ValueError("filt_meth must be one of {'filtfilt', 'fft', 'slab', 'causal'}")
        if reference.n_channels > 1:
            raise ValueError("The reference sound must have only one channel!")
        if bool(reference.n_samples % 2):  # number of samples must be even:
            reference.resize(reference.n_samples + 1)
        if bool(sound.n_samples % 2):
            sound.resize(sound.n_samples + 1)
        if reference.samplerate > sound.samplerate:  # resample higher to lower rate if necessary
            reference = reference.resample(sound.samplerate)
        elif reference.samplerate < sound.samplerate:
            sound = sound.resample(reference.samplerate)
        if high_cutoff is None:
            high_cutoff = reference.samplerate / 2
        # different conditions on filtering method
        is_fir = 'FIR'
        filt_scaling = 1
        if filt_meth == 'fft':
            is_fir = 'TF'
        elif filt_meth in ('slab', 'filtfilt'):
            filt_scaling = 0.5

        fbank = Filter.cos_filterbank(length=length, bandwidth=bandwidth,
                                      low_cutoff=low_cutoff, high_cutoff=high_cutoff, samplerate=reference.samplerate)
        center_freqs, _, _ = Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
        center_freqs = list(Filter._erb2freq(center_freqs))
        center_freqs[-1] = min(center_freqs[-1], reference.samplerate / 2)
        # level of the reference in each of the subbands
        levels_target = fbank.apply(reference).level
        # make it the same shape as levels_signal
        levels_target = numpy.tile(levels_target, (sound.n_channels, 1)).T
        # level of each channel in the sound in each of the subbands
        levels_signal = numpy.ones((len(center_freqs), sound.n_channels))
        for idx in range(sound.n_channels):
            levels_signal[:, idx] = fbank.apply(sound.channel(idx)).level
        amp_diffs_dB = levels_target - levels_signal
        # if we want to keep the scaling
        amp_diffs_dB = filt_scaling * amp_diffs_dB * alpha
        # convert differences from dB to ratio
        amp_diffs = 10 ** (amp_diffs_dB / 20.)
        center_freqs = [0] + center_freqs
        filts = numpy.zeros((length, sound.n_channels))
        for chan in range(sound.n_channels):
            gain = [1] + list(amp_diffs[:, chan])
            filt = Filter.band(frequency=list(center_freqs), gain=gain, samplerate=reference.samplerate, fir=is_fir,
                               length=length)
            filts[:, chan] = filt.data.flatten()
        return Filter(data=filts, samplerate=reference.samplerate, fir=is_fir)

    def save(self, filename):
        """
        Save the filter in Numpy's .npy format to a file.
        Arguments:
            filename(str | pathlib.Path): Full path to which the data is saved.
        """
        fs = numpy.tile(self.samplerate, reps=self.n_filters)
        if self.fir == 'IR':
            fir_coded = 1
        elif self.fir == 'TF':
            fir_coded = 2
        else:
            fir_coded = 0
        fir = numpy.tile(fir_coded, reps=self.n_filters)
        fs = fs[numpy.newaxis, :]
        fir = fir[numpy.newaxis, :]
        to_save = numpy.concatenate((fs, fir, self.data))  # prepend the samplerate as new 'filter'
        numpy.save(filename, to_save)

    @staticmethod
    def load(filename):
        """
        Load a filter from a .npy file.

        Arguments:
            filename(str | pathlib.Path): Full path to the file to load.
        Returns:
            (slab.Filter): The `Filter` loaded from the file.
        """
        data = numpy.load(filename)
        samplerate = data[0][0]  # samplerate is in the first filter
        fir_coded = data[1][0]  # fir is in the first filter
        if fir_coded == 2:
            fir = 'TF'
        elif fir_coded == 1:
            fir = 'IR'
        else:
            fir = 'FIR'
        data = data[2:, :]  # drop the samplerate and fir entries
        return Filter(data, samplerate=samplerate, fir=fir)

    @staticmethod
    def _freq2erb(freq_hz):
        """ Converts Hz to human ERBs, using the formula of Glasberg and Moore. """
        return 9.265 * numpy.log(1 + freq_hz / (24.7 * 9.265))

    @staticmethod
    def _erb2freq(n_erb):
        """ Converts human ERBs to Hz, using the formula of Glasberg and Moore. """
        return 24.7 * 9.265 * (numpy.exp(n_erb / 9.265) - 1)
