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
from slab import sound


class Filter(Signal):
<<<<<<< HEAD
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

        if isinstance(data, str):  # load data from a file
            if samplerate is not None:
                raise ValueError('Cannot specify samplerate when initialising Sound from a file.')
            if data[-3::] == 'wav':
                _ = sound.Sound.read(data)
            elif data[-3::] == "npy":
                _ = self.load(data)

            data = _.data
            samplerate = _.samplerate

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

    def apply(self, sig):
        '''
        Apply the filter to signal sig.
        '''
        if (self.samplerate != sig.samplerate) and (self.samplerate != 1):
            raise ValueError('Filter and signal have different sampling rates.')
        out = copy.deepcopy(sig)
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
                        self[:, filt], [1], sig.data.flatten(), axis=0)
            else:
                raise ValueError(
                    'Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
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

    def tf(self, channels='all', nbins=None, plot=True):
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
            fig = plt.figure()
            plt.plot(w, h, linewidth=2)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [dB]')
            plt.title('Frequency Response')
            plt.grid(True)
            plt.show()
            return fig
        else:
            return w, h

    @staticmethod
    # TODO: oversampling fator needed for cochleagram! HP, LP filters needed
    def cos_filterbank(length=5000, bandwidth=1/3, low_lim=0, hi_lim=None, samplerate=None):
        """Create ERB cosine filterbank of n_filters.
        length: Length of signal to be filtered with the generated
                filterbank. The signal length determines the length of the filters.
        samplerate: Sampling rate associated with the signal waveform.
        bandwidth: of the filters (subbands) in octaves (default 1/3)
        low_lim: Lower limit of frequency range (def  saults to 0).
        hi_lim: Upper limit of frequency range (defaults to samplerate/2).
        Example:
        >>> sig = Sound.pinknoise(samplerate=44100)
        >>> fbank = Filter.cos_filterbank(length=sig.nsamples, bandwidth=1/10, low_lim=100, hi_lim=None, samplerate=sig.samplerate)
        >>> fbank.tf(plot=True)
        >>> sig_filt = fbank.apply(sig)
        """
        samplerate = Signal.get_samplerate(samplerate)
        if not hi_lim:
            hi_lim = samplerate / 2
        freq_bins = numpy.fft.rfftfreq(length, d=1/samplerate)
        nfreqs = len(freq_bins)
        center_freqs, bandwidth, erb_spacing = Filter._center_freqs(
            low_lim=low_lim, hi_lim=hi_lim, bandwidth=bandwidth)
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
    def _center_freqs(low_lim, hi_lim, bandwidth=1/3):
        ref_freq = 1000  # Hz, reference for conversion between oct and erb bandwidth
        ref_erb = Filter._freq2erb(ref_freq)
        erb_spacing = Filter._freq2erb(ref_freq*2**bandwidth) - ref_erb
        h = Filter._freq2erb(hi_lim)
        l = Filter._freq2erb(low_lim)
        nfilters = int(numpy.round((h - l) / erb_spacing))
        center_freqs, erb_spacing = numpy.linspace(l, h, nfilters, retstep=True)
        center_freqs = center_freqs[1:-1] 	# we need to exclude the endpoints
        # convert actual erb_spacing to octaves
        bandwidth = numpy.log2(Filter._erb2freq(ref_erb + erb_spacing) / ref_freq)
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

    def filter_bank_center_freqs(self):  # TODO: test this!
        if self.fir:
            raise NotImplementedError('Not implemented for FIR filter banks.')
        freqs = self.frequencies
        center_freqs = numpy.zeros(self.nfilters)
        for i in range(self.nfilters):  # for each filter
            idx = numpy.argmax(self.channel(i).data)  # get index of max Gain
            center_freqs[i] = freqs[idx]  # look-up freq of index -> centre_freq for that filter
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
        if played_signal.samplerate > recorded_signals.samplerate:  # resample higher to lower rate if necessary
            played_signal = played_signal.resample(recorded_signals.samplerate)
        else:
            recorded_signals = recorded_signals.resample(played_signal.samplerate)

        impulse_responses = numpy.zeros([recorded_signals.nsamples, recorded_signals.nchannels])
        for i in range(recorded_signals.nchannels):
            response = 1/len(recorded_signals[:, i]) * scipy.signal.fftconvolve(
                recorded_signals[:, i], played_signal.data.flatten()[::-1], mode='full')
            impulse_responses[:, i] = response[recorded_signals.nsamples //
                                               2:-recorded_signals.nsamples//2+1]

        return Sound(data=impulse_responses, samplerate=played_signal.samplerate)

    @staticmethod
    def equalizing_filterbank(played_signal, recorded_signals, low_lim=50, hi_lim=20000, bandwidth=1/5, reference='max', plot=True):
        '''
        Generate an equalizing filter from the difference between a signal and its recording
        through a linear time-invariant system (speaker, headphones, microphone).
        The filter can be applied to a signal, which when played through the same system will
        preserve the initial spectrum. The main intent of the function is to help with equalizing
        the differences between transfer functions in a loudspeaker array.
        played_signal: Sound or Signal object of the played waveform
        recorded_signals:
        '''
        # number of samples must be even:
        if bool(played_signal.nsamples % 2):
            played_signal.resize(played_signal.nsamples+1)
        if bool(recorded_signals.nsamples % 2):
            recorded_signals.resize(recorded_signals.nsamples+1)
        if played_signal.samplerate > recorded_signals.samplerate:  # resample higher to lower rate if necessary
            played_signal = played_signal.resample(recorded_signals.samplerate)
        else:
            recorded_signals = recorded_signals.resample(played_signal.samplerate)
        # make filterbank
        fbank = Filter.cos_filterbank(length=1000, bandwidth=bandwidth,
                                      low_lim=low_lim, hi_lim=hi_lim, samplerate=played_signal.samplerate)
        center_freqs, _, _ = Filter._center_freqs(low_lim, hi_lim, bandwidth)
        center_freqs = FIlter._erb2freq(center_freqs)
        # get attenuation values in each subband relative to original signal
        levels_in = fbank.apply(played_signal).level
        levels_in = numpy.tile(levels_in, (recorded_signals.nchannels, 1)
                               ).T  # same shaoe as levels_out
        levels_out = numpy.ones((len(center_freqs), recorded_signals.nchannels))
        for idx in range(recorded_signals.nchannels):
            levels_out[:, idx] = fbank.apply(recorded_signals.channel(idx)).level
        if reference == 'max':
            reference = numpy.max(levels_out.flatten())
        elif reference == 'mean':
            reference = numpy.mean(levels_out.flatten())
        amp_diffs = levels_in - levels_out - reference
        freqs = numpy.concatenate(([0], center_freqs, [played_signal.samplerate/2]))
        filt = numpy.zeros((1000, recorded_signals.nchannels))
        for idx in range(recorded_signals.nchannels):
            amps = numpy.concatenate(([0], amp_diffs[:, idx], [0]))
            filt[:, idx] = scipy.signal.firwin2(
                1000, freq=freqs, gain=amps, fs=played_signal.samplerate)

        if plot:  # plot filter result
            plt.plot(freqs[1:-1], np.mean(amp_diffs, axis=1))

        return Filter(data=filt, samplerate=played_signal.samplerate, fir=True)

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
        data = data[2:, :]  # drop the samplerate entry
        return Filter(data, samplerate=samplerate, fir=fir)

    @staticmethod
    def _freq2erb(freq_hz):
        'Converts Hz to human ERBs, using the formula of Glasberg and Moore.'
        return 9.265 * numpy.log(1 + freq_hz / (24.7 * 9.265))

    @staticmethod
    def _erb2freq(n_erb):
        'Converts human ERBs to Hz, using the formula of Glasberg and Moore.'
        return 24.7 * 9.265 * (numpy.exp(n_erb / 9.265) - 1)

=======
	'''
	Class for generating and manipulating filterbanks and transfer functions.

	'''
	# instance properties
	nfilters = property(fget=lambda self: self.nchannels, doc='The number of filters in the bank.')
	ntaps = property(fget=lambda self: self.nsamples, doc='The number of filter taps.')
	nfrequencies = property(fget=lambda self: self.nsamples, doc='The number of frequency bins.')
	frequencies = property(fget=lambda self: numpy.fft.rfftfreq(self.ntaps*2-1, d=1/self.samplerate) if not self.fir else None, doc='The frequency axis of the filter.') # TODO: freqs for FIR?

	def __init__(self, data, samplerate=None, fir=True):

		if isinstance(data, str): # load data from a file
			if samplerate is not None:
				raise ValueError('Cannot specify samplerate when initialising Sound from a file.')
			if data[-3::]=='wav':
				_ = sound.Sound.read(data)
			elif data[-3::]=="npy":
				_ = self.load(data)

			data = _.data
			samplerate = _.samplerate

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
		if fir: # design a FIR filter
			if kind in ['lp', 'bs']:
				pass_zero = True
			elif kind in ['hp', 'bp']:
				pass_zero = False
			filt = scipy.signal.firwin(length, frequency, pass_zero=pass_zero, fs=samplerate)
		else: # FFR filter
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

	def apply(self, sig):
		'''
		Apply the filter to signal sig.
		'''
		if (self.samplerate != sig.samplerate) and (self.samplerate != 1):
			raise ValueError('Filter and signal have different sampling rates.')
		out = copy.deepcopy(sig)
		if self.fir:
			if self.nfilters == sig.nchannels: # filter each channel with corresponding filter
				for i in range(self.nfilters):
					out.data[:,i] = scipy.signal.lfilter(self.data[:,i], [1], out.data[:,i], axis=0)
			elif (self.nfilters == 1) and (sig.nchannels > 1): # filter each channel
				for i in range(self.nfilters):
					out.data[:,i] = scipy.signal.lfilter(self.data.flatten(), [1], out.data[:,i], axis=0)
			elif (self.nfilters > 1) and (sig.nchannels == 1): # apply all filters in bank to signal
				out.data = numpy.empty((sig.nsamples, self.nfilters))
				for filt in range(self.nfilters):
					out.data[:, filt] = scipy.signal.lfilter(self[:, filt], [1], sig.data.flatten(), axis=0)
			else:
				raise ValueError('Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
		else: # FFT filter
			sig_rfft = numpy.fft.rfft(sig.data, axis=0)
			sig_freq_bins = numpy.fft.rfftfreq(sig.nsamples, d=1/sig.samplerate)
			filt_freq_bins = self.frequencies
			# interpolate the FFT filter bins to match the length of the fft of the signal
			if self.nfilters == sig.nchannels: # filter each channel with corresponding filter
				for chan in range(sig.nchannels):
					_filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, chan])
					out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.nsamples)
			elif (self.nfilters == 1) and (sig.nchannels > 1): # filter each channel
				_filt = numpy.interp(sig_freq_bins, filt_freq_bins, self.data.flatten())
				for chan in range(sig.nchannels):
					out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt, sig.nsamples)
			elif (self.nfilters > 1) and (sig.nchannels == 1): # apply all filters in bank to signal
				out.data = numpy.empty((sig.nsamples, self.nfilters))
				for filt in range(self.nfilters):
					_filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, filt])
					out.data[:, filt] = numpy.fft.irfft(sig_rfft.flatten() * _filt, sig.nsamples)
			else:
				raise ValueError('Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
		return out

	def tf(self, channels='all', nbins=None, plot=True):
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
			channels = list(range(self.nfilters)) # now we have a list of filter indices to process
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
					h_interp[:,idx] = numpy.interp(w_interp, w, h[:,idx])
				h = h_interp
				w = w_interp
		if plot:
			fig = plt.figure()
			plt.plot(w, h, linewidth=2)
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Amplitude [dB]')
			plt.title('Frequency Response')
			plt.grid(True)
			plt.show()
			return fig
		else:
			return w, h

	@staticmethod
	def cos_filterbank(length=5000, bandwidth=1/3, low_lim=0, hi_lim=None, samplerate=None): # TODO: oversampling fator needed for cochleagram! HP, LP filters needed
		"""Create ERB cosine filterbank of n_filters.
		length: Length of signal to be filtered with the generated
			filterbank. The signal length determines the length of the filters.
		samplerate: Sampling rate associated with the signal waveform.
		bandwidth: of the filters (subbands) in octaves (default 1/3)
		low_lim: Lower limit of frequency range (def  saults to 0).
		hi_lim: Upper limit of frequency range (defaults to samplerate/2).
		Example:
		>>> sig = Sound.pinknoise(samplerate=44100)
		>>> fbank = Filter.cos_filterbank(length=sig.nsamples, bandwidth=1/10, low_lim=100, hi_lim=None, samplerate=sig.samplerate)
		>>> fbank.tf(plot=True)
		>>> sig_filt = fbank.apply(sig)
		"""
		samplerate = Signal.get_samplerate(samplerate)
		if not hi_lim:
			hi_lim = samplerate / 2
		freq_bins = numpy.fft.rfftfreq(length, d=1/samplerate)
		nfreqs = len(freq_bins)
		center_freqs, bandwidth, erb_spacing = Filter._center_freqs(low_lim=low_lim, hi_lim=hi_lim, bandwidth=bandwidth)
		nfilters = len(center_freqs)
		filts = numpy.zeros((nfreqs, nfilters))
		freqs_erb = Filter._freq2erb(freq_bins)
		for i in range(nfilters):
			l = center_freqs[i] - erb_spacing
			h = center_freqs[i] + erb_spacing
			avg = center_freqs[i]  # center of filter
			rnge = erb_spacing * 2  # width of filter
			filts[(freqs_erb > l) & (freqs_erb < h), i] = numpy.cos((freqs_erb[(freqs_erb > l) & (freqs_erb < h)]- avg) / rnge * numpy.pi)
		return Filter(data=filts, samplerate=samplerate, fir=False)

	@staticmethod
	def _center_freqs(low_lim, hi_lim, bandwidth=1/3):
		ref_freq = 1000 # Hz, reference for conversion between oct and erb bandwidth
		ref_erb = Filter._freq2erb(ref_freq)
		erb_spacing = Filter._freq2erb(ref_freq*2**bandwidth) - ref_erb
		h = Filter._freq2erb(hi_lim)
		l = Filter._freq2erb(low_lim)
		nfilters = int(numpy.round((h - l) / erb_spacing))
		center_freqs, erb_spacing = numpy.linspace(l, h, nfilters, retstep=True)
		center_freqs = center_freqs[1:-1] 	# we need to exclude the endpoints
		# convert actual erb_spacing to octaves
		bandwidth = numpy.log2(Filter._erb2freq(ref_erb + erb_spacing) / ref_freq)
		return center_freqs, bandwidth, erb_spacing

	@staticmethod
	def collapse_subbands(subbands, filter_bank=None):
		if not filter_bank:
			filter_bank = Filter.cos_filterbank(length=subbands.nsamples, samplerate=subbands.samplerate)
		if subbands.samplerate != filter_bank.samplerate:
			raise ValueError('Signal and filter bank need to have the same samplerate!')
		subbands_rfft = numpy.fft.rfft(subbands.data, axis=0)
		subbands = numpy.fft.irfft(subbands_rfft * filter_bank.data, axis=0)
		return Signal(data=subbands.sum(axis=1), samplerate=filter_bank.samplerate)

	def filter_bank_center_freqs(self): # TODO: test this!
		if self.fir:
			raise NotImplementedError('Not implemented for FIR filter banks.')
		freqs = self.frequencies
		center_freqs = numpy.zeros(self.nfilters)
		for i in range(self.nfilters): # for each filter
			idx = numpy.argmax(self.channel(i).data) # get index of max Gain
			center_freqs[i] = freqs[idx] # look-up freq of index -> centre_freq for that filter
		return center_freqs

	@staticmethod
	def impulse_response(played_signal, recorded_signals):
		"""
		compute the impulse response of the system as the deconvolution of the
		played and recorded signals. the IRs are returned in a sound object
		"""
		if bool(played_signal.nsamples%2):
			played_signal.resize(played_signal.nsamples+1)
		if bool(recorded_signals.nsamples%2):
			recorded_signals.resize(recorded_signals.nsamples+1)
		if played_signal.samplerate > recorded_signals.samplerate: # resample higher to lower rate if necessary
			played_signal = played_signal.resample(recorded_signals.samplerate)
		else:
			recorded_signals = recorded_signals.resample(played_signal.samplerate)

		impulse_responses = numpy.zeros([recorded_signals.nsamples,recorded_signals.nchannels])
		for i in range(recorded_signals.nchannels):
			response = 1/len(recorded_signals[:,i]) * scipy.signal.fftconvolve(recorded_signals[:,i], played_signal.data.flatten()[::-1], mode='full')
			impulse_responses[:,i] = response[recorded_signals.nsamples//2:-recorded_signals.nsamples//2+1]

		return Sound(data=impulse_responses, samplerate=played_signal.samplerate)

	@staticmethod
	def equalizing_filterbank(played_signal, recorded_signals, low_lim=50, hi_lim=20000, bandwidth=1/5, reference='max', plot=True):
		'''
		Generate an equalizing filter from the difference between a signal and its recording
		through a linear time-invariant system (speaker, headphones, microphone).
		The filter can be applied to a signal, which when played through the same system will
		preserve the initial spectrum. The main intent of the function is to help with equalizing
		the differences between transfer functions in a loudspeaker array.
		played_signal: Sound or Signal object of the played waveform
		recorded_signals:
		'''
		# number of samples must be even:
		if bool(played_signal.nsamples%2):
			played_signal.resize(played_signal.nsamples+1)
		if bool(recorded_signals.nsamples%2):
			recorded_signals.resize(recorded_signals.nsamples+1)
		if played_signal.samplerate > recorded_signals.samplerate: # resample higher to lower rate if necessary
			played_signal = played_signal.resample(recorded_signals.samplerate)
		else:
			recorded_signals = recorded_signals.resample(played_signal.samplerate)
		# make filterbank
		fbank = Filter.cos_filterbank(length=1000, bandwidth=bandwidth,low_lim=low_lim, hi_lim=hi_lim, samplerate=played_signal.samplerate)
		center_freqs,_,_ = Filter._center_freqs(low_lim, hi_lim, bandwidth )
		center_freqs = Filter._erb2freq(center_freqs)
		# get attenuation values in each subband relative to original signal
		levels_in = fbank.apply(played_signal).level
		levels_in = numpy.tile(levels_in, (recorded_signals.nchannels, 1)).T # same shaoe as levels_out
		levels_out = numpy.ones((len(center_freqs), recorded_signals.nchannels))
		for idx in range(recorded_signals.nchannels):
			levels_out[:, idx] = fbank.apply(recorded_signals.channel(idx)).level
		if reference == 'max':
			reference = numpy.max(levels_out.flatten())
		elif reference == 'mean':
			reference = numpy.mean(levels_out.flatten())
		amp_diffs = levels_in - levels_out - reference
		freqs = numpy.concatenate(([0], center_freqs, [played_signal.samplerate/2]))
		filt = numpy.zeros((1000, recorded_signals.nchannels))
		for idx in range(recorded_signals.nchannels):
			amps = numpy.concatenate(([0], amp_diffs[:, idx], [0]))
			filt[:, idx] = scipy.signal.firwin2(1000, freq=freqs, gain=amps, fs=played_signal.samplerate)

		if plot: # plot filter result
			plt.plot(freqs[1:-1],np.mean(amp_diffs, axis=1))

		return Filter(data=filt, samplerate=played_signal.samplerate, fir=True)

	def save(self, filename):
		'''
		Save the filter in numpy's .npy format to a file.
		'''
		fs = numpy.tile(self.samplerate, reps=self.nfilters)
		fir = numpy.tile(self.fir, reps=self.nfilters)
		fs = fs[numpy.newaxis, :]
		fir = fir[numpy.newaxis, :]
		to_save = numpy.concatenate((fs, fir, self.data)) # prepend the samplerate as new 'filter'
		numpy.save(filename, to_save)

	@staticmethod
	def load(filename):
		'''
		Load a filter from a .npy file.
		'''
		data = numpy.load(filename)
		samplerate = data[0][0] # samplerate is in the first filter
		fir = bool(data[1][0]) # fir is in the first filter
		data = data[2:, :] # drop the samplerate entry
		return Filter(data, samplerate=samplerate, fir=fir)

	@staticmethod
	def _freq2erb(freq_hz):
		'Converts Hz to human ERBs, using the formula of Glasberg and Moore.'
		return 9.265 * numpy.log(1 + freq_hz / (24.7 * 9.265))

	@staticmethod
	def _erb2freq(n_erb):
		'Converts human ERBs to Hz, using the formula of Glasberg and Moore.'
		return 24.7 * 9.265 * (numpy.exp(n_erb / 9.265) - 1)
>>>>>>> 602be8d64cf51367dc4e041669d4fb1a7bb3c117

if __name__ == '__main__':
    filt = Filter.rectangular_filter(frequency=15000, kind='hp', samplerate=44100)
    sig_filt = filt.apply(filt)
    f, Pxx = scipy.signal.welch(sig_filt.data, sig_filt.samplerate, axis=0)
    plt.semilogy(f, Pxx)
    plt.show()
