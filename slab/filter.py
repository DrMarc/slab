'''
Class for HRTFs, filterbanks, and microphone/speaker transfer functions
'''

import copy
from scipy import signal
import numpy

try:
	import matplotlib
	import matplotlib.pyplot as plt
	have_pyplot = True
except ImportError:
	have_pyplot = False
try:
	import scipy.signal
	have_scipy = True
except ImportError:
	have_scipy = False

# TODO: inverse filter
# TODO: making standard filters (high, low, notch, bp)
# TODO: FIR, but also FFR filters?
# TODO: add gammatone filterbank

from slab.signals import Signal # getting the base class

class Filter(Signal):
	'''
	Class for generating and manipulating filterbanks and transfer functions.
	>>> hrtf = HRTF(data='mit_kemar_normal_pinna.sofa') # initialize from sofa file
	>>> print(hrtf)
	<class 'hrtf.HRTF'> sources 710, elevations 14, samples 710, samplerate 44100.0
	>>> sourceidx = hrtf.cone_sources(20)
	>>> hrtf.plot_sources(sourceidx)
	>>> hrtf.plot_tf(sourceidx,ear='left')

	'''

	def estimate_impulse_length(self):
		if self.fir:
			impulse_length = []
			for i in range(self.nfilters):
				_, p, _ = signal.tf2zpk(self.channel(i), [1])
				r = numpy.max(numpy.abs(p))
				impulse_length.append(int(numpy.ceil(numpy.log(1e-9) / numpy.log(r))))
			return impulse_length
		else:
			return None

	# instance properties
	nfilters = property(fget=lambda self: self.nchannels, doc='The number of filters in the bank.')
	ntaps = property(fget=lambda self: self.nsamples, doc='The number of filter taps.')
	nfrequencies = property(fget=lambda self: self.nsamples, doc='The number of frequency bins.')
	frequencies = property(fget=lambda self: numpy.fft.rfftfreq(self.ntaps*2-1, d=1/self.samplerate) if not self.fir else None, doc='The frequency axis of the filter.') # TODO: freqs for FIR?
	impulse_len = property(fget=estimate_impulse_length, doc='Approximate length of filter impulse responses (if fir).')

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
		# TODO: costum filter shapes -> f and kind are tuples with frequencies in Hz and corresponding attenuations in dB. If f is a numpy array it is taken as the target magnitude of the spectrum (imposing one sound's spectrum on the current sound).

		samplerate = Filter.get_samplerate(samplerate)
		if fir: # design a FIR filter
			if kind in ['lp', 'bs']:
				pass_zero = True
			elif kind in ['hp', 'bp']:
				pass_zero = False
			filt = scipy.signal.firwin(length, frequency, pass_zero=pass_zero)
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
				out.data = signal.filtfilt(self.data, [1], out.data, method="gust", irlen=max(self.impulse_len))
			elif (self.nfilters == 1) and (sig.nchannels > 1): # filter each channel
				out.data = signal.filtfilt(self.data, [1], out.data, method="gust", irlen=max(self.impulse_len))
			elif (self.nfilters > 1) and (sig.nchannels == 1): # apply all filters in bank to signal
				out.data = numpy.empty((sig.n_samples, self.nfilters))
				for filt in range(self.nfilters):
					out.data[:, filt] = signal.filtfilt(self[:, filt], [1], sig.data, method="gust", irlen=max(self.impulse_len))
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
					out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt)
			elif (self.nfilters == 1) and (sig.nchannels > 1): # filter each channel
				_filt = numpy.interp(sig_freq_bins, filt_freq_bins, self.data)
				for chan in range(sig.nchannels):
					out.data[:, chan] = numpy.fft.irfft(sig_rfft[:, chan] * _filt)
			elif (self.nfilters > 1) and (sig.nchannels == 1): # apply all filters in bank to signal
				out.data = numpy.empty((sig.n_samples, self.nfilters))
				for filt in range(self.nfilters):
					_filt = numpy.interp(sig_freq_bins, filt_freq_bins, self[:, filt])
					out.data[:, filt] = numpy.fft.irfft(sig_rfft * _filt)
			else:
				raise ValueError('Number of filters must equal number of signal channels, or either one of them must be equal to 1.')
		return out

	def tf(self, plot=True):
		'''
		Computes the transfer function of a filter (magnitude over frequency).
		If plot=True (default) then plot the response, else return magnitude and frequency vectors.
		'''
		if self.fir:
			w, h = signal.freqz(self.channel(0), worN=512, fs=self.samplerate)
			h = signal.absolute(h)
		else:
			w = self.frequencies
			h = self.data
		if plot:
			plt.plot(w, h, linewidth=2)
			plt.xlabel('Frequency (Hz)')
			plt.ylabel('Gain')
			plt.title('Frequency Response')
			plt.grid(True)
		else:
			return w, h


	def inverse(self):
		'''
		Inverse transfer function.
		'''
		pass

if __name__ == '__main__':
	filt = Filter.rectangular_filter(f=15000, kind='hp', samplerate=44100)
	sig_filt = filt.apply(filt)
	f, Pxx = scipy.signal.welch(sig_filt.data, sig_filt.samplerate, axis=0)
	import matplotlib.pyplot as plt
	plt.semilogy(f, Pxx)
	plt.show()
