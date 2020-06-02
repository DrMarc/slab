'''
Base class for Signal data (sounds and filters).
This module uses doctests. Use like so:
python -m doctest slab.py
'''

import copy
import numpy
try:
	import scipy.signal
	have_scipy = True
except ImportError:
	have_scipy = False

_default_samplerate = 8000 # Hz

class Signal:
	'''
	Base class for Signal data (sounds and filters)
	Provides duration, nsamples, times, nchannels properties,
	slicing, and conversion between samples and times.
	This class is intended to be subclassed. See Sound class for an example.

	Minimal example:
	class SignalWithInfo(Signal):
	def __init__(self,data,samplerate=None,info=None):
		# call the baseclass init() so that we don't have to repeat that code here:
		super().__init__(data,samplerate)
		# add the new attribute
		self.info = info


	The following arguments are used to initialise a Signal object:

	``data``
		Can be an array, a function or a sequence (list or tuple).
		If its an array, it should have shape ``(nsamples, nchannels)``. If its a
		function, it should be a function f(t). If its a sequence, the items
		in the sequence can be functions, arrays or Signal objects.
		The output will be a multi-channel Signal with channels corresponding to
		Signals for each element of the sequence.
	``samplerate=None``
		The samplerate, if necessary, will use the default (for an array or
		function) or the samplerate of the data (for a filename).

		Examples:
	>>> import slab
	>>> import numpy
	>>> sig = slab.Signal(numpy.ones([10,2]),samplerate=10)
	>>> print(sig)
	<class 'slab.signals.Signal'> duration 1.0, samples 10, channels 2, samplerate 10

		**Properties**
	>>> sig.duration
	1.0
	>>> sig.nsamples
	10
	>>> sig.nchannels
	2
	>>> len(sig.times)
	10

		**Slicing**
	Signal implements __getitem__ and __setitem___ and thus supports slicing.
	Slicing returns numpy.ndarrays or floats, not Signal objects.
	You can also set values using slicing:
	>>> sig[:5] = 0

	will set the first 5 samples to zero.
	You can also select a subset of channels:
	>>> sig[:,1]
	array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])

	would be data in the second channel. To extract a channel as a Signal or subclass object
	use sig.channel(1).

	As a convenient shortcut, channel data can also be returned and set by instance attributes:
	>>> sig.ch1 # this is equivalent to sig[:,1]
	array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])

	These channel attributes also allow setting:
	>>> sig.ch1 = 1

	Signals support arithmatic operations (add, sub, mul, truediv, neg ['-sig' inverts phase]):
	>>> sig2 = sig * 2
	>>> sig2[-1,1]
	2.0

		**Static methods**

	Signal.in_samples(time,samplerate)
		Converts time values in seconds to samples.
		This is used to enable input in either samples (int) or seconds (float) in the class.

	Signal.get_samplerate(samplerate)
		Return samplerate if supplied, otherwise return the default samplerate.

	Signal.set_default_samplerate(samplerate)
		Sets the global default samplerate for Signal objects, by default 8000 Hz.

		**Instance methods**

	sig.channel(n)
		Returns the nth channel as new object of the calling class.
		>>> print(sig.channel(0))
		<class 'slab.signals.Signal'> duration 1.0, samples 10, channels 1, samplerate 10

	sig.resize(L)
		Extends or contracts the length of the data in the object in place to have L samples.
		>>> sig.resize(8)
		>>> print(sig)
		<class 'slab.signals.Signal'> duration 0.8, samples 8, channels 2, samplerate 10
	'''

	# instance properties
	nsamples = property(fget=lambda self: self.data.shape[0], fset=lambda self,l: self.resize(l),
						doc='The number of samples in the Signal. Setting calls resize.')
	duration = property(fget=lambda self: self.data.shape[0] / self.samplerate,
						doc='The length of the Signal in seconds.')
	times = property(fget=lambda self: numpy.arange(self.data.shape[0], dtype=float) / self.samplerate,
						doc='An array of times (in seconds) corresponding to each sample.')
	nchannels = property(fget=lambda self: self.data.shape[1],
						doc='The number of channels in the Signal.')

	# __methods (class creation, printing, and slice functionality)
	def __init__(self, data, samplerate=None):
		self.samplerate = Signal.get_samplerate(samplerate)
		if isinstance(data, numpy.ndarray):
			self.data = numpy.array(data, dtype='float')
		elif isinstance(data, (list, tuple)):
			kwds = {}
			if samplerate is not None:
				kwds['samplerate'] = samplerate
			channels = tuple(Signal(c, **kwds) for c in data)
			self.data = numpy.hstack(channels)
			self.samplerate = channels[0].samplerate
		elif hasattr(data, 'data') and hasattr(data, 'samplerate'): # any object with data and samplerate attributes can be recast as Signal
			self.data = data.data
			self.samplerate = data.samplerate
		else:
			raise TypeError('Cannot initialise Signal with data of class ' + str(data.__class__))
		if len(self.data.shape) == 1:
			self.data.shape = (len(self.data), 1)
		elif self.data.shape[1] > self.data.shape[0]:
			self.data = self.data.T

	def __repr__(self):
		return f'{type(self)} (\n{repr(self.data)}\n{repr(self.samplerate)})'

	def __str__(self):
		return f'{type(self)} duration {self.duration}, samples {self.nsamples}, channels {self.nchannels}, samplerate {self.samplerate}'

	def __getitem__(self, key):
		return self.data.__getitem__(key)

	def __setitem__(self, key, value):
		return self.data.__setitem__(key, value)

	# def __getattr__(self, chan): # allows to get channels as sig.ch0, sig.ch1
	# 	if chan[:2] == 'ch':
	# 		return self.__dict__['data'][:,int(chan[2:])]
	# 	else:
	# 		raise AttributeError(chan)
	#
	# def __setattr__(self, chan, new):  # allows to set channel data as sig.ch0 = 0
	# 	if chan[:2] == 'ch': # setting channel data
	# 		self.__dict__['data'][:,int(chan[2:])] = new
	# 	else: # setting any other attribute
	# 		self.__dict__[chan] = new

	# arithmatic operators
	def __add__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other, type(self)):
			new.data = self.data + other.data
		else:
			new.data = self.data + other
		return new
	__radd__ = __add__

	def __sub__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other, type(self)):
			new.data = self.data - other.data
		else:
			new.data = self.data - other
		return new
	__rsub__ = __sub__

	def __mul__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other, type(self)):
			new.data = self.data * other.data
		else:
			new.data = self.data * other
		return new
	__rmul__ = __mul__

	def __truediv__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other, type(self)):
			new.data = self.data / other.data
		else:
			new.data = self.data / other
		return new
	__rtruediv__ = __truediv__

	def __neg__(self):
		new = copy.deepcopy(self)
		new.data = self.data*-1
		return new

	def __len__(self):
		return self.nsamples

	# static methods (belong to the class, but can be called without creating an instance)
	@staticmethod
	def in_samples(ctime, samplerate):
		'''Converts time values in seconds to samples.
		This is used to enable input in either samples (integers) or seconds (floating point numbers) in the class.
		ctime can be of type int, float, numpy.ndarray of numpy.int64 or numpy.float64, list of a mix of ints or floats, or tuple of a mix of ints or floats.
		'''
		if isinstance(ctime, (int, numpy.int64)): # single int is treated as samples
			out = ctime
		elif isinstance(ctime, (float, numpy.float64)):
			out = int(numpy.rint(ctime * samplerate))
		elif isinstance(ctime, numpy.ndarray):
			if isinstance(ctime[0], numpy.int64):
				out = ctime
			elif isinstance(ctime[0], numpy.float64):
				out = numpy.int64(numpy.rint(ctime * samplerate))
			else:
				ValueError('Unsupported type in numpy.ndarray (must be int64 of float 64)')
		elif isinstance(ctime, (list, tuple)): # convert one by one
			out = numpy.empty_like(ctime)
			for i, c in enumerate(ctime):
				out[i] = numpy.int64(Signal.in_samples(c, samplerate))
		else:
			ValueError('Unsupported type for ctime (must be int, float, numpy.ndarray of ints or floats, list or tuple)!')
		return out

	@staticmethod
	def get_samplerate(samplerate):
		'Return samplerate if supplied, otherwise return the default samplerate.'
		if samplerate is None:
			return _default_samplerate
		else:
			return samplerate

	@staticmethod
	def set_default_samplerate(samplerate):
		'Sets the global default samplerate for Signal objects, by default 8000 Hz.'
		global _default_samplerate
		_default_samplerate = samplerate

	# instance methods (belong to instances created from the class)
	def channel(self, n):
		'Returns the nth channel as new object of the calling class.'
		new = copy.deepcopy(self)
		new.data = self.data[:, n]
		new.data.shape = (len(new.data), 1)
		return new

	def channels(self):
		'Returns generator that yields channel data as objects of the calling class.'
		return (self.channel(i) for i in range(self.nchannels))

	def resize(self, L):
		'Extends or contracts the length of the data in the object in place to have L samples.'
		L = Signal.in_samples(L, self.samplerate)
		if L == len(self.data):
			pass # already correct size
		elif L < len(self.data):
			self.data = self.data[:L, :]
		else:
			padding = numpy.zeros((L - len(self.data), self.nchannels))
			self.data = numpy.concatenate((self.data, padding))

	def resample(self, samplerate):
		'Returns a resampled version of the sound. Requires scipy.signal.'
		if not have_scipy:
			raise ImportError('Resampling requires scipy.signal.')
		if self.samplerate == samplerate:
			return self
		else:
			out = copy.deepcopy(self)
			new_nsamples = int(numpy.rint(samplerate*self.duration))
			new_signal = numpy.zeros((new_nsamples, self.nchannels))
			for chan in range(self.nchannels):
				new_signal[:, chan] = scipy.signal.resample(self.channel(chan), new_nsamples).flatten()
			out.data = new_signal
			out.samplerate = samplerate
			return out

	def envelope(self, envelope=None, times=None, kind='gain'):
		'''
		If envelope is None, returns the Hilbert envelope of a signal as new Signal.
		If envelope is a list or numpy array, returns a new object of the same type,
		with the signal data multiplied by the envelope. The envelope is linearely
		interpolated to the same length as the signal. The kind parameter ('gain' or
		'dB') determines the unit of the envelope values. If time points (in seconds,
		clamped to the the signal duration) for the amplitude values in envelope are
		supplied, then the interpolation is piecewise linear between pairs of time
		and envelope valued (must have same length).
		Example:
		> sig = Sound.tone()
		> sig.envelope(envelope=[0, 1, 0.2, 0.2, 0])
		> sig.waveform()
		'''
		if envelope is None: # no envelope supplied, compute Hilbert envelope
			if not have_scipy:
				raise ImportError('Calculating envelopes requires scipy.signal.')
			envs = numpy.abs(scipy.signal.hilbert(self.data, axis=0))
			filt = scipy.signal.firwin(1000, 50, pass_zero=True, fs=self.samplerate) # 50Hz lowpass filter to remove fine-structure
			envs = scipy.signal.filtfilt(filt, [1], envs, axis=0)
			envs[envs<=0] = numpy.finfo(float).eps # remove negative values and zeroes
			if kind == 'dB':
				envs = 20 * numpy.log10(envs) # convert amplitude to dB
			new = Signal(envs, samplerate=self.samplerate)
		else: # envelope supplied, generate new signal
			new = copy.deepcopy(self)
			if times is None:
				times = numpy.linspace(0, 1, len(envelope)) * self.duration
			else: # times vector was supplied
				if len(times) != len(envelope):
					raise ValueError('Envelope and times need to be of equal length!')
				times = numpy.array(times)
				times[times > self.duration] = self.duration # clamp between 0 and sound duration
				times[times < 0] = 0
			envelope = numpy.interp(self.times, times, numpy.array(envelope)) # get an envelope value for each sample time
			if kind == 'dB':
				envelope = 10**(envelope/20.) # convert dB to gain factors
			new.data *= envelope[:,numpy.newaxis] # multiply
		return new

	def delay(self, duration=1, chan=0, filter_length=2048):
		'''
		Delays one channel (chan) by duration (in seconds if float, or samples if int).
		If duration is a vector with self.nsamples entries, then each sample is delayed
		by the corresponsding number of seconds. This option is used by the itd_ramp
		method of the Binaural class.
		filter_length determines the accuracy of the reconstruction when using fractional
		sample delays and is 2048, or the signal length for shorter signals.
		'''
		if chan >= self.nchannels:
			raise ValueError('Channel must be smaller than number of channels in signal!')
		if filter_length%2:
			raise ValueError('Filter_length must be even!')
		if self.nsamples < filter_length: # reduce the filter_length to the signal length of short signals
			filter_length = self.nsamples-1 if self.nsamples%2 else self.nsamples # make even
		centre_tap = int(filter_length / 2)
		t = numpy.array(range(filter_length))
		if isinstance(duration, (int, float, numpy.int64, numpy.float64)): # just a constant delay
			duration = Signal.in_samples(duration, self.samplerate)
			x = t-duration
			window = 0.54 - 0.46 * numpy.cos(2 * numpy.pi * (x+0.5) / filter_length) # Hamming window
			if numpy.abs(duration) < 1e-10:
				tap_weight = numpy.zeros_like(t)
				tap_weight[centre_tap] = 1
			else:
				tap_weight = window * numpy.sinc(x-centre_tap)
			self.data[:, chan] = numpy.convolve(self.data[:, chan], tap_weight, mode='same')
		else: # dynamic delay
			if len(duration) != self.nsamples:
				ValueError('Duration shorter or longer than signal!')
			duration *= self.samplerate # assuming vector in seconds, convert to samples
			padding = numpy.zeros(centre_tap)
			sig = numpy.concatenate((padding, self.channel(chan), padding), axis=None) # for zero-padded convolution (potential edge artifacts!)
			for i, current_delay in enumerate(duration):
				x = t-current_delay
				window = 0.54 - 0.46 * numpy.cos(2 * numpy.pi * (x+0.5) / filter_length) # Hamming window
				if numpy.abs(current_delay) < 1e-10:
					tap_weight = numpy.zeros_like(t)
					tap_weight[centre_tap] = 1
				else:
					tap_weight = window * numpy.sinc(x-centre_tap)
					sig_portion = sig[i:i+filter_length]
					self.data[i, chan] = numpy.convolve(sig_portion, tap_weight, mode='valid') # sig_portion and tap_weight have the same length, so the valid part of the convolution is just one sample, which gets written into the signal at the current index
