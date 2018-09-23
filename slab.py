'''
slab exports classes for handling signals like sounds, filters and HRTFs.
This module uses doctests. Use like so:
python -m doctest slab.py
'''

# Signal
import copy
import numpy

# Sound
import array
import time
try:
	import soundfile
	have_soundfile = True
except ImportError:
	have_soundfile = False
try:
	import soundcard
	have_soundcard = True
except ImportError:
	have_soundcard = False
import platform
if platform.system() == 'Windows':
	import winsound
else:
	import subprocess
try:
	import scipy.signal
	have_scipy = True
except ImportError:
	have_scipy = False
try:
	import matplotlib.pyplot as plt
	have_pyplot = True
except ImportError:
	have_pyplot = False

# HRTF
try:
	import h5py
	import h5netcdf
	have_h5 = True
except ImportError:
	have_h5 = False
import warnings
import pathlib

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
	<class 'slab.Signal'> duration 1.0, samples 10, channels 2, samplerate 10

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
		<class 'slab.Signal'> duration 1.0, samples 10, channels 1, samplerate 10

	sig.resize(L)
		Extends or contracts the length of the data in the object in place to have L samples.
		>>> sig.resize(8)
		>>> print(sig)
		<class 'slab.Signal'> duration 0.8, samples 8, channels 2, samplerate 10
	'''

	# instance properties
	nsamples = property(fget=lambda self:self.data.shape[0],
						doc='The number of samples in the Signal.')
	duration = property(fget=lambda self:self.data.shape[0] / self.samplerate,
						doc='The length of the Signal in seconds.')
	times = property(fget=lambda self:numpy.arange(self.data.shape[0], dtype=float) / self.samplerate,
						doc='An array of times (in seconds) corresponding to each sample.')
	nchannels = property(fget=lambda self:self.data.shape[1],
						doc='The number of channels in the Signal.')

	# __methods (class creation, printing, and slice functionality)
	def __init__(self,data,samplerate=None):
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
		else:
			raise TypeError('Cannot initialise Signal with data of class ' + str(data.__class__))
		if len(self.data.shape)==1:
			self.data.shape = (len(self.data), 1)
		elif self.data.shape[1] > self.data.shape[0]:
			self.data = self.data.T

	def __repr__(self):
		return f'{type(self)} (\n{repr(self.data)}\n{repr(self.samplerate)})'

	def __str__(self):
		return f'{type(self)} duration {self.duration}, samples {self.nsamples}, channels {self.nchannels}, samplerate {self.samplerate}'

	def __getitem__(self,key):
		return self.data.__getitem__(key)

	def __setitem__(self,key,value):
		return self.data.__setitem__(key,value)

	# arithmatic operators
	def __add__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other,type(self)):
			new.data = self.data + other.data
		else:
			new.data = self.data + other
		return new
	__radd__ = __add__

	def __sub__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other,type(self)):
			new.data = self.data - other.data
		else:
			new.data = self.data - other
		return new
	__rsub__ = __sub__

	def __mul__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other,type(self)):
			new.data = self.data * other.data
		else:
			new.data = self.data * other
		return new
	__rmul__ = __mul__

	def __truediv__(self, other):
		new = copy.deepcopy(self)
		if isinstance(other,type(self)):
			new.data = self.data / other.data
		else:
			new.data = self.data / other
		return new
	__rtruediv__ = __truediv__

	def __neg__(self):
		new = copy.deepcopy(self)
		new.data = self.data*-1
		return new

	# static methods (belong to the class, but can be called without creating an instance)
	@staticmethod
	def in_samples(ctime,samplerate):
		'''Converts time values in seconds to samples.
		This is used to enable input in either samples (int) or seconds (float) in the class.'''
		if not isinstance(ctime, int):
			ctime = int(numpy.rint(ctime * samplerate))
		return ctime

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

	def resize(self, L):
		'Extends or contracts the length of the data in the object in place to have L samples.'
		if isinstance(L,float): # support L in samples or seconds
			L = int(numpy.rint(L*self.samplerate))
		if L == len(self.data):
			pass # already correct size
		elif L < len(self.data):
			self.data = self.data[:L, :]
		else:
			padding = numpy.zeros((L - len(self.data), self.nchannels))
			self.data = numpy.concatenate((self.data, padding))


class Sound(Signal):
	# TODO: debug dynamicripple, add ability to get output of different stages of an auditory periphery model from a sound,
	# add other own stim functions (transitions, babbelnoise)? Add reverb (image model) room simulation.
	# add auditory spectrogram (filterbank) -> gammatone filterbank followed by halfwave rectification, cube root compression and 10 Hz low pass filtering
	'''
	Class for working with sounds, including loading/saving, manipulating and playing.
	Examples:
	>>> import slab
	>>> import numpy
	>>> print(slab.Sound(numpy.ones([10,2]),samplerate=10))
	<class 'slab.Sound'> duration 1.0, samples 10, channels 2, samplerate 10
	>>> print(slab.Sound(numpy.ones([10,2]),samplerate=10).left)
	<class 'slab.Sound'> duration 1.0, samples 10, channels 1, samplerate 10

	** Properties**

	Sound.left: left (0th)channel
	Sound.right: right (1st) channel

	**Reading, writing and playing**
	Sound.write(sound, filename) or writesound(sound, filename):
		Write a sound object to a wav file.
		Example:
		>>> sig = slab.Sound.tone(500, 8000, samplerate=8000)
		>>> sig.write('tone.wav')

	Sound.read(filename) or readsound(filename):
		Load the file given by filename and returns a Sound object.
		Sound file can be either a .wav or a .aif file.
		Example:
		>>> sig2 = slab.Sound('tone.wav')
		>>> print(sig2)
		<class 'slab.Sound'> duration 1.0, samples 8000, channels 1, samplerate 8000

	Sound.play(*sounds, **kwargs):
	Plays a sound or sequence of sounds. For example::
		>>> sig.play(sleep=True)

	If ``sleep=True`` the function will wait
	until the sounds have finished playing before returning.

	**Generating sounds**
	All sound generating methods can be used with durations arguments in samples (int) or seconds (float).
	One can also set the number of channels by setting the keyword argument nchannels to the desired value.
	See doc string in respective function.
	- tone(frequency, duration, phase=0, samplerate=None, nchannels=1)
	- harmoniccomplex(f0, duration, amplitude=1, phase=0, samplerate=None, nchannels=1)
	- whitenoise(duration, samplerate=None, nchannels=1)
	- powerlawnoise(duration, alpha, samplerate=None, nchannels=1,normalise=False)
	Pinknoise and brownnoise are wrappers of powerlawnoise with different exponents.
	- click(duration, peak=None, samplerate=None, nchannels=1)
	- clicktrain(duration, freq, peak=None, samplerate=None, nchannels=1)
	- silence(duration, samplerate=None, nchannels=1)
	- vowel(vowel='a', gender=''male'', duration=1., samplerate=None, nchannels=1)
	- irn(delay, gain, niter, duration, samplerate=None, nchannels=1)
	- dynamicripple(Am=0.9, Rt=6, Om=2, Ph=0, duration=1., f0=1000, samplerate=None, BW=5.8, RO=0, df=1/16, ph_c=None)

	**Timing and sequencing**
	- sequence(*sounds, samplerate=None)
	- sig.repeat(n)
	- sig.resample(samplerate)
	- sig.ramp(when='both', duration=0.01, envelope=None, inplace=True)

	**Plotting**
	Examples:
		>>> vowel = slab.Sound.vowel(vowel='a', duration=.5, samplerate=8000)
		>>> vowel.ramp()
		>>> pxx, freqs, bins, im = vowel.spectrogram(low=100, high=4000, log_power=True)
		>>> Z, freqs, phase = vowel.spectrum(low=100, high=4000, log_power=True)
		>>> vowel.waveform(start=0, end=.1)
	'''
	# instance properties
	left = property(fget=lambda self: self.channel(0),
					doc='The left channel for a stereo sound.')
	right = property(fget=lambda self: self.channel(1),
					 doc='The right channel for a stereo sound.')

	def _get_level(self):
		'''
		Returns level in dB SPL (RMS) assuming array is in Pascals.
		In the case of multi-channel sounds, returns an array of levels
		for each channel, otherwise returns a float.
		'''
		if self.nchannels == 1:
			rms_value = numpy.sqrt(numpy.mean(numpy.square(self.data-numpy.mean(self.data))))
			rms_dB = 20.0*numpy.log10(rms_value/2e-5)
			return rms_dB
		else:
			chans = (self.channel(i) for i in range(self.nchannels))
			levels = (c.level for c in chans)
			return numpy.array(tuple(l for l in levels))

	def _set_level(self, level):
		'''
		Sets level in dB SPL (RMS) assuming array is in Pascals. ``level``
		should be a value in dB, or a tuple of levels, one for each channel.
		'''
		rms_dB = self._get_level()
		if self.nchannels>1:
			level = numpy.array(level)
			if level.size==1:
				level = level.repeat(self.nchannels)
			level = numpy.reshape(level, (1, self.nchannels))
			rms_dB = numpy.reshape(rms_dB, (1, self.nchannels))
		gain = 10**((level-rms_dB)/20.)
		self.data *= gain

	level = property(fget=_get_level, fset=_set_level, doc='''
		Can be used to get or set the level of a sound, which should be in dB.
		For single channel sounds a value in dB is used, for multiple channel
		sounds a value in dB can be used for setting the level (all channels
		will be set to the same level), or a list/tuple/array of levels. It
		is assumed that the unit of the sound is Pascals.
		''')

	def __init__(self,data,samplerate=None):
		if isinstance(data, str): # additional options for Sound initialization (from a file)
			if samplerate is not None:
				raise ValueError('Cannot specify samplerate when initialising Sound from a file.')
			_ = Sound.read(data)
			self.data = _.data
			self.samplerate = _.samplerate
		else:
			# delegate to the baseclass init
			super().__init__(data,samplerate)

	# static methods (creating sounds)
	@staticmethod
	def read(filename):
		'''
		Load the file given by filename and returns a Sound object.
		Sound file can be either a .wav or a .aif file.
		'''
		ext = filename.split('.')[-1].lower()
		if ext=='wav':
			import wave as sndmodule
		elif ext=='aif' or ext=='aiff':
			import aifc as sndmodule
		else:
			raise NotImplementedError('Can only load aif or wav soundfiles')
		wav = sndmodule.open(filename, "r")
		nchannels, _, framerate, nframes, _, _ = wav.getparams()
		frames = wav.readframes(nframes * nchannels)
		out = numpy.frombuffer(frames, dtype=numpy.dtype('h'))
		data = numpy.zeros((nframes, nchannels))
		for i in range(nchannels):
			data[:, i] = out[i::nchannels]
			data[:, i] /= 2**15
		return Sound(data, samplerate=framerate)

	@staticmethod
	def tone(frequency=500, duration=1., phase=0, samplerate=None, nchannels=1):
		'''
		Returns a pure tone at frequency for duration, using the default
		samplerate or the given one. The ``frequency`` and ``phase`` parameters
		can be single values, in which case multiple channels can be
		specified with the ``nchannels`` argument, or they can be sequences
		(lists/tuples/arrays) in which case there is one frequency or phase for
		each channel.
		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration,samplerate)
		frequency = numpy.array(frequency)
		phase = numpy.array(phase)
		if frequency.size>nchannels and nchannels==1:
			nchannels = frequency.size
		if phase.size>nchannels and nchannels==1:
			nchannels = phase.size
		if frequency.size==nchannels:
			frequency.shape = (1, nchannels)
		if phase.size==nchannels:
			phase.shape =(nchannels, 1)
		t = numpy.arange(0, duration, 1)/samplerate
		t.shape = (t.size, 1) # ensures C-order (in contrast to tile(...).T )
		x = numpy.sin(phase + 2*numpy.pi * frequency * numpy.tile(t, (1, nchannels)))
		return Sound(x, samplerate)

	@staticmethod
	def harmoniccomplex(f0=500, duration=1., amplitude=1, phase=0, samplerate=None, nchannels=1):
		'''
		Returns a harmonic complex composed of pure tones at integer multiples
		of the fundamental frequency ``f0``.
		The ``amplitude`` and ``phase`` keywords can be set to either a single
		value or an array of values. In the former case the value is set for all
		harmonics, and harmonics up to the sampling frequency are
		generated. In the latter each harmonic parameter is set
		separately, and the number of harmonics generated corresponds
		to the length of the array.
		Example:
		>>> sig = Sound.harmoniccomplex(f0=200, amplitude=[80,70,60,50])
		>>> _ = sig.spectrum()

		'''
		samplerate = Sound.get_samplerate(samplerate)
		phases = numpy.array(phase).flatten()
		amplitudes = numpy.array(amplitude).flatten()
		if len(phases)>1 or len(amplitudes)>1:
			if (len(phases)>1 and len(amplitudes)>1) and (len(phases) != len(amplitudes)):
				raise ValueError('Please specify the same number of phases and amplitudes')
			Nharmonics = max(len(phases),len(amplitudes))
		else:
			Nharmonics = int(numpy.floor( samplerate/(2*f0) ) )
		if len(phases) == 1:
			phases = numpy.tile(phase, Nharmonics)
		if len(amplitudes) == 1:
			amplitudes = numpy.tile(amplitude, Nharmonics)
		out = Sound.tone(f0, duration, phase = phases[0], samplerate = samplerate, nchannels = nchannels)
		out.level = amplitudes[0]
		for i in range(1,Nharmonics):
			tmp = Sound.tone(frequency=(i+1)*f0, duration=duration, phase=phases[i], samplerate=samplerate, nchannels=nchannels)
			tmp.level = amplitudes[i]
			out += tmp
		return out

	@staticmethod
	def whitenoise(duration=1.0, samplerate=None, nchannels=1, normalise=True):
		'''
		Returns a white noise. If the samplerate is not specified, the global
		default value will be used. nchannels = 2 produces uncorrelated noise (dichotic).
		>>> noise = Sound.whitenoise(1.0,nchannels=2)

		To make a diotic noise:
		>>> noise.data[:,1] = noise.data[:,0]

		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration,samplerate)
		x = numpy.random.randn(duration,nchannels)
		if normalise:
			for i in range(nchannels):
				x[:, i] = ((x[:, i] - numpy.amin(x[:, i]))/(numpy.amax(x[:, i]) - numpy.amin(x[:, i])) - 0.5) * 2
		return Sound(x, samplerate)

	@staticmethod
	def powerlawnoise(duration=1.0, alpha=1, samplerate=None, nchannels=1, normalise=True):
		'''
		Returns a power-law noise for the given duration. Spectral density per unit of bandwidth scales as 1/(f**alpha).
		Example:
		>>> noise = Sound.powerlawnoise(0.2, 1, samplerate=8000)
		
		Arguments:
		``duration`` : Duration of the desired output.
		``alpha`` : Power law exponent.
		``samplerate`` : Desired output samplerate
		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		n = duration
		n2 = int(n/2)
		f = numpy.array(numpy.fft.fftfreq(n, d=1.0/samplerate), dtype=complex)
		f.shape = (len(f), 1)
		f = numpy.tile(f, (1, nchannels))
		if n%2 == 1:
			z = (numpy.random.randn(n2, nchannels) + 1j * numpy.random.randn(n2, nchannels))
			a2 = 1.0 / (f[1:(n2+1), :]**(alpha/2.0))
		else:
			z = (numpy.random.randn(n2-1, nchannels) + 1j * numpy.random.randn(n2-1, nchannels))
			a2 = 1.0 / (f[1:n2, :]**(alpha/2.0))
		a2 *= z
		if n%2 == 1:
			d = numpy.vstack((numpy.ones((1, nchannels)), a2,
					  numpy.flipud(numpy.conj(a2))))
		else:
			d = numpy.vstack((numpy.ones((1, nchannels)), a2,
					  1.0 / (numpy.abs(f[n2])**(alpha/2.0))*
					  numpy.random.randn(1, nchannels),
					  numpy.flipud(numpy.conj(a2))))
		x = numpy.real(numpy.fft.ifft(d.flatten()))
		x.shape = (n, nchannels)
		if normalise:
			for i in range(nchannels):
				x[:, i] = ((x[:, i] - numpy.amin(x[:, i]))/(numpy.amax(x[:, i]) - numpy.amin(x[:, i])) - 0.5) * 2
		return Sound(x, samplerate)

	@staticmethod
	def pinknoise(duration=1.0, samplerate=None, nchannels=1, normalise=True):
		'Returns pink noise, i.e :func:`powerlawnoise` with alpha=1'
		return Sound.powerlawnoise(duration, 1.0, samplerate=samplerate,
								   nchannels = nchannels, normalise=normalise)

	@staticmethod
	def brownnoise(duration=1.0, samplerate=None, nchannels=1, normalise=True):
		'Returns brown noise, i.e :func:`powerlawnoise` with alpha=2'
		return Sound.powerlawnoise(duration, 2.0, samplerate=samplerate,
								   nchannels = nchannels, normalise=normalise)

	@staticmethod
	def irn(delay=0.01, gain=1, niter=16, duration=1.0, samplerate=None):
		'''
		Returns an iterated ripple noise. The noise is obtained many attenuated and
		delayed version of the original broadband noise.
		'''
		samplerate = Sound.get_samplerate(samplerate)
		delay = Sound.in_samples(delay, samplerate)
		noise = Sound.whitenoise(duration, samplerate=samplerate)
		x = numpy.array(noise.data.T)[0]
		irn_add = numpy.fft.fft(x)
		n_spl, spl_dur = len(irn_add), float(1/samplerate)
		w = 2 * numpy.pi*numpy.fft.fftfreq(n_spl, spl_dur)
		d = float(delay)
		for k in range(1, niter+1):
			irn_add += (gain**k) * irn_add * numpy.exp(-1j * w * k * d)
		irn_add = numpy.fft.ifft(irn_add)
		x = numpy.real(irn_add)
		return Sound(x, samplerate)

	@staticmethod
	def click(duration=0.0001, peak=None, samplerate=None, nchannels=1):
		'''
		Returns a click of the given duration (*100 microsec*).
		If ``peak`` is not specified, the amplitude will be 1, otherwise
		``peak`` refers to the peak dB SPL of the click, according to the
		formula ``28e-6*10**(peak/20.)``.
		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		if peak is not None:
			amplitude = 28e-6*10**(float(peak)/20)
		else:
			amplitude = 1
		x = amplitude*numpy.ones((duration, nchannels))
		return Sound(x, samplerate)

	@staticmethod
	def clicktrain(duration=1.0, freq=500, clickduration=1, peak=None, samplerate=None):
		'Returns a series of n clicks (see :func:`click`) at a frequency of freq (*500 Hz*).'
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		clickduration = Sound.in_samples(clickduration, samplerate)
		interval = int(numpy.rint(1/freq * samplerate))
		n = numpy.rint(duration/interval)
		oneclick = Sound.click(clickduration, peak=peak, samplerate=samplerate)
		oneclick.resize(interval)
		oneclick.repeat(n)
		return oneclick

	@staticmethod
	def silence(duration=1.0, samplerate=None, nchannels=1):
		'Returns a silent, zero sound for the given duration.'
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		return Sound(numpy.zeros((duration, nchannels)), samplerate)

	@staticmethod
	def vowel(vowel='a', gender='male', duration=1., samplerate=None, nchannels=1):
		'''
		Returns a vowel sound.
		vowel: 'a', 'e', 'i', 'o', 'u', 'ae', 'oe', or 'ue' (pre-set format frequencies)
		or 'none' for random formants in the range of the vowel formants.
		gender: 'male', 'female'
		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		if vowel == 'a':
			formants = (0.73, 1.09, 2.44)
		elif vowel == 'e':
			formants = (0.36, 2.25, 3.0)
		elif vowel == 'i':
			formants = (0.27, 2.29, 3.01)
		elif vowel == 'o':
			formants = (0.35, 0.5, 2.6)
		elif vowel == 'u':
			formants = (0.3, 0.87, 2.24)
		elif vowel == 'ae':
			formants = (0.86, 2.05, 2.85)
		elif vowel == 'oe':
			formants = (0.4, 1.66, 1.96)
		elif vowel == 'ue':
			formants = (0.25, 1.67, 2.05)
		elif vowel == 'none':
			BW = 0.3
			formants = (0.22/(1-BW)+(0.86/(1+BW)-0.22/(1-BW))*numpy.random.rand(),
						0.87/(1-BW)+(2.25/(1+BW)-0.87/(1-BW))*numpy.random.rand(),
						1.96/(1-BW)+(3.30/(1+BW)-1.96/(1-BW))*numpy.random.rand())
		else:
			raise ValueError('Unknown vowel: "%s"' % (vowel))
		ST = 1000/samplerate
		times = ST * numpy.arange(duration)
		T05 = 2.5
		if gender == 'male':
			t_rep = 12
		elif gender == 'female':
			t_rep = 8
		env = numpy.exp(-numpy.log(2)/T05 * numpy.mod(times, t_rep))
		env = numpy.mod(times, t_rep)**0.25 * env
		min_env = numpy.min(env[(times >= t_rep/2) & (times <= t_rep-ST)])
		env = numpy.maximum(env, min_env)
		vowel = numpy.zeros(len(times))
		for f in formants:
			A = numpy.min((0, -6*numpy.log2(f)))
			vowel = vowel + 10**(A/20) * env * numpy.sin(2 * numpy.pi * f * numpy.mod(times, t_rep))
		if nchannels > 1:
			vowel = numpy.tile(vowel, (nchannels, 1))
		vowel = Sound(data=vowel, samplerate=samplerate)
		vowel.filter(f=0.75*samplerate/2, type='lp')
		return vowel

	@staticmethod
	def erb_noise(samplerate=None, duration=1.0, f_lower=125, f_upper=4000):
		'''
		Returns an equally-masking noise (ERB noise) in the band between
		f_lower and f_upper.
		Example:
		>>> sig = Sound.erb_noise()
		>>> sig.ramp()
		>>> _ = sig.spectrum()
		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		n = 2**(duration-1).bit_length() # next power of 2
		st = 1 / samplerate
		df = 1 / (st * n)
		frq = df * numpy.arange(n/2)
		frq[0] = 1 # avoid DC = 0
		lev = -10*numpy.log10(24.7*(4.37*frq))
		filt = 10.**(lev/20)
		noise = numpy.random.randn(n)
		noise = numpy.real(numpy.fft.ifft(numpy.concatenate((filt, filt[::-1])) * numpy.fft.fft(noise)))
		noise = noise/numpy.sqrt(numpy.mean(noise**2))
		band = numpy.zeros(len(lev))
		band[round(f_lower/df):round(f_upper/df)] = 1
		fnoise = numpy.real(numpy.fft.ifft(numpy.concatenate((band, band[::-1])) * numpy.fft.fft(noise)))
		fnoise = fnoise[:duration]
		return Sound(data=fnoise, samplerate=samplerate)

	@staticmethod
	def dynamicripple(Am=0.9, Rt=6, Om=2, Ph=0, duration=1., f0=1000, samplerate=None, BW=5.8, RO=0, df=1/16, ph_c=None):
		'''
		Return a moving ripple stimulus
		s = mvripfft(para)
		[s, ph_c, fdx] = mvripfft(para, cond, ph_c)
		para = [Am, Rt, Om, Ph]
			Am: modulation depth, 0 < Am < 1, DEFAULT = .9;
			Rt: rate (Hz), integer preferred, typically, 1 .. 128, DEFAULT = 6;
			Om: scale (cyc/oct), any real number, typically, .25 .. 4, DEFAULT = 2;
			Ph: (optional) symmetry (Pi) at f0, -1 < Ph < 1, DEFAULT = 0.
		cond = (optional) [T0, f0, SF, BW, RO, df]
			T0: duartion (sec), DEFAULT = 1.
			f0: center freq. (Hz), DEFAULT = 1000.
			samplerate: sample freq. (Hz), must be power of 2, DEFAULT = 16384
			BW: excitation band width (oct), DEFAULT = 5.8.
			RO: roll-off (dB/oct), 0 means log-spacing, DEFAULT = 0;
			df: freq. spacing, in oct (RO=0) or in Hz (RO>0), DEFAULT = 1/16.
			ph_c: component phase
		Converted to python by Jessica Thompson based on Jonathan Simon and Didier Dipereux's
		matlab program [ripfft.m], based on Jian Lin's C program [rip.c].
		Example:
		>>> ripple = Sound.dynamicripple()

		'''
		samplerate = Sound.get_samplerate(samplerate)
		#duration = Sound.in_samples(duration, samplerate) # TODO: duration has to be in seconds for this function!
		dur = int(numpy.ceil(duration))    # duration rounded up for generating purpose
		Ri = Rt*dur # modulation lag, in number of df's
		if RO:  #compute tones freqs
			R1 = numpy.round(2**(numpy.array(-1, 1)*BW/2)*f0/df)
			fr = df*(numpy.arange(R1[1], R1[2]))
		else:   #compute log-spaced tones freqs
			R1 = numpy.round(BW/2/df)
			fr = f0*2**(numpy.arange(-R1, R1+1)*df)
		M = len(fr) # of component
		S = 0j*numpy.zeros((int(dur*samplerate/2), 1)) # memory allocation
		fdx = numpy.round(fr*dur)+1 # freq. index MATLAB INDEXING??? -> not an index!!!
		x = numpy.log2(fr/f0) # tono. axis (oct)
		# roll-off and phase relation
		r = 10**(-x*RO/20) # roll-off, -x*RO = 20log10(r)
		if ph_c is not None:
			th = ph_c
		else:
			th = 2*numpy.pi*numpy.random.rand(M, 1) # component phase, theta
		ph_c = th
		ph = (2*Om*x+Ph)*numpy.pi # ripple phase, phi, modulation phase
		# modulation
		maxidx = numpy.zeros((M, 2))
		maxidx[:, 0] = fdx-Ri
		S[numpy.max(maxidx,axis=1).astype(int)] = S[numpy.max(maxidx,axis=1).astype(int)]+numpy.dot(r,numpy.exp(1j*(th-ph))) # lower side
		minidx = numpy.ones((M,2))
		minidx[:,0] = fdx+Ri
		minidx[:,1] = minidx[:,1] * dur*samplerate/2
		S[numpy.min(minidx, axis=1).astype(int)] = S[numpy.min(minidx, axis=1).astype(int)] + numpy.dot(r,numpy.exp(1j*(th+ph)))    # upper side
		S = S * Am/2
		S[0] = 0
		S = S[:dur*samplerate/2]
		# original stationary spectrum
		S[fdx.astype(int)] = S[fdx.astype(int)] + r*numpy.exp(1j*th) # moved here to save computation
		# time waveform
		s = numpy.fft.ifft(numpy.concatenate((S, [0], numpy.flipud(S[1:dur*samplerate/2]).conj()))) # make it double side
		s = s[:int(numpy.round(duration*samplerate))].real # only real part is good. *2 was ignored
		return Sound(s, samplerate)

	@staticmethod
	def sequence(*sounds):
		'Returns the sequence of sounds in the list sounds joined together'
		samplerate = sounds[0].samplerate
		for sound in sounds:
			if sound.samplerate != samplerate:
				raise ValueError('All sounds must have the same sample rate.')
		sounds = tuple(s.data for s in sounds)
		x = numpy.vstack(sounds)
		return Sound(x, samplerate)

	# instance methods
	def write(self, filename, normalise=False):
		'''
		Save the sound as a WAV.
		If the normalise keyword is set to True, the amplitude of the sound will be
		normalised to 1.
		'''
		ext = filename.split('.')[-1].lower()
		if ext == 'wav':
			import wave as sndmodule
		else:
			raise NotImplementedError('Can only save as wav soundfiles')
		w = sndmodule.open(filename, 'wb')
		w.setnchannels(self.nchannels)
		w.setsampwidth(2)
		w.setframerate(int(self.samplerate))
		x = numpy.array(self.data, copy=True)
		am = numpy.amax(x)
		z = numpy.zeros(x.shape[0]*self.nchannels, dtype='int16')
		x.shape=(x.shape[0], self.nchannels)
		for i in range(self.nchannels):
			if normalise:
				x[:, i] /= am
			x[:, i] = (x[:, i]) * 2 ** 15
			z[i::self.nchannels] = x[::1, i]
		data = numpy.array(z, dtype='int16')
		data = array.array('h', data)
		w.writeframes(data.tobytes())
		w.close()

	def ramp(self, when='both', duration=0.01, envelope=None):
		'''
		Adds a ramp on/off to the sound (in place)

		``when='onset'``
			Can take values 'onset', 'offset' or 'both'
		``duration=0.01``
			The time over which the ramping happens (in samples or seconds)
		``envelope``
			A ramping function, if not specified uses ``sin(pi*t/2)**2``. The
			function should be a function of one variable ``t`` ranging from
			0 to 1, and should increase from ``f(0)=0`` to ``f(0)=1``. The
			reverse is applied for the offset ramp.
		'''
		when = when.lower().strip()
		if envelope is None:
			envelope = lambda t: numpy.sin(numpy.pi * t / 2) ** 2
		sz = Sound.in_samples(duration, self.samplerate)
		multiplier = envelope(numpy.reshape(numpy.linspace(0.0, 1.0, sz), (sz, 1)))
		if when == 'onset' or when == 'both':
			self.data[:sz, :] *= multiplier
		if when == 'offset' or when == 'both':
			self.data[self.nsamples-sz:, :] *= multiplier[::-1]

	def shift(self, duration, fractional=False, filter_length=2048):
		'''
		Returns the sound delayed by duration, which can be the number of
		samples or a length of time in seconds. Normally, only integer
		numbers of samples will be used, but if ``fractional=True`` then
		the filtering method from
		`http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
		will be used (introducing some small numerical errors). With this
		method, you can specify the ``filter_length``, larger values are
		slower but more accurate, especially at higher frequencies. The large
		default value of 2048 samples provides good accuracy for sounds with
		frequencies above 20 Hz, but not for lower frequency sounds. If you are
		restricted to high frequency sounds, a smaller value will be more
		efficient. Note that if ``fractional=True`` then
		``duration`` is assumed to be a time not a number of samples.
		>>> sig = Sound(numpy.ones([10,1]),samplerate=10)
		>>> _ = sig.shift(1)
		>>> _ = sig.shift(-1)
		>>> _ = sig.shift(1.5,fractional=True)

		'''
		if not fractional:
			if not isinstance(duration, int):
				duration = int(numpy.rint(duration*self.samplerate))
			if duration >= 0:
				y = numpy.vstack((numpy.zeros((duration, self.nchannels)), self.data))
				return Sound(y, samplerate=self.samplerate)
			else:
				return Sound(self.data[-duration:, :], samplerate=self.samplerate)
		else:
			if self.nchannels > 1:
				sounds = [self.channel(i).shifted(duration, fractional=True, filter_length=filter_length) for i in range(self.nchannels)]
				return Sound(numpy.hstack(sounds), samplerate=self.samplerate)
			# Adapted from http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
			delay = duration*self.samplerate
			if delay >= 0:
				idelay = int(delay)
			elif delay < 0:
				idelay = -int(-delay)
			delay -= idelay
			centre_tap = filter_length // 2
			t = numpy.arange(filter_length)
			x = t-delay
			if numpy.abs(numpy.round(delay)-delay) < 1e-10:
				tap_weight = numpy.array(x == centre_tap, dtype=float)
			else:
				sinc = numpy.sin(numpy.pi*(x-centre_tap))/(numpy.pi*(x-centre_tap))
				window = 0.54-0.46*numpy.cos(2.0*numpy.pi*(x+0.5)/filter_length) # Hamming window
				tap_weight = window*sinc
			if filter_length < 256 or not have_scipy:
				y = numpy.convolve(tap_weight, self.data.flatten())
			else:
				y = scipy.signal.fftconvolve(tap_weight, self.data.flatten())
			y = y[filter_length//2:-filter_length//2]
			sound = Sound(y, self.samplerate)
			sound = sound.shift(idelay)
			return sound

	def repeat(self, n):
		'Repeats the sound n times'
		self.data = numpy.vstack((self.data,)*int(n))

	@staticmethod
	def crossfade(sound1, sound2, overlap=0.01):
		# TODO: take any number of sounds
		'''
		Return a new sound that is a crossfade of sound1 and sound2.
		Overlap by overlap samples (if int) or seconds (if float, *0.01*).
		Example:
		>>> noise = Sound.whitenoise(duration=1.0)
		>>> vowel = Sound.vowel()
		>>> noise2vowel = Sound.crossfade(noise,vowel,overlap=0.4)
		>>> noise2vowel.play()
		'''
		if sound1.nchannels != sound2.nchannels:
			raise ValueError('Cannot crossfade sounds with unequal numbers of channels.')
		if sound1.samplerate != sound2.samplerate:
			raise ValueError('Cannot crossfade sounds with unequal samplerates.')
		overlap = Sound.in_samples(overlap, samplerate=sound1.samplerate)
		n_total = sound1.nsamples + sound2.nsamples - overlap
		silence = Sound.silence(sound1.nsamples - overlap, samplerate=sound1.samplerate, nchannels=sound1.nchannels)
		sound1.ramp(duration=overlap, when='offset')
		sound1.resize(n_total) # extend sound1 to total length
		sound2.ramp(duration=overlap, when='onset')
		sound2 = Sound.sequence(silence, sound2) # sound2 has to be prepended with silence
		return sound1 + sound2

	def pulse(self, pulse_freq=4, duty=0.75):
		'''
		Apply a pulse envelope to the sound with a pulse frequency (in Hz, *4*) and duty cycle (*3/4*).
		'''
		pulse_period = 1/pulse_freq
		n_pulses = round(self.duration / pulse_period) # number of pulses in the stimulus
		pulse_period = self.duration / n_pulses # period in s, fits into stimulus duration
		pulse_samples = Sound.in_samples(pulse_period * duty, self.samplerate) # duty cycle in s
		fall_samples = Sound.in_samples(5/1000, self.samplerate) # 5ms rise/fall time
		fall = numpy.cos(numpy.pi * numpy.arange(fall_samples) / (2 * (fall_samples)))**2
		pulse = numpy.concatenate((1-fall, numpy.ones(pulse_samples - 2 * fall_samples), fall))
		pulse = numpy.concatenate((pulse, numpy.zeros(Sound.in_samples(pulse_period, self.samplerate)-len(pulse))))
		envelope = numpy.tile(pulse, n_pulses)
		envelope = envelope[:,None] # add an empty axis to get to the same shape as self.data: (n_samples, 1)
		self.data *= numpy.broadcast_to(envelope, self.data.shape) # if data is 2D (>1 channel) broadcase the envelope to fit

	def resample(self, samplerate):
		'Returns a resampled version of the sound.'
		if not have_scipy:
			raise ImportError('Resampling requires scipy.')
		y = numpy.array(scipy.signal.resample(self, int(numpy.rint(samplerate*self.duration))), dtype='float64')
		return Sound(y, samplerate=samplerate)

	def filter(self, f=1, type='hp'):
		'''
		Filters the sound in place.
		f: edge frequency in Hz (*1*) or tuple of frequencies for bp and notch.
		type: 'lp', *'hp'*, bp, 'notch'
		TODO: For costum filter shapes f and type are tuples with frequencies
		in Hz and corresponding attenuations in dB. If f is a numpy array it is
		taken as the target magnitude of the spectrum (imposing one sound's
		spectrum on the current sound).
		Examples:
		>>> sig = Sound.whitenoise()
		>>> sig.filter(f=3000, type='lp')
		>>> _ = sig.spectrum()
		'''
		#n = 2**(self.nsamples-1).bit_length() # next power of 2
		n = self.nsamples
		n_unique_pts = int(numpy.ceil((n+1)/2))
		st = 1 / self.samplerate
		df = 1 / (st * n)
		filt = numpy.zeros(n_unique_pts-1)
		if type == 'lp':
			filt[:round(f/df)] = 1
		if type == 'hp':
			filt[round(f/df):] = 1
		if type == 'bp':
			filt[round(f[0]/df):round(f[1]/df)] = 1
		if type == 'notch':
			filt[:round(f[0]/df)] = 1
			filt[round(f[1]/df):] = 1
		if n % 2 == 0: # even
			filt = numpy.concatenate((filt, filt[::-1]))
		else: # odd
			filt = numpy.concatenate((filt, [0], filt[::-1]))
		for chan in range(self.nchannels):
			_fft = numpy.fft.fft(self.data[:, chan])
			self.data[:,chan] = numpy.real(numpy.fft.ifft(filt * _fft))

	def aweight(self):
		#TODO: untested! Filter all chans.
		'Returns A-weighted first channel as new sound.'
		f1 = 20.598997
		f2 = 107.65265
		f3 = 737.86223
		f4 = 12194.217
		A1000 = 1.9997
		numerators = [(2 * numpy.pi * f4)**2 * (10**(A1000 / 20)), 0, 0, 0, 0]
		denominators = numpy.convolve([1, 4 * numpy.pi * f4, (2 * numpy.pi * f4)**2], [1, 4 * numpy.pi * f1, (2 * numpy.pi * f1)**2])
		denominators = numpy.convolve(numpy.convolve(denominators, [1, 2 * numpy.pi * f3]), [1, 2 * numpy.pi * f2])
		b, a = scipy.signal.filter_design.bilinear(numerators, denominators, self.samplerate)
		data = scipy.signal.lfilter(b, a, self.data[:,0])
		return Sound(data, self.samplerate)

	@staticmethod
	def record(duration=1.0, samplerate=44100):
		'Record from inbuilt microphone. Note that most soundcards can only record at 44100 Hz samplerate.'
		if not have_soundcard:
			raise NotImplementedError('Need module soundcard for recording (https://github.com/bastibe/SoundCard).')
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		print(duration)
		mic = soundcard.default_microphone()
		data = mic.record(samplerate=samplerate, numframes=duration, channels=1)
		return Sound(data, samplerate=samplerate)

	def play(self, sleep=False):
		'Plays the sound.'
		if have_soundcard:
			soundcard.default_speaker().play(self.data, samplerate=self.samplerate)
		else:
			if self.nchannels > 2:
				raise ValueError("Can only play sounds with 1 or 2 channels.")
			wavfile = 'tmp.wav'
			self.write(wavfile, normalise=False)
			if platform.system() == 'Windows':
				winsound.PlaySound('%s.wav' % wavfile, winsound.SND_FILENAME)
			elif platform.system() == 'Darwin':
				subprocess.Popen(["afplay", wavfile])
			else:  # Linus/Unix, install sox (sudo apt-get install sox libsox-fmt-all)
				try:
					subprocess.Popen(["play", wavfile])
				except:
					raise NotImplementedError('Install sox [sudo apt-get install sox libsox-fmt-all] to play sounds on Unix/Linux systems.')
			if sleep:
				time.sleep(self.duration)

	def spectrogram(self, low=None, high=None, log_power=True, other=None, plot=True, **kwds):
		'''
		Plots a spectrogram of the sound
		Arguments:
		``low=None``, ``high=None``
			If these are left unspecified, it shows the full spectrogram,
			otherwise it shows only between ``low`` and ``high`` in Hz.
		``log_power=True``
			If True the colour represents the log of the power.
		``**kwds``
			Are passed to Pylab's ``specgram`` command.
		Returns the values returned by pylab's ``specgram``, namely
		``(pxx, freqs, bins, im)`` where ``pxx`` is a 2D array of powers,
		``freqs`` is the corresponding frequencies, ``bins`` are the time bins,
		and ``im`` is the image axis.
		'''
		if self.nchannels > 1:
			raise ValueError('Can only plot spectrograms for mono sounds.')
		if other is not None:
			x = self.data.flatten() - other.data.flatten()
		else:
			x = self.data.flatten()
		pxx, freqs, bins, im = plt.specgram(x, Fs=self.samplerate, **kwds)
		if low is not None or high is not None:
			restricted = True
			if low is None:
				low = 0
			if high is None:
				high = numpy.amax(freqs)
			I = numpy.logical_and(low <= freqs, freqs <= high)
			I2 = numpy.where(I)[0]
			I2 = [numpy.max(numpy.min(I2) - 1, 0), numpy.min((numpy.max(I2) + 1, len(freqs) - 1))]
			Z = pxx[I2[0]:I2[-1], :]
		else:
			restricted = False
			Z = pxx
		if log_power:
			Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
			Z = 10 * numpy.log10(Z)
		Z = numpy.flipud(Z)
		if plot:
			if restricted:
				plt.imshow(Z, extent=(0, numpy.amax(bins), freqs[I2[0]], freqs[I2[-1]]),
					   origin='upper', aspect='auto')
			else:
				plt.imshow(Z, extent=(0, numpy.amax(bins), freqs[0], freqs[-1]),
					   origin='upper', aspect='auto')
			plt.title('Spectrogram')
			plt.xlabel('Time [sec]')
			plt.ylabel('Frequency [Hz]')
			plt.show()
		return pxx, freqs, bins, im

	def log_spectrogram(self, bands_per_octave=24):
		'''
		Returns the constant Q transform (log-spaced spectrogram) of a sound
		and plots it.
		bands_per_octave: number of bands per octave (*24*)
		'''
		high_edge = self.samplerate/2
		low_edge = 50 # Hz
		nfft = self.nsamples
		f_ratio = 2**(1/bands_per_octave) # Constant-Q bandwidth
		n_cqt = int(numpy.floor(numpy.log(high_edge/low_edge)/numpy.log(f_ratio)))
		pxx, fft_frqs, bins = self.spectrogram(plot=False)[:3]
		if n_cqt < 1:
			print("Warning: n_cqt not positive definite.")
		log_frqs = numpy.array([low_edge * numpy.exp(numpy.log(2)*i/bands_per_octave) for i in numpy.arange(n_cqt)])
		logf_bws = log_frqs * (f_ratio - 1)
		logf_bws[logf_bws<self.samplerate/nfft] = self.samplerate/nfft
		ovf_ctr = 0.5475 # Norm constant so CQT'*CQT close to 1.0
		tmp2 = 1/(ovf_ctr * logf_bws)
		tmp = (log_frqs.reshape(1, -1) - fft_frqs.reshape(-1, 1)) * tmp2
		Q = numpy.exp(-0.5 * tmp * tmp)
		Q *= 1 / (2 * numpy.sqrt((Q*Q).sum(0)))
		Q = Q.T
		cq_ft = numpy.sqrt(numpy.array(numpy.mat(Q) * numpy.mat(pxx)))
		plt.imshow(cq_ft, extent=(0, 61, log_frqs[0], log_frqs[-1]),origin='upper', aspect='auto')
		ticks = numpy.round(32000 * 2 ** (numpy.array(range(16), dtype=float)*-1))
		ticks = ticks[numpy.logical_and(ticks<log_frqs[-1], ticks>log_frqs[0])]
		plt.gca().set_yticks(ticks)
		plt.show()
		return cq_ft

	def spectrum(self, low=None, high=None, log_power=True, display=True):
		'''
		Returns the spectrum of the sound and optionally plots it.
		Arguments:
		``low``, ``high``
			If these are left unspecified, it shows the full spectrum,
			otherwise it shows only between ``low`` and ``high`` in Hz.
		``log_power=True``
			If True it returns the log of the power.
		``display=False``
			Whether to plot the output.
		Returns ``(Z, freqs, phase)``
		where ``Z`` is a 1D array of powers, ``freqs`` is the corresponding
		frequencies, ``phase`` is the unwrapped phase of spectrum.
		'''
		if self.nchannels>1:
			x = self.data[0,:] # silently use first channel only
		x = self.data.flatten()
		n = len(x)
		fftx = numpy.fft.fft(x) # take the fourier transform
		pxx = numpy.abs(fftx) # only keep the magnitude
		nUniquePts = int(numpy.ceil((n+1)/2))
		pxx = pxx[0:nUniquePts]
		pxx = pxx/n # scale by the number of points so that the magnitude does not depend on the length of the signal
		pxx = pxx**2 # square to get the power
		pxx[1:] *= 2 # we dropped half the FFT, so multiply by 2 to keep the same energy, except at the DC term at p[0] (which is unique) (not sure if necessary with rfft! CHECK!)
		freqs = numpy.linspace(0,1,len(pxx)) * (self.samplerate/2)
		phase = numpy.unwrap(numpy.mod(numpy.angle(fftx), 2 * numpy.pi))
		phase = phase[0:nUniquePts]
		if low is not None or high is not None:
			if low is None:
				low = 0
			if high is None:
				high = numpy.amax(freqs)
			I = numpy.logical_and(low <= freqs, freqs <= high)
			I2 = numpy.where(I)[0]
			Z = pxx[I2]
			freqs = freqs[I2]
			phase = phase[I2]
		else:
			restricted = False
			Z = pxx
		if log_power:
			Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
			Z = 10 * numpy.log10(Z)
		if display:
			plt.subplot(211)
			plt.semilogx(freqs, Z)
			ticks_freqs = numpy.round(32000 * 2 ** (numpy.array(range(16), dtype=float)*-1))
			plt.xticks(ticks_freqs, map(str, ticks_freqs.astype(int)))
			plt.grid()
			plt.xlim((freqs[1], freqs[-1]))
			plt.ylabel('Power [dB/Hz]') if log_power else plt.ylabel('Power')
			plt.title('Spectrum')
			plt.subplot(212)
			plt.semilogx(freqs, phase)
			ticks_freqs = numpy.round(32000 * 2 ** (numpy.array(range(16), dtype=float)*-1))
			plt.xticks(ticks_freqs, map(str, ticks_freqs.astype(int)))
			plt.grid()
			plt.xlim((freqs[1], freqs[-1]))
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Phase [rad]')
			plt.show()
		return (Z, freqs, phase)

	def waveform(self, start=0, end=None):
		'''
		Plots the waveform of the sound.
		Arguments:
		``start``, ``end`` (samples or time)
		If these are left unspecified, it shows the full waveform
		'''
		start = self.in_samples(start,self.samplerate)
		if end is None:
			end = self.nsamples
		end = self.in_samples(end,self.samplerate)
		for i in range(self.nchannels):
			plt.plot(self.times[start:end],self.channel(i)[start:end])
		plt.title('Waveform')
		plt.xlabel('Time [sec]')
		plt.ylabel('Amplitude')
		plt.show()


class Filter(Signal):
	pass # class for HRTFs and microphone/speaker filter functions
	# methods pertaining to both should go here:
	# calculating/plotting transfer function
	# applying the filter to a Sound
	# inverse filter
	# making standard filters (high, low, notch, bp)
	# FIR, but also FFR filters?
	# add gammatone filterbank


class HRTF():
	'''
	Class for reading and manipulating head-related transfer functions. This is essentially
	a collection of two Filter objects (hrtf.left and hrtf.right) with functions to manage them.
	>>> hrtf = HRTF(data='mit_kemar_normal_pinna.sofa') # initialize from sofa file
	>>> print(hrtf)
	<class 'slab.HRTF'> sources 710, elevations 14, samples 710, samplerate 44100.0
	>>> sourceidx = hrtf.cone_sources(20)
	>>> hrtf.plot_sources(sourceidx)
	>>> hrtf.plot_tf(sourceidx,ear='left')

	'''
	# instance properties
	nsources = property(fget=lambda self:len(self.sources),
						doc='The number of sources in the HRTF.')
	nelevations = property(fget=lambda self:len(self.elevations()),
						doc='The number of elevations in the HRTF.')

	def __init__(self,data,samplerate=None,sources=None,listener=None,verbose=False):
		if isinstance(data, str):
			if samplerate is not None:
				raise ValueError('Cannot specify samplerate when initialising HRTF from a file.')
			if pathlib.Path(data).suffix != '.sofa':
				raise NotImplementedError('Only .sofa files can be read at the moment.')
			else: # load from SOFA file
				try:
					f = HRTF._sofa_load(data,verbose)
				except:
					raise ValueError('Unable to read file.')
				data = HRTF._sofa_get_FIR(f)
				self.samplerate = HRTF._sofa_get_samplerate(f)
				self.left = Filter(data[:,0,:],self.samplerate) # create a Filter object for left ear data
				self.right = Filter(data[:,1,:],self.samplerate) # create a Filter object for right ear data
				self.listener = HRTF._sofa_get_listener(f)
				self.sources = HRTF._sofa_get_sourcepositions(f)
		else:
			self.samplerate = samplerate
			self.left = Filter(data[:,0,:],self.samplerate)
			self.right = Filter(data[:,1,:],self.samplerate)
			self.sources = sources
			self.listener = listener

	def __repr__(self):
		return f'{type(self)} (\n{repr(self.left.data)} \n{repr(self.right.data)} \n{repr(self.samplerate)})'

	def __str__(self):
		return f'{type(self)} sources {self.nsources}, elevations {self.nelevations}, samples {self.left.nsamples}, samplerate {self.samplerate}'

	# Static methods (used in __init__)
	@staticmethod
	def _sofa_load(filename,verbose=False):
		'Reads a SOFA file and returns a h5netcdf structure'
		if not have_h5:
			raise ImportError('Reading from sofa files requires h5py and h5netcdf.')
		f = h5netcdf.File(filename,'r')
		if verbose:
			f.items()
		return f

	@staticmethod
	def _sofa_get_samplerate(f):
		'returns the sampling rate of the recordings'
		attr = dict(f.variables['Data.SamplingRate'].attrs.items()) # get attributes as dict
		if attr['Units'].decode('UTF-8') == 'hertz': # extract and decode Units
			return float(numpy.array(f.variables['Data.SamplingRate'],dtype='float'))
		else: # Khz?
			warnings.warn('Unit other than Hz. ' + attr['Units'].decode('UTF-8') + '. Assuming kHz.')
			return 1000 * float(numpy.array(f.variables['Data.SamplingRate'],dtype='float'))

	@staticmethod
	def _sofa_get_sourcepositions(f):
		'returns an array of positions of all sound sources'
		# spherical coordinates, (azi,ele,radius), azi 0..360 (0=front, 90=left, 180=back), ele -90..90
		attr = dict(f.variables['SourcePosition'].attrs.items()) # get attributes as dict
		unit = attr['Units'].decode('UTF-8').split(',')[0] # extract and decode Units
		if unit != 'degree':
			warnings.warn('Non-degree unit: ' + unit)
		return numpy.array(f.variables['SourcePosition'],dtype='float')

	@staticmethod
	def _sofa_get_listener(f):
		'''Returns dict with listener attributes from a sofa file handle.
		Keys: pos, view, up, viewvec, upvec. Used for adding a listener vector in plot functions.'''
		lis = {}
		lis['pos'] = numpy.array(f.variables['ListenerPosition'],dtype='float')[0]
		lis['view']= numpy.array(f.variables['ListenerView'],dtype='float')[0]
		lis['up']  = numpy.array(f.variables['ListenerUp'],dtype='float')[0]
		lis['viewvec'] = numpy.concatenate([lis['pos'],lis['pos']+lis['view']])
		lis['upvec'] = numpy.concatenate([lis['pos'],lis['pos']+lis['up']])
		return lis

	@staticmethod
	def _sofa_get_FIR(f):
		'Returns an array of FIR filters for all source positions from a sofa file handle.'
		datatype = f.attrs['DataType'].decode('UTF-8') # get data type
		if datatype != 'FIR':
			warnings.warn('Non-FIR data: ' + datatype)
		return numpy.array(f.variables['Data.IR'],dtype='float')

	# instance methods
	def elevations(self):
		'Return the list of sources'
		return sorted(list(set(self.sources[:,1])))

	def plot_tf(self,sourceidx,ear,linesep=20):
		"Plots a transfer functions of FIR filters for a given ear ['left','right'] at a given sourcepositions index"
		n = 0
		if ear == 'left':
			data = self.left.data
		elif ear == 'right':
			data = self.right.data
		else:
			raise ValueError("Unknown value for ear. Use 'left' or 'right'")
		for s in sourceidx:
			w, h = scipy.signal.freqz(data[s])
			freqs = self.samplerate*w/(2*numpy.pi)/1000 # convert rad/sample to kHz
			plt.plot(freqs,20 * numpy.log10(abs(h)) + n,label=str(self.sources[s,1])+'˚')
			n += linesep
		#plt.xscale('log')
		plt.ylabel('Amplitude [dB]')
		plt.xlabel('Frequency [kHz]')
		#plt.legend(loc='upper left')
		plt.grid()
		plt.axis('tight')
		plt.xlim(4,18)
		#layout(fig)
		plt.show()

	def remove_ctf(self): # UNFINISHED, UNTESTED!!!
		'''Removes the constant (non-spatial) portion of the transfer functions from an HRTF object (in place).
		Returns the constant transfer function.'''
		ctf = []
		n = len(self.fir[:, 0])
		for idx in range(n):
			_, h = scipy.signal.freqz(self.fir[idx, 0])
			ctf += numpy.log10(abs(h))/n
		for idx in range(n):
			w, h = scipy.signal.freqz(self.fir[idx, 0])
			self.fir[idx, 0] = numpy.log10(abs(h)) - ctf
		return ctf

	def median_sources(self):
		'DOC'
		idx = numpy.where(self.sources[:,0]==0)[0]
		return sorted(idx, key=lambda x: self.sources[x,1])

	def cone_sources(self,cone):
		'Return indices of sources along an off-axis sphere slice'
		cone = numpy.sin(numpy.deg2rad(cone))
		azimuth = numpy.deg2rad(self.sources[:,0])
		elevation = numpy.deg2rad(self.sources[:,1]-90)
		x = numpy.sin(elevation) * numpy.cos(azimuth)
		y = numpy.sin(elevation) * numpy.sin(azimuth)
		eles = self.elevations()
		out = []
		for ele in eles: # for each elevation, find the source closest to the target y
			subidx, = numpy.where((self.sources[:,1]==ele) & (x>=0))
			cmin = numpy.min(numpy.abs(y[subidx]-cone))
			idx, = numpy.where( (self.sources[:,1]==ele) & (numpy.abs(y-cone)==cmin) )
			out.append(idx[0])
		return sorted(out, key=lambda x: self.sources[x,1])

	def plot_sources(self,idx=False):
		'DOC'
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax = Axes3D(fig)
		azimuth = numpy.deg2rad(self.sources[:,0])
		elevation = numpy.deg2rad(self.sources[:,1]-90)
		r = self.sources[:,2]
		x = r * numpy.sin(elevation) * numpy.cos(azimuth)
		y = r * numpy.sin(elevation) * numpy.sin(azimuth)
		z = r * numpy.cos(elevation)
		ax.scatter(x,y,z, c = 'b', marker='.')
		ax.scatter(0,0,0, c = 'r', marker='o')
		if self.listener: # TODO: view dir is inverted!
			x_, y_, z_, u, v, w = zip(*[self.listener['viewvec'],self.listener['upvec']])
			ax.quiver(x_, y_, z_, u, v, w, length = 0.5, colors=['r','b','r','r','b','b'])
		if idx:
			ax.scatter(x[idx],y[idx],z[idx], c='r', marker='o')
		ax.set_xlabel('X [m]')
		ax.set_ylabel('Y [m]')
		ax.set_zlabel('Z [m]')
		plt.show()


if __name__ == '__main__':
	sig1 = Sound.whitenoise()
	lev = sig1.level
	sig1.filter((500, 1000), type='bp')
	sig1.log_spectrogram()
	sig2 = Sound.clicktrain()
	sig2.level = 60
	sig3 = Sound.crossfade(sig1, sig2, overlap=0.5)
	sig3.play()
	sig3.waveform()
