#TODO: 
# irns, irno nor working (delay in samples or sec?)
# extend vowel to include more vowels (based on own makevowel.m)
# add other own stim functions (musical_rain, transitions)?
# add auditory spectrogram (filterbank) -> gammatone filterbank followed by halfwave rectification, cube root compression and 10 Hz low pass filtering
# add sound.rec using soundcard


import numpy
import array
import time
import math

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

from platform import system
system = system()
if system == 'Windows':
	import winsound
else:
	import subprocess
try:
	import scipy
	have_scipy = True
except ImportError:
	have_scipy = False
try:
	import matplotlib.pyplot as plt
	have_pyplot = True
except ImportError:
	have_pyplot = False


default_samplerate = 8000 # Hz

def get_samplerate(samplerate):
	if samplerate is None:
		return default_samplerate
	else:
		return samplerate
    
def set_default_samplerate(samplerate):
	'''
	Sets the default samplerate for Sound objects, by default 8000 Hz.
	'''
	global default_samplerate
	default_samplerate = samplerate

class Sound(numpy.ndarray):
	'''
	Class for working with sounds, including loading/saving, manipulating and playing.
	
	**Initialisation**
	
	The following arguments are used to initialise a sound object
	
	``data``
		Can be a filename, an array, a function or a sequence (list or tuple).
		If its a filename, the sound file (WAV or AIFF) will be loaded. If its
		an array, it should have shape ``(nsamples, nchannels)``. If its a
		function, it should be a function f(t). If its a sequence, the items
		in the sequence can be filenames, functions, arrays or Sound objects.
		The output will be a multi-channel sound with channels the corresponding
		sound for each element of the sequence. 
	``samplerate=None``
		The samplerate, if necessary, will use the default (for an array or
		function) or the samplerate of the data (for a filename).
	``duration=None``
		The duration of the sound, if initialising with a function.
	
	Examples:
	>>> import sound
	>>> import numpy
	>>> print(sound.Sound(lambda t: numpy.sin(2*numpy.pi*500*t),samplerate=10,duration=10))
	Sound duration 10.0, channels 1, samplerate 10
	>>> print(sound.Sound(numpy.ones([10,2]),samplerate=10))
	Sound duration 1.0, channels 2, samplerate 10
	>>> print(sound.Sound(numpy.ones([10,2]),samplerate=10).left)
	Sound duration 1.0, channels 1, samplerate 10
	
	**Reading, writing and playing**
	
	Sound.write(sound, filename) or writesound(sound, filename):
		Write a sound object to a wav file.
		Example:
		>>> sig = sound.Sound.tone(500, 8000, samplerate=8000)
		>>> sig = sound.Sound(lambda t: numpy.sin(2*numpy.pi*500*t),samplerate=8000,duration=1)
		>>> sig.write('tone.wav')

	Sound.read(filename) or readsound(filename):
		Load the file given by filename and returns a Sound object. 
		Sound file can be either a .wav or a .aif file.
		Example:
		>>> sig2 = sound.Sound('tone.wav')
		>>> print(sig2)
		Sound duration 1.0, channels 1, samplerate 8000

	play(*sounds, **kwargs):
	Plays a sound or sequence of sounds. For example::
		>>> sig.play(sleep=True)
		>>> sound.play(sig,sig2)
	
	If ``sleep=True`` the function will wait
	until the sounds have finished playing before returning.

	**Properties**
	>>> sig = sound.Sound.tone(500, 8000, samplerate=8000)
	>>> sig.duration
	1.0
	>>> sig.nsamples
	8000
	>>> sig.nchannels
	1
	>>> len(sig.times)
	8000
	>>> sig.left == sig.channel(0)
	>>> #sig.right
	
	**Generating sounds**
	
	All sound generating methods can be used with durations arguments in samples (int) or seconds (float). One can also set the number of channels by setting the keyword argument nchannels to the desired value. Notice that for noise the channels will be generated independantly.
	
	tone(frequency, duration, phase=0, samplerate=None, nchannels=1):
		Returns a pure tone at frequency for duration, using the default
		samplerate or the given one. The ``frequency`` and ``phase`` parameters
		can be single values, in which case multiple channels can be
		specified with the ``nchannels`` argument, or they can be sequences
		(lists/tuples/arrays) in which case there is one frequency or phase for
		each channel.
		Examples:	
		>>> print(sound.Sound.tone(500, duration=8000, samplerate=8000)) # duration in samples
		Sound duration 1.0, channels 1, samplerate 8000
		>>> print(sound.Sound.tone(500, duration=1.0, samplerate=8000)) # duration in seconds, same result as above
		Sound duration 1.0, channels 1, samplerate 8000
		>>> sig = sound.Sound.tone([500,501], 10, samplerate=8000)
		
	harmoniccomplex(f0, duration, amplitude=1, phase=0, samplerate=None, nchannels=1):
		Returns a harmonic complex composed of pure tones at integer multiples
		of the fundamental frequency ``f0``. 
		The ``amplitude`` and ``phase`` keywords can be set to either a single 
		value or an array of values. In the former case the value is set for all
		harmonics, and harmonics up to the sampling frequency are generated. 
		In the latter each harmonic parameter is set separately, and the number 
		of harmonics generated corresponds to the length of the array.	
		Example:
		>>> sound.Sound.play(sound.Sound.harmoniccomplex(500, duration=1.0, amplitude=(1,1,1), samplerate=8000),sleep=True)

	whitenoise(duration, samplerate=None, nchannels=1):
		Returns a white noise. If the samplerate is not specified, the global
		default value will be used. nchannels = 2 produces uncorrelated noise (dichotic).
		>>> noise = sound.Sound.whitenoise(1.0,nchannels=2)
		
		To make a diotic noise:
		>>> noise[:,1] = noise[:,0]
		
	powerlawnoise(duration, alpha, samplerate=None, nchannels=1,normalise=False):
		Returns a power-law noise for the given duration. Spectral density per unit of bandwidth scales as 1/(f**alpha).
		Example:
		>>> sound.Sound.play(Sound.powerlawnoise(0.2, 1, samplerate=8000),sleep=True)
		
		Arguments:
		``duration`` : Duration of the desired output.
		``alpha`` : Power law exponent.
		``samplerate`` : Desired output samplerate
		pinknoise and brownnoise are wrappers of powerlawnoise with different exponents.

	click(duration, peak=None, samplerate=None, nchannels=1)
		Returns a click of the given duration.
		If ``peak`` is not specified, the amplitude will be 1, otherwise
		``peak`` refers to the peak dB SPL of the click, according to the
		formula ``28e-6*10**(peak/20)``.
		Example
		>>> sig = sound.Sound.click(1, samplerate=8000)
		
	clicktrain(duration, freq, peak=None, samplerate=None, nchannels=1)
		Returns a series of clicks at a frequency of freq.
		>>> sig = sound.Sound.clicktrain(500, 8000, samplerate=8000)
		
	silence(duration, samplerate=None, nchannels=1)
		Returns a silent, zero sound for the given duration. Set nchannels to set the number of channels.
		>>> sil = sound.Sound.silence(0.5, samplerate=8000)
		
	vowel(vowel=None, formants=None, pitch=100, duration=1., samplerate=None, nchannels=1):
		Returns an artifically created spoken vowel sound (following the 
		source-filter model of speech production) with a given ``pitch``.
		The vowel can be specified by either providing ``vowel`` as a string
		('a', 'i' or 'u') or by setting ``formants`` to a sequence of formant
		frequencies.
		The returned sound is normalized to a maximum amplitude of 1.
		Examples:
		>>> vowel_a = sound.Sound.vowel(vowel='a', pitch=110, duration=.5, samplerate=8000)
		>>> vowel_u = sound.Sound.vowel(vowel='u', pitch=90, duration=.5, samplerate=8000)
		>>> vowels = sound.Sound.sequence(vowel_a, sil, vowel_u)
		>>> sound.Sound.play(vowels,sleep=True)
	
	irns(delay, gain, niter, duration, samplerate=None, nchannels=1):
		Returns an IRN_S noise. The iterated ripple noise is obtained trough
		a cascade of gain and delay filtering.
		>>> irn = sound.Sound.irns(delay=0.001, gain=1, niter=32, duration=1., samplerate=8000)
		>>> sound.Sound.play(irn,sleep=True)
		>>> irn = sound.Sound.irno(delay=0.001, gain=1, niter=32, duration=1., samplerate=8000)
		>>> sound.Sound.play(irn,sleep=True)
		
	**Timing and sequencing**
	
	sequence(*sounds, samplerate=None)
	repeat(n)
	shifted(duration, fractional=False, filter_length=2048)
	resized

	**Slicing**

	One can slice sound objects in various ways, for example ``sound[0.1:0.2]``
	returns the part of the sound between 100 ms and 200 ms (not including the
	right hand end point). You can also set values using slicing, e.g.
	``sound[:50] = 0`` will silence the first 50 samples of the sound. The syntax
	is the same as usual for Python slicing. In addition, you can select a
	subset of the channels by doing, for example, ``sound[:, -5:]`` would be
	the last 5 channels. For time indices, either times or samples can be given,
	e.g. ``sound[:100]`` gives the first 100 samples. In addition, steps can
	be used for example to reverse a sound as ``sound[::-1]``.
	
	**Arithmetic operations**
	
	Standard arithemetical operations and numpy functions work as you would
	expect with sounds, e.g:
	>>> sig = sound.Sound.click(1, samplerate=8000)
	>>> print(sig + sig)
	Sound duration 0.000125, channels 1, samplerate 8000
	>>> print(10 * sig)
	>>> print(abs(sig))

	**Level**
	
	level
	atlevel
	maxlevel
	atmaxlevel
	
	**Ramping**

	ramp(when='both', duration=0.01, envelope=None, inplace=True)
	Adds a ramp on/off to the sound
	``when='onset'``  Can take values 'onset', 'offset' or 'both'
	``duration=0.01`` The time over which the ramping happens (in samples or seconds)
	``envelope``      A ramping function, if not specified uses ``sin(pi*t/2)**2``. The
			          function should be a function of one variable ``t`` ranging from
			          0 to 1, and should increase from ``f(0)=0`` to ``f(0)=1``. The
			          reverse is applied for the offset ramp.
	``inplace``       Whether to apply ramping to current sound or return a new array.
	>>> tmp = vowel_a.ramp()
	
	ramped
	
	**Plotting**
	Examples:
		>>> pxx, freqs, bins, im = vowels.spectrogram(low=100, high=4000, log_power=True)
		>>> Z, freqs, phase = vowels.spectrum(low=100, high=4000, log_power=True)
		>>> vowels.waveform(start=0, end=.1)

	'''

	duration = property(fget=lambda self:len(self) / self.samplerate,
						doc='The length of the sound in seconds.')
	nsamples = property(fget=lambda self:len(self),
						doc='The number of samples in the sound.')
	times = property(fget=lambda self:numpy.arange(len(self), dtype=float) / self.samplerate,
					 doc='An array of times (in seconds) corresponding to each sample.')
	nchannels = property(fget=lambda self:self.shape[1],
						 doc='The number of channels in the sound.')
	left = property(fget=lambda self:self.channel(0),
					doc='The left channel for a stereo sound.')
	right = property(fget=lambda self:self.channel(1),
					 doc='The right channel for a stereo sound.')

	def __new__(cls, data, samplerate=None, duration=None):
		if isinstance(data, numpy.ndarray):
			if samplerate is None:
				raise ValueError('Must specify samplerate to initialise Sound with array.')
			if duration is not None:
				raise ValueError('Cannot specify duration when initialising Sound with array.')
			x = numpy.array(data, dtype='float')
		elif isinstance(data, str):
			if duration is not None:
				raise ValueError('Cannot specify duration when initialising Sound from file.')
			if samplerate is not None:
				raise ValueError('Cannot specify samplerate when initialising Sound from a file.')
			x = Sound.read(data)
			samplerate = x.samplerate
		elif callable(data):
			if samplerate is None:
				raise ValueError('Must specify samplerate to initialise Sound with function.')
			if duration is None:
				raise ValueError('Must specify duration to initialise Sound with function.')
			L = int(numpy.rint(duration * samplerate))
			t = numpy.arange(L, dtype=float) / samplerate
			x = data(t)
		elif isinstance(data, (list, tuple)):
			kwds = {}
			if samplerate is not None:
				kwds['samplerate'] = samplerate
			if duration is not None:
				kwds['duration'] = duration
			channels = tuple(Sound(c, **kwds) for c in data)
			x = numpy.hstack(channels)
			samplerate = channels[0].samplerate
		else:
			raise TypeError('Cannot initialise Sound with data of class ' + str(data.__class__))
		if len(x.shape)==1:
			x.shape = (len(x), 1)
		x = x.view(cls)
		x.samplerate = samplerate
		return x

	def channel(self, n):
		'''
		Returns the nth channel of the sound.
		'''
		return Sound(self[:, n], self.samplerate)

	def __add__(self, other):
		if isinstance(other, Sound):
			if int(other.samplerate) > int(self.samplerate):
				self = self.resample(other.samplerate)
			elif int(other.samplerate) < int(self.samplerate):
				other = other.resample(self.samplerate)
			if len(self) > len(other):
				other = other.resized(len(self))
			elif len(self) < len(other):
				self = self.resized(len(other))

			return Sound(numpy.ndarray.__add__(self, other), samplerate=self.samplerate)
		else:
			x = numpy.ndarray.__add__(self, other)
			return Sound(x, self.samplerate)
	__radd__ = __add__
	
	# getslice and setslice need to be implemented for compatibility reasons,
	# but __getitem__ covers all the functionality so we just use that
	def __getslice__(self, start, stop):
		return self.__getitem__(slice(start, stop))

	def __setslice__(self, start, stop, seq):
		return self.__setitem__(slice(start, stop), seq)
	
	def __getitem__(self,key):
		channel = slice(None)
		if isinstance(key, tuple):
			channel = key[1]
			key = key[0]

		if isinstance(key, int):
			return numpy.ndarray.__getitem__(self, key)
		if isinstance(key, float):
			return numpy.ndarray.__getitem__(self, round(key*self.samplerate))

		sliceattr = [v for v in [key.start, key.stop] if v is not None]
		attrisint = numpy.array([isinstance(v, int) for v in sliceattr])
		s = sum(attrisint)
		if s!=0 and s!=len(sliceattr):
			raise ValueError('Slice attributes must be all ints or all times')
		if s==len(sliceattr): # all ints
			start = key.start or 0
			stop = key.stop or self.shape[0]
			step = key.step or 1
			if start>=0 and stop<=self.shape[0]:
				return Sound(numpy.ndarray.__getitem__(self, (key, channel)),
							 self.samplerate)
			else:
				startpad = numpy.max(-start, 0)
				endpad = numpy.max(stop-self.shape[0], 0)
				startmid = numpy.max(start, 0)
				endmid = numpy.min(stop, self.shape[0])
				atstart = numpy.zeros((startpad, self.shape[1]))
				atend = numpy.zeros((endpad, self.shape[1]))
				return Sound(vstack((atstart,
									 asarray(self)[startmid:endmid:step],
									 atend)), self.samplerate)		
		start = key.start or 0
		stop = key.stop or self.duration
		step = key.step or 1
		if int(step)!=step:
			#resampling
			raise NotImplementedError
		start = int(numpy.rint(start*self.samplerate))
		stop = int(numpy.rint(stop*self.samplerate))
		return self.__getitem__((slice(start,stop,step),channel))
	
	def __setitem__(self,key,value):
		channel=slice(None)
		if isinstance(key,tuple):
			channel=key[1]
			key=key[0]
		
		if isinstance(key,int):
			return numpy.ndarray.__setitem__(self,(key,channel),value)
		if isinstance(key,float):
			return numpy.ndarray.__setitem__(self,(int(numpy.rint(key*self.samplerate)),channel),value)

		sliceattr = [v for v in [key.start, key.step, key.stop] if v is not None]
		attrisint = numpy.array([isinstance(v, int) for v in sliceattr])
		s = sum(attrisint)
		if s!=0 and s!=len(sliceattr):
			raise ValueError('Slice attributes must be all ints or all times')
		if s==len(sliceattr): # all ints
			# If value is a mono sound its shape will be (N, 1) but the numpy
			# setitem will have shape (N,) so in this case it's a shape mismatch
			# so we squeeze the array to make sure this doesn't happen.
			if isinstance(value,Sound) and channel!=slice(None):
				value=value.squeeze()
			return numpy.asarray(self).__setitem__((key,channel),value) # returns None

		if key.__getattribute__('step') is not None:
			# resampling?
			raise NotImplementedError
		
		start = key.start
		stop = key.stop or self.duration
		if (start is not None and start<0*ms) or stop > self.duration:
			raise IndexError('Slice bigger than Sound object')
		if start is not None: start = int(rint(start*self.samplerate))
		stop = int(rint(stop*self.samplerate))
		return self.__setitem__((slice(start,stop),channel),value)

	def extended(self, duration):
		'''
		Returns the Sound with length extended by the given duration, which
		can be the number of samples or a length of time in seconds.
		'''
		duration = get_duration(duration, self.samplerate)
		return self[:self.nsamples+duration]

	def resized(self, L):
		'''
		Returns the Sound with length extended (or contracted) to have L samples.
		'''
		if L == len(self):
			return self
		elif L < len(self):
			return Sound(self[:L, :], samplerate=self.samplerate)
		else:
			padding = zeros((L - len(self), self.nchannels))
			return Sound(concatenate((self, padding)), samplerate=self.samplerate)

	def shifted(self, duration, fractional=False, filter_length=2048):
		'''
		Returns the sound delayed by duration, which can be the number of
		samples or a length of time in seconds. Normally, only integer
		numbers of samples will be used, but if ``fractional=True`` then
		the filtering method from
		`http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html <http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html>`__
		will be used (introducing some small numerical errors). With this
		method, you can specify the ``filter_length``, larger values are
		slower but more accurate, especially at higher frequencies. The large
		default value of 2048 samples provides good accuracy for sounds with
		frequencies above 20 Hz, but not for lower frequency sounds. If you are
		restricted to high frequency sounds, a smaller value will be more
		efficient. Note that if ``fractional=True`` then
		``duration`` is assumed to be a time not a number of samples.
		'''
		if not fractional:
			if not isinstance(duration, int):
				duration = int(rint(duration*self.samplerate))
			if duration>=0:
				y = vstack((zeros((duration, self.nchannels)), self))
				return Sound(y, samplerate=self.samplerate)
			else:
				return self[-duration:, :]
		else:
			if not have_scipy:
				raise ImportError('SHIFTED by fractional samples requires scipy.')
			if self.nchannels>1:
				sounds = [self.channel(i).shifted(duration, fractional=True, filter_length=filter_length) for i in xrange(self.nchannels)]
				return Sound(hstack(sounds), samplerate=self.samplerate)
			# Adapted from
			# http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
			delay = duration*self.samplerate
			if delay>=0:
				idelay = int(delay)
			elif delay<0:
				idelay = -int(-delay)
			delay -= idelay
			centre_tap = filter_length // 2
			t = arange(filter_length)
			x = t-delay
			if abs(round(delay)-delay)<1e-10:
				tap_weight = array(x==centre_tap, dtype=float)
			else:
				sinc = sin(pi*(x-centre_tap))/(pi*(x-centre_tap))
				window = 0.54-0.46*cos(2.0*pi*(x+0.5)/filter_length) # Hamming window
				tap_weight = window*sinc
			if filter_length<256:
				y = convolve(tap_weight, self.flatten())
			else:
				y = scipy.signal.fftconvolve(tap_weight, self.flatten())
			y = y[filter_length/2:-filter_length/2]
			sound = Sound(y, self.samplerate)
			sound = sound.shifted(idelay)
			return sound

	def repeat(self, n):
		'''
		Repeats the sound n times
		'''
		x = numpy.vstack((self,)*int(n))
		return Sound(x, samplerate=self.samplerate)

	def resample(self, samplerate):
		'''
		Returns a resampled version of the sound.
		'''
		if not have_scipy:
			raise ImportError('Resampling requires scipy.')
		y = numpy.array(scipy.signal.resample(self,int(numpy.rint(samplerate*self.duration))),dtype='float64')
		return Sound(y,samplerate=samplerate)
	
	def play(self, sleep=False):
		'''
		Plays the sound. If sleep=True then the function will wait until the sound has finished
		playing before returning.
		'''
		if self.nchannels>2:
			raise ValueError("Can only play sounds with 1 or 2 channels.")
		a = numpy.amax(numpy.abs(self))
		#TEST!
		wavfile = 'tmp.wav'
		self.write(wavfile,normalise=True)
		if system == 'Windows':
			winsound.PlaySound('%s.wav' % wavfile, winsound.SND_FILENAME)
		elif system == 'Darwin':
			subprocess.Popen(["afplay", wavfile])
		else:  # Linus/Unix, install sox (sudo apt-get install sox libsox-fmt-all)
			subprocess.Popen(["play", wavfile])
		if sleep:
			time.sleep(self.duration)

	def spectrogram(self, low=None, high=None, log_power=True, other = None, **kwds):
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
		if self.nchannels>1:
			raise ValueError('Can only plot spectrograms for mono sounds.')
		if other is not None:
			x = self.flatten()-other.flatten()
		else:
			x = self.flatten()
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
		return (pxx, freqs, bins, im)

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
			raise ValueError('Can only plot spectrum for mono sounds.')

		# Flatten array, fft operates on the last axis by default
		sp = numpy.fft.fft(numpy.array(self).flatten())
		freqs = numpy.array(numpy.arange(len(sp)), dtype=float) / len(sp) * float(self.samplerate)
		pxx = numpy.abs(sp) ** 2
		phase = numpy.unwrap(numpy.mod(numpy.angle(sp), 2 * numpy.pi))
		if low is not None or high is not None:
			restricted = True
			if low is None:
				low = 0*Hz
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
			ticks_freqs = 32000 * 2 ** -numpy.array(range(18), dtype=float)
			plt.xticks(ticks_freqs, map(str, ticks_freqs))
			plt.grid()
			plt.xlim((freqs[0], freqs[-1]))
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Power [dB/Hz]') if log_power else ylabel('Power')
			plt.subplot(212)
			plt.semilogx(freqs, phase)
			ticks_freqs = 32000 * 2 ** -numpy.array(range(18), dtype=float)
			plt.xticks(ticks_freqs, map(str, ticks_freqs))
			plt.grid()
			plt.xlim((freqs[0], freqs[-1]))
			plt.xlabel('Frequency [Hz]')
			plt.ylabel('Phase [rad]')
			plt.title('Spectrum')
			plt.show()
		return (Z, freqs, phase)

	def waveform(self, start=0, end=None):
		start = get_duration(start,self.samplerate)
		if end is None:
			end = self.nsamples
		end = get_duration(end,self.samplerate)
		t = self.times
		for i in range(self.nchannels):
			plt.plot(self.times[start:end],self.channel(i)[start:end])
		plt.title('Waveform')
		plt.xlabel('Time [sec]')
		plt.ylabel('Amplitude')
		plt.show()

	def get_level(self):
		'''
		Returns level in dB SPL (RMS) assuming array is in Pascals.
		In the case of multi-channel sounds, returns an array of levels
		for each channel, otherwise returns a float.
		'''
		if self.nchannels==1:
			rms_value = numpy.sqrt(numpy.mean((asarray(self)-numpy.mean(asarray(self)))**2))
			rms_dB = 20.0*numpy.log10(rms_value/2e-5)
			return rms_dB
		else:
			return array(tuple(self.channel(i).get_level() for i in xrange(self.nchannels)))

	def set_level(self, level):
		'''
		Sets level in dB SPL (RMS) assuming array is in Pascals. ``level``
		should be a value in dB, or a tuple of levels, one for each channel.
		'''
		rms_dB = self.get_level()
		if self.nchannels>1:
			level = array(level)
			if level.size==1:
				level = level.repeat(self.nchannels)
			level = reshape(level, (1, self.nchannels))
			rms_dB = reshape(rms_dB, (1, self.nchannels))
		else:
			rms_dB = float(rms_dB)
			level = float(level)
		gain = 10**((level-rms_dB)/20.)
		self *= gain

	level = property(fget=get_level, fset=set_level, doc='''
		Can be used to get or set the level of a sound, which should be in dB.
		For single channel sounds a value in dB is used, for multiple channel
		sounds a value in dB can be used for setting the level (all channels
		will be set to the same level), or a list/tuple/array of levels. It
		is assumed that the unit of the sound is Pascals.
		''')
	
	def atlevel(self, level):
		'''
		Returns the sound at the given level in dB SPL (RMS) assuming array is
		in Pascals. ``level`` should be a value in dB, or a tuple of levels,
		one for each channel.
		'''
		newsound = self.copy()
		newsound.level = level
		return newsound
	
	def get_maxlevel(self):
		return amax(self.level)
	
	def set_maxlevel(self, level):
		self.level += level-self.maxlevel
		
	maxlevel = property(fget=get_maxlevel, fset=set_maxlevel, doc='''
		Can be used to set or get the maximum level of a sound. For mono
		sounds, this is the same as the level, but for multichannel sounds
		it is the maximum level across the channels. Relative level differences
		will be preserved. The specified level should be a value in dB, and it
		is assumed that the unit of the sound is Pascals. 
		''')

	def atmaxlevel(self, level):
		'''
		Returns the sound with the maximum level across channels set to the
		given level. Relative level differences will be preserved. The specified
		level should be a value in dB and it is assumed that the unit of the
		sound is Pascals.
		'''
		newsound = self.copy()
		newsound.maxlevel = level
		return newsound
			
	def ramp(self, when='both', duration=0.01, envelope=None, inplace=True):
		'''
		Adds a ramp on/off to the sound
		
		``when='onset'``
			Can take values 'onset', 'offset' or 'both'
		``duration=0.01``
			The time over which the ramping happens (in samples or seconds)
		``envelope``
			A ramping function, if not specified uses ``sin(pi*t/2)**2``. The
			function should be a function of one variable ``t`` ranging from
			0 to 1, and should increase from ``f(0)=0`` to ``f(0)=1``. The
			reverse is applied for the offset ramp.
		``inplace``
			Whether to apply ramping to current sound or return a new array.
		'''
		when = when.lower().strip()
		if envelope is None: envelope = lambda t:numpy.sin(numpy.pi * t / 2) ** 2
		sz = get_duration(duration,self.samplerate)
		multiplier = envelope(numpy.reshape(numpy.linspace(0.0, 1.0, sz), (sz, 1)))
		if inplace:
			target = self
		else:
			target = Sound(copy(self), self.samplerate)
		if when == 'onset' or when == 'both':
			target[:sz, :] *= multiplier
		if when == 'offset' or when == 'both':
			target[target.nsamples-sz:, :] *= multiplier[::-1]
		return target
	
	def fft(self,n=None):
		'''
		Performs an n-point FFT on the sound object, that is an array of the same size containing the DFT of each channel.
		n defaults to the number of samples of the sound, but can be changed manually setting the ``n`` keyword argument
		'''
		if n is None:
			n=self.shape[0]
		res=zeros(n,self.nchannels)
		for i in range(self.nchannels):
			res[:,i]=fft(asarray(self)[:,i].flatten(),n=n)
		return res

	@staticmethod
	def tone(frequency, duration, phase=0, samplerate=None, nchannels=1):
		'''
		Returns a pure tone at frequency for duration, using the default
		samplerate or the given one. The ``frequency`` and ``phase`` parameters
		can be single values, in which case multiple channels can be
		specified with the ``nchannels`` argument, or they can be sequences
		(lists/tuples/arrays) in which case there is one frequency or phase for
		each channel.
		'''
		samplerate = get_samplerate(samplerate)
		duration = get_duration(duration,samplerate)
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
	def harmoniccomplex(f0, duration, amplitude=1, phase=0, samplerate=None, nchannels=1):
		'''
		Returns a harmonic complex composed of pure tones at integer multiples
		of the fundamental frequency ``f0``. 
		The ``amplitude`` and
		``phase`` keywords can be set to either a single value or an
		array of values. In the former case the value is set for all
		harmonics, and harmonics up to the sampling frequency are
		generated. In the latter each harmonic parameter is set
		separately, and the number of harmonics generated corresponds
		to the length of the array.
		'''
		samplerate = get_samplerate(samplerate)
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
			
		x = amplitudes[0]*Sound.tone(f0, duration, phase = phases[0], 
							   samplerate = samplerate, nchannels = nchannels)
		for i in range(1,Nharmonics):
			x += amplitudes[i]*Sound.tone((i+1)*f0, duration, phase = phases[i], 
									samplerate = samplerate, nchannels = nchannels)
		return Sound(x,samplerate)
	
	@staticmethod
	def whitenoise(duration, samplerate=None, nchannels=1):
		'''
		Returns a white noise. If the samplerate is not specified, the global
		default value will be used. nchannels = 2 produces uncorrelated noise (dichotic).
		>>> noise = Sound.whitenoise(1.0,nchannels=2)
		
		To make a diotic noise:
		>>> noise[:,1] = noise[:,0]
		
		'''
		samplerate = get_samplerate(samplerate)
		duration = get_duration(duration,samplerate)
		x = numpy.random.randn(duration,nchannels)
		return Sound(x, samplerate)

	@staticmethod
	def powerlawnoise(duration, alpha, samplerate=None, nchannels=1,normalise=False):
		'''
		Returns a power-law noise for the given duration. Spectral density per unit of bandwidth scales as 1/(f**alpha).
		Example:
		>>> noise = Sound.powerlawnoise(0.2, 1, samplerate=8000)
		
		Arguments:
		``duration`` : Duration of the desired output.
		``alpha`` : Power law exponent.
		``samplerate`` : Desired output samplerate
		'''
		samplerate = get_samplerate(samplerate)
		duration = get_duration(duration,samplerate)
		
		# Adapted from http://www.eng.ox.ac.uk/samp/software/powernoise/powernoise.m
		# Little MA et al. (2007), "Exploiting nonlinear recurrence and fractal
		# scaling properties for voice disorder detection", Biomed Eng Online, 6:23
		n=duration
		n2=int(n/2)
		
		f=numpy.array(numpy.fft.fftfreq(n,d=1.0/samplerate), dtype=complex)
		f.shape=(len(f),1)
		f=numpy.tile(f,(1,nchannels))
		
		if n%2==1:
			z=(numpy.random.randn(n2,nchannels)+1j*numpy.random.randn(n2,nchannels))
			a2=1.0/( f[1:(n2+1),:]**(alpha/2.0))
		else:
			z=(numpy.random.randn(n2-1,nchannels)+1j*numpy.random.randn(n2-1,nchannels))
			a2=1.0/(f[1:n2,:]**(alpha/2.0))
		
		a2*=z
		
		if n%2==1:
			d=numpy.vstack((numpy.ones((1,nchannels)),a2,
					  numpy.flipud(numpy.conj(a2))))
		else:
			d=numpy.vstack((numpy.ones((1,nchannels)),a2,
					  1.0/( numpy.abs(f[n2])**(alpha/2.0) )*
					  numpy.random.randn(1,nchannels),
					  numpy.flipud(numpy.conj(a2))))
		
		
		x=numpy.real(numpy.fft.ifft(d.flatten()))           

		x.shape=(n,nchannels)

		if normalise:
			for i in range(nchannels):
				#x[:,i]=normalise_rms(x[:,i])
				x[:,i] = ((x[:,i] - amin(x[:,i]))/(amax(x[:,i]) - amin(x[:,i])) - 0.5) * 2;
		
		return Sound(x,samplerate)
	
			
	@staticmethod
	def pinknoise(duration, samplerate=None, nchannels=1, normalise=False):
		'''
		Returns pink noise, i.e :func:`powerlawnoise` with alpha=1
		'''
		return Sound.powerlawnoise(duration, 1.0, samplerate=samplerate,
								   nchannels=nchannels, normalise=normalise)
	
	@staticmethod
	def brownnoise(duration, samplerate=None, nchannels=1, normalise=False):
		'''
		Returns brown noise, i.e :func:`powerlawnoise` with alpha=2
		'''
		return Sound.powerlawnoise(duration, 2.0, samplerate=samplerate,
								   nchannels=nchannels, normalise=normalise)
	
	@staticmethod
	def irns(delay, gain, niter, duration, samplerate=None, nchannels=1):
		'''
		Returns an IRN_S noise. The iterated ripple noise is obtained trough
		a cascade of gain and delay filtering. 
		For more details: see Yost 1996 or chapter 15 in Hartman Sound Signal Sensation.
		'''
		if not have_scipy:
			raise ImportError('INRS requires scipy. Use IRNO instead.')
		if nchannels!=1:
			raise ValueError("nchannels!=1 not supported.")
		samplerate = get_samplerate(samplerate)
		delay = get_duration(delay,samplerate)
		noise=Sound.whitenoise(duration,samplerate=samplerate)
		x=numpy.array(noise.T)[0]
		IRNfft=numpy.fft.fft(x)
		Nspl,spl_dur=len(IRNfft),float(1/samplerate)
		w=2*numpy.pi*numpy.fft.fftfreq(Nspl,spl_dur)
		d=float(delay)
		for k in range(1,niter+1):
			nchoosek=math.factorial(niter)/(math.factorial(niter-k)*math.factorial(k))
			IRNfft+=nchoosek*(gain**k)*IRNfft*numpy.exp(-1j*w*k*d)
		IRNadd = numpy.fft.ifft(IRNfft)
		x=numpy.real(IRNadd)
		return Sound(x,samplerate)
	
	@staticmethod
	def irno(delay, gain, niter, duration, samplerate=None, nchannels=1):
		'''
		Returns an IRN_O noise. The iterated ripple noise is obtained many attenuated and
		delayed version of the original broadband noise. 
		For more details: see Yost 1996 or chapter 15 in Hartman Sound Signal Sensation.
		'''
		samplerate = get_samplerate(samplerate)
		delay = get_duration(delay,samplerate)
		noise=Sound.whitenoise(duration,samplerate=samplerate)
		x=numpy.array(noise.T)[0]
		IRNadd=numpy.fft.fft(x)
		Nspl,spl_dur=len(IRNadd),float(1/samplerate)
		w=2*numpy.pi*numpy.fft.fftfreq(Nspl,spl_dur)
		d=float(delay)
		for k in range(1,niter+1):
			IRNadd+=(gain**k)*IRNadd*numpy.exp(-1j*w*k*d)
		IRNadd = numpy.fft.ifft(IRNadd)
		x=numpy.real(IRNadd)
		return Sound(x, samplerate)

	@staticmethod
	def click(duration, peak=None, samplerate=None, nchannels=1):
		'''
		Returns a click of the given duration.
		
		If ``peak`` is not specified, the amplitude will be 1, otherwise
		``peak`` refers to the peak dB SPL of the click, according to the
		formula ``28e-6*10**(peak/20.)``.
		'''
		samplerate = get_samplerate(samplerate)
		duration = get_duration(duration,samplerate)
		if peak is not None:
			amplitude = 28e-6*10**(float(peak)/20)
		else:
			amplitude = 1
		x = amplitude*numpy.ones((duration,nchannels))
		return Sound(x, samplerate)
	
	@staticmethod
	def clicktrain(duration, freq, clickduration=1, peak=None, samplerate=None, nchannels=1):
		'''
		Returns a series of n clicks (see :func:`click`) at a frequency of freq.
		'''
		samplerate = get_samplerate(samplerate)
		duration = get_duration(duration,samplerate)
		clickduration = get_duration(clickduration,samplerate)
		interval = numpy.rint(1/freq * samplerate)
		n = numpy.rint(duration/interval)
		oneclick = Sound.click(clickduration, peak=peak, samplerate=samplerate)
		sil = Sound.silence(interval-clickduration)
		return Sound.sequence(oneclick,sil).repeat(n)

	@staticmethod
	def silence(duration, samplerate=None, nchannels=1):
		'''
		Returns a silent, zero sound for the given duration. Set nchannels to set the number of channels.
		'''
		samplerate = get_samplerate(samplerate)
		duration = get_duration(duration,samplerate)
		x=numpy.zeros((duration,nchannels))
		return Sound(x, samplerate)

	@staticmethod
	def vowel(vowel=None, formants=None, pitch=100, duration=1, samplerate=None, nchannels=1):
		'''
		Returns an artifically created spoken vowel sound (following the 
		source-filter model of speech production) with a given ``pitch``.
		
		The vowel can be specified by either providing ``vowel`` as a string
		('a', 'i' or 'u') or by setting ``formants`` to a sequence of formant
		frequencies.
		
		The returned sound is normalized to a maximum amplitude of 1.
		
		The implementation is based on the MakeVowel function written by Richard
		O. Duda, part of the Auditory Toolbox for Matlab by Malcolm Slaney:
		http://cobweb.ecn.purdue.edu/~malcolm/interval/1998-010/                
		'''    
		if not have_scipy:
			raise ImportError('VOWEL requires scipy.')
		samplerate = get_samplerate(samplerate)
		duration = get_duration(duration, samplerate)
		
		if not (vowel or formants):
			raise ValueError('Need either a vowel or a list of formants')
		elif (vowel and formants):
			raise ValueError('Cannot use both vowel and formants')
			
		if vowel:
			if vowel == 'a' or vowel == '/a/':
				formants = (730.0, 1090.0, 2440.0)
			elif vowel == 'i' or vowel == '/i/':
				formants = (270.0, 2290.0, 3010.0)
			elif vowel == 'u' or vowel == '/u/':
				formants = (300.0, 870.0, 2240.0)
			else:
				raise ValueError('Unknown vowel: "%s"' % (vowel))            
		
		points = numpy.arange(0, duration - 1, samplerate / pitch)
			
		indices = numpy.floor(points).astype(int)
		
		y = numpy.zeros(duration)
	
		y[indices] = (indices + 1) - points
		y[indices + 1] = points - indices
		
		# model the sound source (periodic glottal excitation)  
		a = numpy.exp(-250. * 2 * numpy.pi / samplerate)
		y = scipy.signal.lfilter([1],[1, 0, -a * a], y.copy())
		
		# model the filtering by the vocal tract
		bandwidth = 50.
		
		for f in formants:
			cft = f / samplerate
			q = f / bandwidth
			rho = numpy.exp(-numpy.pi * cft / q)
			theta = 2 * numpy.pi * cft * numpy.sqrt(1 - 1/(4.0 * q * q))
			a2 = -2 * rho * numpy.cos(theta)
			a3 = rho * rho
			y = scipy.signal.lfilter([1 + a2 + a3], [1, a2, a3], y.copy()) 
		
		#normalize sound
		data = y / numpy.max(numpy.abs(y), axis=0)        
		data.shape = (data.size, 1)
		return Sound(numpy.tile(data, (nchannels, 1)),  samplerate=samplerate)

	@staticmethod
	def sequence(*args, **kwds):
		'''
		Returns the sequence of sounds in the list sounds joined together
		'''
		samplerate = kwds.pop('samplerate', None)
		if len(kwds):
			raise TypeError('Unexpected keywords to function sequence()')
		sounds = []
		for arg in args:
			if isinstance(arg, (list, tuple)):
				sounds.extend(arg)
			else:
				sounds.append(arg)
		if samplerate is None:
			samplerate = max(s.samplerate for s in sounds)
			rates = numpy.unique([int(s.samplerate) for s in sounds])
			if len(rates)>1:
				sounds = tuple(s.resample(samplerate) for s in sounds)
		x = numpy.vstack(sounds)
		return Sound(x, samplerate)

	def write(self, filename, normalise=False):
		'''
		Save the sound as a WAV.
		
		If the normalise keyword is set to True, the amplitude of the sound will be
		normalised to 1.
		'''
		ext = filename.split('.')[-1].lower()
		if ext=='wav':
			import wave as sndmodule
		else:
			raise NotImplementedError('Can only save as wav soundfiles')
		w = sndmodule.open(filename, 'wb')
		w.setnchannels(self.nchannels)
		w.setsampwidth(2)
		w.setframerate(int(self.samplerate))
		x = numpy.array(self,copy=True)
		am= numpy.amax(x)
		z = numpy.zeros(x.shape[0]*self.nchannels, dtype='int16')
		x.shape=(x.shape[0],self.nchannels)
		for i in range(self.nchannels):
			if normalise:
				x[:,i] /= am
			x[:,i] = (x[:,i]) * 2 ** 15
			z[i::self.nchannels] = x[::1,i]
		data = numpy.array(z, dtype='int16')
		data = array.array('h', data)
		w.writeframes(data.tobytes())
		w.close()
	
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
		nchannels, sampwidth, framerate, nframes, comptype, compname = wav.getparams()
		frames = wav.readframes(nframes * nchannels)
		out = numpy.frombuffer(frames, dtype=numpy.dtype('h'))
		data = numpy.zeros((nframes, nchannels))
		for i in range(nchannels):
			data[:, i] = out[i::nchannels]
			data[:, i] /= 2**15
		return Sound(data, samplerate=framerate)

	def __repr__(self):
		arrayrep = repr(numpy.asarray(self))
		arrayrep = '\n'.join('    '+l for l in arrayrep.split('\n'))
		return 'Sound(\n'+arrayrep+',\n    '+repr(self.samplerate)+')'
	
	def __str__(self):
		return f'Sound duration {self.duration}, channels {self.nchannels}, samplerate {self.samplerate}'

	def __reduce__(self):
		return (_load_Sound_from_pickle, (numpy.asarray(self), float(self.samplerate)))


def _load_Sound_from_pickle(arr, samplerate):
	return Sound(arr, samplerate=samplerate)

def play(*sounds, **kwds):
	'''
	Plays a sound or sequence of sounds. For example::
	
		play(sound)
		play(sound1, sound2)
		play([sound1, sound2, sound3])
		
	If ``sleep=True`` the function will wait
	until the sounds have finished playing before returning.
	'''
	sleep = kwds.pop('sleep', False)
	if len(kwds):
		raise TypeError('Unexpected keyword arguments to function play()')
	sound = Sound.sequence(*sounds)
	sound.play(sleep=sleep)
play.__doc__ = Sound.play.__doc__

def writesound(sound, filename):
	sound.write(filename)
writesound.__doc__ = Sound.write.__doc__

def get_duration(duration,samplerate):
	if not isinstance(duration, int):
		duration = int(numpy.rint(duration * samplerate)) # would be faster to use math.ceil here
	return duration


if __name__ == '__main__':
	vowel_a = Sound.vowel(vowel='a', pitch=110, duration=.5, samplerate=8000)
	print(vowel_a)
	vowel_a == vowel_a # most overloaded methods don't work interactively for some reason
