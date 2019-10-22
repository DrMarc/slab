import array
import time
import pathlib
import numpy
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
try:
	import scipy.signal
	have_scipy = True
except ImportError:
	have_scipy = False
try:
	import matplotlib
	import matplotlib.pyplot as plt
	have_pyplot = True
except ImportError:
	have_pyplot = False

from slab.signals import Signal
from slab.filter import Filter
from slab import DATAPATH

try: # try getting a previously set calibration intensity from file
	_calibration_intensity = numpy.load(DATAPATH + 'calibration_intensity.npy')
except FileNotFoundError:
	_calibration_intensity = 0 # dB, difference between rms intensity and measured output intensity


class Sound(Signal):
	# TODO: debug dynamicripple, add ability to get output of different stages of an auditory periphery model from a sound
	# add other own stim functions (babbelnoise)?
	# add generator for sliding window processing
	'''
	Class for working with sounds, including loading/saving, manipulating and playing.
	Examples:
	>>> import slab
	>>> import numpy
	>>> print(slab.Sound(numpy.ones([10,2]),samplerate=10))
	<class 'slab.sound.Sound'> duration 1.0, samples 10, channels 2, samplerate 10
	>>> print(slab.Sound(numpy.ones([10,2]),samplerate=10).channel(0))
	<class 'slab.sound.Sound'> duration 1.0, samples 10, channels 1, samplerate 10

	** Properties**
	Level:
	>>> sig = slab.Sound.tone()
	>>> sig.level = 80
	>>> sig.level
	80.0

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
		<class 'slab.sound.Sound'> duration 1.0, samples 8000, channels 1, samplerate 8000

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
	- sig.ramp(when='both', duration=0.01, envelope=None)

	**Plotting**
	Examples:
		>>> vowel = slab.Sound.vowel(vowel='a', duration=.5, samplerate=8000)
		>>> vowel.ramp()
		>>> vowel.spectrogram(dyn_range = 50)
		>>> vowel.spectrum(low=100, high=4000, log_power=True)
		>>> vowel.waveform(start=0, end=.1)
	'''
	# instance properties
	def _get_level(self):
		'''
		Returns level in dB SPL (RMS) assuming array is in Pascals.
		In the case of multi-channel sounds, returns an array of levels
		for each channel, otherwise returns a float.
		'''
		if self.nchannels == 1:
			rms_value = numpy.sqrt(numpy.mean(numpy.square(self.data-numpy.mean(self.data))))
			if rms_value == 0:
				rms_dB = 0
			else:
				rms_dB = 20.0*numpy.log10(rms_value/2e-5)
			return rms_dB + _calibration_intensity
		else:
			chans = self.channels()
			levels = [c.level for c in chans]
			return numpy.array(levels)

	def _set_level(self, level):
		'''
		Sets level in dB SPL (RMS) assuming array is in Pascals. ``level``
		should be a value in dB, or a tuple of levels, one for each channel.
		'''
		rms_dB = self._get_level()
		if self.nchannels > 1:
			level = numpy.array(level)
			if level.size == 1:
				level = level.repeat(self.nchannels)
			level = numpy.reshape(level, (1, self.nchannels))
			rms_dB = numpy.reshape(rms_dB, (1, self.nchannels))
		level -= _calibration_intensity # account for calibration intensity
		gain = 10**((level-rms_dB)/20.)
		self.data *= gain

	level = property(fget=_get_level, fset=_set_level, doc='''
		Can be used to get or set the level of a sound, which should be in dB.
		For single channel sounds a value in dB is used, for multiple channel
		sounds a value in dB can be used for setting the level (all channels
		will be set to the same level), or a list/tuple/array of levels. It
		is assumed that the unit of the sound is Pascals.
		''')

	def __init__(self, data, samplerate=None):
		if isinstance(data, pathlib.Path): # Sound initialization from a file name (pathlib object)
			data = str(data)
		if isinstance(data, str): # Sound initialization from a file name (string)
			if samplerate is not None:
				raise ValueError('Cannot specify samplerate when initialising Sound from a file.')
			_ = Sound.read(data)
			self.data = _.data
			self.samplerate = _.samplerate
		else:
			# delegate to the baseclass init
			super().__init__(data, samplerate)

	# static methods (creating sounds)
	@staticmethod
	def read(filename):
		'''
		Load the file given by filename (wav) and returns a Sound object.
		'''
		if not have_soundfile:
			raise ImportError('You need SoundFile to read files (pip install git+https://github.com/bastibe/SoundFile.git')
		data, samplerate = soundfile.read(filename)
		return Sound(data, samplerate=samplerate)

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
		duration = Sound.in_samples(duration, samplerate)
		frequency = numpy.array(frequency)
		phase = numpy.array(phase)
		if frequency.size > nchannels and nchannels == 1:
			nchannels = frequency.size
		if phase.size > nchannels and nchannels == 1:
			nchannels = phase.size
		if frequency.size == nchannels:
			frequency.shape = (1, nchannels)
		if phase.size == nchannels:
			phase.shape = (1, nchannels)
		t = numpy.arange(0, duration, 1)/samplerate
		t.shape = (t.size, 1) # ensures C-order
		x = numpy.sin(phase + 2*numpy.pi * frequency * numpy.tile(t, (1, nchannels)))
		return Sound(x, samplerate)

	@staticmethod
	def harmoniccomplex(f0=500, duration=1., amplitude=0, phase=0, samplerate=None, nchannels=1):
		'''
		Returns a harmonic complex composed of pure tones at integer multiples
		of the fundamental frequency ``f0``.
		The ``amplitude`` and ``phase`` keywords can be set to either a single
		value or an array of values. In the former case the value is set for all
		harmonics, and harmonics up to the sampling frequency are
		generated. In the latter each harmonic parameter is set
		separately, and the number of harmonics generated corresponds
		to the length of the array. Amplitudes are relateve to full scale
		(i.e. 0 corresponds to maximum intensity; -30 would be 30 dB softer).
		Example:
		>>> sig = Sound.harmoniccomplex(f0=200, amplitude=[0,-10,-20,-30])
		>>> _ = sig.spectrum()

		'''
		samplerate = Sound.get_samplerate(samplerate)
		phases = numpy.array(phase).flatten()
		amplitudes = numpy.array(amplitude).flatten()
		if len(phases) > 1 or len(amplitudes) > 1:
			if (len(phases) > 1 and len(amplitudes) > 1) and (len(phases) != len(amplitudes)):
				raise ValueError('Please specify the same number of phases and amplitudes')
			Nharmonics = max(len(phases), len(amplitudes))
		else:
			Nharmonics = int(numpy.floor(samplerate/(2*f0)))
		if len(phases) == 1:
			phases = numpy.tile(phase, Nharmonics)
		if len(amplitudes) == 1:
			amplitudes = numpy.tile(amplitude, Nharmonics)
		out = Sound.tone(f0, duration, phase=phases[0], samplerate=samplerate, nchannels=nchannels)
		lvl = out.level
		out.level += amplitudes[0]
		for i in range(1, Nharmonics):
			tmp = Sound.tone(frequency=(i+1)*f0, duration=duration, phase=phases[i], samplerate=samplerate, nchannels=nchannels)
			tmp.level = lvl + amplitudes[i]
			out += tmp
		return out

	@staticmethod
	def whitenoise(duration=1.0, samplerate=None, nchannels=1, normalise=True):
		'''
		Returns a white noise. If the samplerate is not specified, the global
		default value will be used. nchannels = 2 produces uncorrelated noise (dichotic).
		>>> noise = Sound.whitenoise(1.0,nchannels=2)

		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		x = numpy.random.randn(duration, nchannels)
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
								   nchannels=nchannels, normalise=normalise)

	@staticmethod
	def brownnoise(duration=1.0, samplerate=None, nchannels=1, normalise=True):
		'Returns brown noise, i.e :func:`powerlawnoise` with alpha=2'
		return Sound.powerlawnoise(duration, 2.0, samplerate=samplerate,
								   nchannels=nchannels, normalise=normalise)

	@staticmethod
	def irn(delay=0.01, gain=1, niter=16, duration=1.0, samplerate=None): # TODO: produces no pitch!
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
	def chirp(duration=1.0, from_freq=100, to_freq=None, samplerate=None, kind='quadratic'):
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		t = numpy.arange(0, duration, 1) / samplerate # generate a time vector
		t.shape = (t.size, 1) # ensures C-order
		if not to_freq:
			to_freq = samplerate / 2
		chirp = scipy.signal.chirp(t, from_freq, t[-1], to_freq, method=kind, vertex_zero=True)
		return Sound(chirp, samplerate=samplerate)

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
		vowel.filter(f=0.75*samplerate/2, kind='lp')
		return vowel

	@staticmethod
	def multitone_masker(duration=1.0, f_lower=125, f_upper=4000, bandwidth=1/3, samplerate=None):
		'''
		Returns a noise made of ERB-spaced random-phase sinetones in the band
		between f_lower and f_upper. This noise does not have random amplitude
		variations and is useful for testing CI patients. See Oxenham 2014, Trends Hear.
		Example:
		>>> sig = Sound.multitone_masker()
		>>> sig.ramp()
		>>> _ = sig.spectrum()
		'''
		samplerate = Sound.get_samplerate(samplerate)
		duration = Sound.in_samples(duration, samplerate)
		# get centre_freqs
		freqs, _, _ = Filter._center_freqs(low_lim=f_lower, hi_lim=f_upper, bandwidth=bandwidth)
		rand_phases = numpy.random.rand(len(freqs)) * 2 * numpy.pi
		sig = Sound.tone(frequency=freqs, duration=duration, phase=rand_phases, samplerate=samplerate)
		# collapse across channels
		data = numpy.sum(sig.data, axis=1) / len(freqs)
		return Sound(data, samplerate=samplerate)

	@staticmethod
	def erb_noise(duration=1.0, f_lower=125, f_upper=4000, samplerate=None):
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
		Converted to python by Jessica Thompson based on Jonathan Simon and Didier Dipereux's matlab program [ripfft.m], based on Jian Lin's C program [rip.c].
		Example:
		>>> ripple = Sound.dynamicripple()

		'''
		samplerate = Sound.get_samplerate(samplerate)
		#duration = Sound.in_samples(duration, samplerate) # TODO: duration has to be in seconds for this function!
		duration = Sound.in_samples(duration, samplerate)
		Ri = Rt*duration # modulation lag, in number of df's
		if RO:  #compute tones freqs
			R1 = numpy.round(2**(numpy.array(-1, 1)*BW/2)*f0/df)
			fr = df*(numpy.arange(R1[1], R1[2]))
		else:   #compute log-spaced tones freqs
			R1 = numpy.round(BW/2/df)
			fr = f0*2**(numpy.arange(-R1, R1+1)*df)
		M = len(fr) # of component
		S = 0j*numpy.zeros((int(duration*samplerate/2), 1)) # memory allocation
		fdx = int(numpy.round(fr*duration)+1) # freq. index MATLAB INDEXING??? -> not an index!!!
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
		idx = numpy.max(maxidx,axis=1).astype(int)
		S[idx] = S[idx] +numpy.dot(r, numpy.exp(1j*(th-ph))) # lower side
		minidx = numpy.ones((M,2))
		minidx[:,0] = fdx+Ri
		minidx[:,1] = minidx[:,1] * duration*samplerate/2
		idx = numpy.min(minidx, axis=1).astype(int)
		S[idx] = S[idx] + numpy.dot(r, numpy.exp(1j*(th+ph))) # upper side
		S = S * Am/2
		S[0] = 0
		S = S[:duration*samplerate/2]
		# original stationary spectrum
		S[fdx] = S[fdx] + r*numpy.exp(1j*th) # moved here to save computation
		# time waveform
		s = numpy.fft.ifft(numpy.concatenate((S, [0], numpy.flipud(S[1:duration*samplerate/2]).conj()))) # make it double side
		s = s[:int(numpy.round(duration*samplerate))].real # only real part is good. *2 was ignored
		return Sound(s, samplerate)

	@staticmethod
	def sequence(*sounds):
		'Joins the sounds in the list ''sounds'' into a new sound object.'
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
		if bool(self.samplerate%1):
			self = self.resample(int(self.samplerate))
		if not have_soundfile:
			raise ImportError('You need SoundFile to write files (pip install git+https://github.com/bastibe/SoundFile.git')
		if isinstance(filename, pathlib.Path):
			filename = str(filename)
		if normalise:
			self.data /= numpy.amax(self.data)
		soundfile.write(filename, self.data, self.samplerate)

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
		if when in ('onset', 'both'):
			self.data[:sz, :] *= multiplier
		if when in ('offset', 'both'):
			self.data[self.nsamples-sz:, :] *= multiplier[::-1]

	def repeat(self, n):
		'Repeats the sound n times.'
		self.data = numpy.vstack((self.data,)*int(n))

	def copychannel(self, n):
		'''Copies a single-channel sound inplace to make an n-channel sound.
		If a multi-channel sound is supplied, all channels except the first are silently dropped.'''
		self.data = numpy.repeat(self.channel(0), n, axis=1)

	@staticmethod
	def crossfade(sound1, sound2, overlap=0.01):
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
		envelope = envelope[:, None] # add an empty axis to get to the same shape as self.data: (n_samples, 1)
		self.data *= numpy.broadcast_to(envelope, self.data.shape) # if data is 2D (>1 channel) broadcase the envelope to fit

	def filter(self, f=100, kind='hp'):
		'''
		Filters a sound in place. This is a convenience function to avoid calling
		the Filter class.
		f: edge frequency in Hz (*100*) or tuple of frequencies for bp and notch.
		type: 'lp', *'hp'*, bp, 'notch'
		Examples:
		>>> sig = Sound.whitenoise()
		>>> sig.filter(f=3000, kind='lp')
		>>> _ = sig.spectrum()
		'''
		#n = 2**((self.nsamples-1).bit_length()-1) + 1 # next smaller power of 2, +1 because odd needed
		n = min(1000, self.nsamples)
		filt = Filter.rectangular_filter(frequency=f, kind=kind, samplerate=self.samplerate, length=n)
		self.data = filt.apply(self).data

	def aweight(self):
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
		data_chans = []
		for chan in self.channels():
			data = scipy.signal.lfilter(b, a, chan.data.flatten())
			data_chans.append(data) # concatenate channel data
		return Sound(data_chans, self.samplerate)

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
			#raise NotImplementedError('Need module soundcard for cross-platform playing (pip install git+https://github.com/bastibe/SoundCard).')
			self.write('tmp.wav')
			Sound.play_file('tmp.wav')
		if sleep:
			time.sleep(self.duration)

	@staticmethod
	def play_file(fname):
		from platform import system
		system = system()
		if system == 'Windows':
			import winsound
			winsound.PlaySound('%s.wav' % fname, winsound.SND_FILENAME)
		elif system == 'Darwin':
			import subprocess
			subprocess.call(['afplay', fname])
		else:  # Linux/Unix, install sox (sudo apt-get install sox libsox-fmt-all)
			import subprocess
			try:
				subprocess.call(['play', fname])
			except:
				raise NotImplementedError('Need sox for playing from files on Linux. Install: sudo apt-get install sox libsox-fmt-all')

	def waveform(self, start=0, end=None):
		'''
		Plots the waveform of the sound.
		Arguments:
		``start``, ``end`` (samples or time)
		If these are left unspecified, it shows the full waveform
		'''
		start = self.in_samples(start, self.samplerate)
		if end is None:
			end = self.nsamples
		end = self.in_samples(end, self.samplerate)
		for i in range(self.nchannels):
			if i == 0:
				plt.plot(self.times[start:end], self.channel(i)[start:end], label='left')
			elif i == 1:
				plt.plot(self.times[start:end], self.channel(i)[start:end], label='right')
			else:
				plt.plot(self.times[start:end], self.channel(i)[start:end])
		plt.title('Waveform')
		plt.xlabel('Time [sec]')
		plt.ylabel('Amplitude')
		plt.legend()
		plt.show()

	## features ##
	def spectrogram(self, window_dur=0.005, dyn_range=120, other=None, plot=True):
		'''
		Plots a spectrogram of the sound
		Arguments:
		``window_dur``
			Duration of time window for short-term FFT (*0.005sec*)
		``dyn_range``
			Dynamic range in dB to plot (*120*)
		```other```
			If a sound object is given, subtract the waveform and plot the difference spectrogram.
		If plot is False, returns the values returned by scipy.signal's ``spectrogram``, namely
		``freqs, times, power`` where ``power`` is a 2D array of powers,
		``freqs`` is the corresponding frequencies, and ``times`` are the time bins.
		'''
		if self.nchannels > 1:
			raise ValueError('Can only compute spectrograms for mono sounds.')
		if other is not None:
			x = self.data.flatten() - other.data.flatten()
		else:
			x = self.data.flatten()
		# set default for step_dur optimal for Gaussian windows.
		step_dur = window_dur/numpy.sqrt(numpy.pi)/8
		# convert window & step durations from seconds to numbers of samples
		window_nsamp = Sound.in_samples(window_dur, self.samplerate) * 2
		step_nsamp = Sound.in_samples(step_dur, self.samplerate)
		# make the window. A Gaussian filter needs a minimum of 6σ - 1 samples, so working
		# backward from window_nsamp we can calculate σ.
		window_sigma = (window_nsamp+1)/6
		window = scipy.signal.windows.gaussian(window_nsamp, window_sigma)
		# convert step size into number of overlapping samples in adjacent analysis frames
		noverlap = window_nsamp - step_nsamp
		# compute the power spectral density
		freqs, times, power = scipy.signal.spectrogram(x, mode='psd', fs=self.samplerate, scaling='density', noverlap=noverlap, window=window, nperseg=window_nsamp)
		if plot:
			p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
			power = 10 * numpy.log10(power / (p_ref ** 2))  # logarithmic power for plotting
			# set lower bound of colormap (vmin) from dynamic range.
			dB_max = power.max()
			vmin = dB_max-dyn_range
			fig, ax = plt.subplots()
			cmap = matplotlib.cm.get_cmap('Greys')
			extent = (times.min(), times.max(), freqs.min(), freqs.max())
			ax.imshow(power, origin='lower', aspect='auto', cmap=cmap, extent=extent, vmin=vmin, vmax=None)
			plt.title('Spectrogram')
			plt.xlabel('Time [sec]')
			plt.ylabel('Frequency [Hz]')
			plt.show()
		else:
			return freqs, times, power

	def cochleagram(self, bandwidth=1/5, plot=True):
		'''
		Computes a cochleagram of the sound by filtering with a bank of erb-spaced
		filters cos filter with given bandwidth (*1/5* th octave) and applying a
		cube-root compression to the resulting envelopes.
		If plot is False, returns the envelopes.
		''' # TODO: freq axis is incorrect (lin instead of erb)
		fbank = Filter.cos_filterbank(bandwidth=bandwidth, low_lim=20, hi_lim=None, samplerate=self.samplerate)
		freqs = fbank.filter_bank_center_freqs()
		subbands = fbank.apply(self.channel(0))
		envs = subbands.envelope()
		envs.data[envs.data<1e-9] = 0 # remove small values that cause waring with numpy.power
		envs = envs.data ** (1/3) # apply non-linearity (cube-root compression)
		if plot:
			cmap = matplotlib.cm.get_cmap('Greys')
			_, ax = plt.subplots()
			ax.imshow(envs.T, origin='lower', aspect='auto', cmap=cmap)
			labels = list(freqs.astype(int))
			ax.yaxis.set_major_formatter(matplotlib.ticker.IndexFormatter(labels)) # centre frequencies as ticks
			ax.set_xlim([0, self.duration])
			plt.title('Cochleagram')
			plt.xlabel('Time [sec]')
			plt.ylabel('Frequency [Hz]')
			plt.show()
		else:
			return envs

	def spectrum(self, low=16, high=None, log_power=True, plot=True):
		'''
		Returns the spectrum of the sound and optionally plots it.
		Arguments:
		``low``, ``high``
			If these are left unspecified, it shows the full spectrum,
			otherwise it shows only between ``low`` and ``high`` in Hz.
		``log_power=True``
			If True it returns the log of the power.
		``plot=True``
			Whether to plot the output.
		If plot=False, returns ``Z, freqs``
		where ``Z`` is a 1D array of powers and ``freqs`` is the corresponding
		frequencies.
		'''
		freqs = numpy.fft.rfftfreq(self.nsamples, d=1/self.samplerate)
		sig_rfft = numpy.zeros((len(freqs),self.nchannels))
		for chan in range(self.nchannels):
			sig_rfft[:,chan] = numpy.abs(numpy.fft.rfft(self.data[:,chan], axis=0))
		pxx = sig_rfft/len(freqs) # scale by the number of points so that the magnitude does not depend on the length of the signal
		pxx = pxx**2 # square to get the power
		if low is not None or high is not None:
			if low is None:
				low = 0
			if high is None:
				high = numpy.amax(freqs)
			I = numpy.logical_and(low <= freqs, freqs <= high)
			I2 = numpy.where(I)[0]
			Z = pxx[I2,:]
			freqs = freqs[I2]
		else:
			Z = pxx
		if log_power:
			Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
			Z = 10 * numpy.log10(Z)
		if plot:
			plt.subplot(111)
			plt.semilogx(freqs, Z)
			ticks_freqs = numpy.round(32000 * 2 ** (numpy.arange(12, dtype=float)*-1))
			plt.xticks(ticks_freqs, map(str, ticks_freqs.astype(int)))
			plt.grid()
			plt.xlim((freqs[1], freqs[-1]))
			plt.ylabel('Power [dB/Hz]') if log_power else plt.ylabel('Power')
			plt.title('Spectrum')
			plt.show()
		else:
			return Z, freqs

	def spectral_feature(self, feature='centroid', mean='rms', frame_duration=None, rolloff=0.85):
		'''
		Computes one of several features of the spectrogram of a sound and returns either a
		new Signal with the feature value at each sample, or the average (*rms* or mean) feature value over all samples.
		Available features:
		*centroid* is the centre of mass of the short-term spectrum, and fwhm is the width of a Gaussian of the same variance as the spectrum around the centroid.
		Examples:
		>>> sig = slab.Sound.tone(frequency=500, nchannels=2)
		>>> sig.spectral_feature(feature='centroid')
		500.

		'flux' is a measure of how quickly the power spectrum of a signal is changing, calculated by comparing the power spectrum for one frame against the power spectrum from the previous frame. Returns the root-mean-square over the entire stimulus of the change in power spectrum between adjacent time windows, measured as Euclidean distance.
		>>> sig = slab.Sound.tone()
		>>> numpy.testing.assert_allclose(sig.spectral_feature(feature='flux'), desired=0, atol=1e-04)

		'flatness' measures how tone-like a sound is, as opposed to being noise-like.
		It is calculated by dividing the geometric mean of the power spectrum by the arithmetic mean. (Dubnov, Shlomo  "Generalization of spectral flatness measure for non-gaussian linear processes" IEEE Signal Processing Letters, 2004, Vol. 11.)

		'rolloff' is...

		'''
		if not frame_duration:
			if mean is not None:
				frame_duration = int(self.nsamples/2) # long frames if not averaging
			else:
				frame_duration = 0.05 # 50ms frames by default
		out_all = []
		for chan in self.channels():
			freqs, times, power = chan.spectrogram(window_dur=frame_duration, plot=False)
			norm = power / power.sum(axis=0, keepdims=True) # normalize successive frames
			if feature == 'centroid':
				out = numpy.sum(freqs[:,numpy.newaxis] * norm, axis=0)
			elif feature == 'fwhm':
				cog = numpy.sum(freqs[:,numpy.newaxis] * norm, axis=0)
				sq_dist_from_cog = (freqs[:,numpy.newaxis] - cog[numpy.newaxis,:]) ** 2
				sigma = numpy.sqrt(numpy.sum(sq_dist_from_cog * norm, axis=0))
				out = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma
			elif feature == 'flux':
				norm = numpy.c_[norm[:, 0], norm] # duplicate first frame to give 0 diff
				delta_p = numpy.diff(norm, axis=1) # diff now has same shape as norm
				out = numpy.sqrt((delta_p**2).sum(axis=0)) / power.shape[0]
			elif feature == 'rolloff':
				cum = numpy.cumsum(norm, axis=0)
				rolloff_idx = numpy.argmax(cum >= rolloff, axis=0)
				out = freqs[rolloff_idx] # convert from index to Hz
			elif feature == 'flatness':
				norm[norm == 0] = 1
				gmean = numpy.exp(numpy.log(power + 1e-20).mean(axis=0))
				amean = power.sum(axis=0) / power.shape[0]
				out = gmean / amean
			else:
				raise ValueError('Unknown feature name.')
			if mean is None:
				out = numpy.interp(self.times, times, out) # interpolate to sound samples
			elif mean == 'rms':
				out = numpy.sqrt(numpy.mean(out**2)) # average feature time series
			elif mean == 'average':
				out = out.mean()
			out_all.append(out) # concatenate channel data
		if mean is None:
			out_all = Signal(data=out_all, samplerate=self.samplerate) # cast as Signal
		return out_all

	def pitch_tracking(self, window_dur=0.005):
		fbank = Filter.cos_filterbank(length=self.nsamples, bandwidth=1/10, low_lim=30, hi_lim=None, samplerate=self.samplerate)
		self_bands = fbank.apply(self) # apply filterbank
		envs = numpy.abs(scipy.signal.hilbert(self.data, axis=0)) # get envelope
		# X[X < 0] = 0
		# smooth the results with a moving average filter
		# autocorrelation in each band and frame
		# for k in range(0, iNumBands):
		# 	# get current block
		# 	if X[k, np.arange(i_start, i_stop + 1)].sum() < 1e-20:
		# 		continue
		# 	else:
		# 		x_tmp[np.arange(0, i_stop - i_start + 1)] = X[k, np.arange(i_start, i_stop + 1)]
		# 	afCorr = np.correlate(x_tmp, x_tmp, "full") / np.dot(x_tmp, x_tmp)
		# 	# aggregate bands with simple sum before peak picking
		# 	afSumCorr += afCorr[np.arange(iBlockLength, afCorr.size)]
		# if afSumCorr.sum() < 1e-20:
		# 	continue
		# # find the highest local maximum
		# iPeaks = find_peaks(afSumCorr, height=0)
		# if iPeaks[0].size:
		# 	eta_min = np.max([eta_min, iPeaks[0][0] - 1])
		# f[n] = np.argmax(afSumCorr[np.arange(eta_min + 1, afSumCorr.size)]) + 1
		# # convert to Hz
		# f[n] = f_s / (f[n] + eta_min + 1)
		pass

	def crest_factor(self):
		'''
		The crest factor is the ratio of the peak amplitude and the RMS value of a waveform
		and indicates how extreme the peaks in a waveform are. Returns the crest factor in dB.
		Numerically identical to the peak-to-average power ratio.
		'''
		jwd = self.data - numpy.mean(self.data)
		crest = jwd.max() / numpy.sqrt(numpy.mean(numpy.square(jwd)))
		return 20 * numpy.log10(crest)

	def onset_slope(self):
		'''
		Returns the centroid of a histogram of onset slopes as a measure of how many
		quick intensity increases the sound has. These onset-like features make the
		sound easier to localize via envelope ITD.
		'''
		env = self.envelope(kind='dB') # get envelope
		diffs = numpy.diff(env.data, axis=0) * self.samplerate # compute db change per sec
		diffs[diffs < 0] = 0 # keep positive changes (onsets)
		hist, bins = numpy.histogram(diffs, range=(1, diffs.max()), bins=1000) # compute histogram of differences
		bin_centers = (bins[:-1] + bins[1:]) / 2
		norm = hist / hist.sum() # normalize histogram so that it summs to 1
		return numpy.sum(bin_centers * norm) # compute centroid of histogram

	def time_windows(self, duration=1024):
		'''
		Returns overlapping time windows as a generator.
		Use the generator if you need to modify each segment in place like this
		(this saves you the trouble of crossfading the segments correcly afterwards):
		>>> sig = slab.Sound.tone()
		>>> windows = sig.time_windows()
		>>> win = 1 # dummy value to get started
		>>> send = None
		>>> while win: # get windows one by one
		>>>		win = windows.send(send) # returns each window as Sound object
		>>>		win.ramp() # modify the windowed signal
		>>>		send = win

		If you don't need to write back into the sound
		>>> windows = sig.time_windows()
		>>> for w in windows:
		>>>		w.waveform() # process window here

		'''
		window_nsamp = Sound.in_samples(duration, self.samplerate) * 2
		step_nsamp = numpy.floor(window_nsamp/numpy.sqrt(numpy.pi)/8).astype(int) # step_dur optimal for Gaussian windows
		# make the window. A Gaussian filter needs a minimum of 6σ - 1 samples.
		window_sigma = numpy.ceil((window_nsamp+1)/6)
		window = numpy.tile(scipy.signal.windows.gaussian(window_nsamp, window_sigma), (self.nchannels, 1)).T
		noverlap = window_nsamp - step_nsamp
		# loop through windows, yield each one
		modified_segment = None
		modified = numpy.zeros_like(self.data)
		idx = 0
		while idx < self.nsamples:
			segment = Sound(self.data[idx:min(self.nsamples,idx+window_nsamp),:],  samplerate=self.samplerate)
			segment.resize(window_nsamp) # in case the last window is too short
			segment *= window
			modified_segment = yield segment # return a new sound object
			if modified_segment:
				assert modified_segment.data.shape == segment.data.shape
				end_idx = min(modified.shape[0], idx+window_nsamp)
				modified[idx:end_idx,:] += modified_segment.data[:end_idx-idx,:]
			idx += step_nsamp
		if numpy.any(modified):
			self.data = modified

	@staticmethod
	def calibrate(intensity=None):
		'''
		Calibrate the presentation intensity of a setup.
		Enter the calibration intensity, if you know it.
		If None, plays a 1kHz tone. Please measure actual
		intensity with a sound level meter and appropriate
		coupler.
		'''
		global _calibration_intensity
		if not intensity:
			tone = Sound.tone(duration=5.0, frequency=1000) # make 1kHz tone
			print('Playing 1kHz test tone for 5 seconds. Please measure intensity.')
			tone.play() # play it
			intensity = input('Enter measured intensity in dB: ') # ask for measured intesnity
			intensity = intensity - tone.level # subtract measured from rms intensity
		# set and save
		_calibration_intensity = intensity
		Sound.save_calibration_intensity()

	@staticmethod
	def save_calibration_intensity():
		'''
		Saves the calibration intensity value to a file in the data folder.
		This file is loaded upon import of the toolbox.
		'''
		numpy.save(DATAPATH + 'calibration_intensity.npy', _calibration_intensity)

def apply_to_path(path, method, kwargs={}, out_path=None):
	'''
	Apply a function to all wav files in a directory. This function need to take a Sound
	as first argument. Other a arguments are passed as dictionary of keyword arguments and values. If out_path is supplied, then the sounds are saved with their original file name in the new path/directory.
	Example:
	>> slab.apply_to_path('.', slab.Sound.spectral_feature, {'feature':'fwhm'})
	>> slab.apply_to_path('.', slab.Sound.ramp, out_path='modified')
	>> slab.apply_to_path('.', slab.Sound.ramp, kwargs={'duration':0.3}, out_path='./test')
	'''
	if not callable(method):
		raise ValueError('Method must be callable.')
	if isinstance(path, str):
		path = pathlib.Path(path)
	if isinstance(out_path, str):
		out_path = pathlib.Path(out_path)
	files = path.glob('*.wav')
	results = dict()
	for file in files:
		sig = Sound(file)
		res = method(sig, **kwargs)
		if out_path:
			if hasattr(res, 'write'): # if objects with write methods were returned, write them to sq_dist_from_cog
				res.write(out_path.joinpath(file.name))
			else: # otherwise assum the modification was in-place and write sig
				sig.write(out_path.joinpath(file.name))
		results[str(file.stem)] = res
	return results # a dictionary of results for each file name

if __name__ == '__main__':
	sig1 = Sound.harmoniccomplex()
	lev = sig1.level
	sig1.filter((500, 1000), kind='bp')
	sig1.log_spectrogram()
	sig2 = Sound.clicktrain()
	sig2.level = 60
	sig3 = Sound.crossfade(sig1, sig2, overlap=0.5)
	sig3.play()
	sig3.waveform()
